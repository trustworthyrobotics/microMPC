"""Nominal condensed Koopman MPC with SQP-linearized obstacle avoidance.

Each SQP iteration
replaces the spherical keep-out set by tangent halfspaces along the current
nominal trajectory, then solves the resulting control-condensed QP with PD.
The obstacle is not part of the simulated dynamics; collision checks are
therefore geometric only.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

from . import planning as base


NP_DTYPE, JAX_DTYPE = np.float32, jnp.float32
OBSTACLE_CENTER = np.full(3, 2.5, dtype=NP_DTYPE)
OBSTACLE_RADIUS = 1.25
MODEL_ERROR_BUFFER = 0.15
SQP_ITERATIONS = 2
PD_MODE = "converged"
PD_PRIMAL_TOL = 1e-2
PD_FIXED_POINT_TOL = 1e-2
PD_STEP_SAFETY = 0.99
PD_EXTRAPOLATION = 1.0


class ObstaclePrimalDualMPC(base.PrimalDualMPC):
    """Control-condensed QP with matrix-only PD iterations inside SQP.

    The compiled solve accepts and returns explicit warm state for batching.
    """

    def __init__(
        self,
        *args,
        obstacle_center=OBSTACLE_CENTER,
        obstacle_radius=OBSTACLE_RADIUS,
        model_error_buffer=MODEL_ERROR_BUFFER,
        sqp_iterations=SQP_ITERATIONS,
        pd_mode=PD_MODE,
        pd_primal_tol=PD_PRIMAL_TOL,
        pd_fixed_point_tol=PD_FIXED_POINT_TOL,
        pd_step_safety=PD_STEP_SAFETY,
        pd_extrapolation=PD_EXTRAPOLATION,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if self.factorize:
            raise ValueError("PD requires the cached inverse path (FACTORIZE=False)")
        if obstacle_radius <= 0.0 or model_error_buffer < 0.0:
            raise ValueError(
                "Obstacle radius must be positive and buffer must be nonnegative"
            )
        if pd_mode not in {"fixed", "converged"}:
            raise ValueError("solver mode must be 'fixed' or 'converged'")
        if pd_primal_tol <= 0.0 or pd_fixed_point_tol <= 0.0:
            raise ValueError("PD tolerances must be positive")
        if not 0.0 < pd_step_safety < 1.0:
            raise ValueError("pd_step_safety must lie strictly between zero and one")
        if not 0.0 <= pd_extrapolation <= 1.0:
            raise ValueError("pd_extrapolation must lie between zero and one")
        solver_dtype = self.gamma_x.dtype
        self.center = jax.device_put(
            jnp.asarray(obstacle_center, dtype=solver_dtype), self.device
        )
        self.physical_radius = float(obstacle_radius)
        self.buffer = float(model_error_buffer)
        self.effective_radius = float(obstacle_radius + model_error_buffer)
        if sqp_iterations < 1:
            raise ValueError("sqp_iterations must be positive")
        self.sqp_iterations = int(sqp_iterations)
        self.pd_mode = pd_mode
        self.pd_primal_tol = float(pd_primal_tol)
        self.pd_fixed_point_tol = float(pd_fixed_point_tol)
        self.pd_step_safety = float(pd_step_safety)
        self.pd_extrapolation = float(pd_extrapolation)
        self.obstacle_gamma = self.gamma_x[:, :3].reshape(self.H, 3, self.n_controls)
        self.obstacle_base_obs = self.obs_x[:, :3]
        self.obstacle_bias = self.decoder_bias[:3]
        self._solve_jit = jax.jit(self._build_obstacle_solve_impl())
        self.warm_u = self.u_hover_tiled
        # Nonnegative multipliers for upper, lower, and obstacle inequalities.
        self.warm_upper_dual = jnp.zeros_like(self.warm_u)
        self.warm_lower_dual = jnp.zeros_like(self.warm_u)
        self.warm_obstacle_dual = jnp.zeros(self.H, dtype=solver_dtype)
        self.warm_obstacle_aux = jnp.zeros(self.H, dtype=solver_dtype)
        self.warm_reference = None

    def seed_reference(self, start_position, goal_position):
        """Smooth dogleg used only to choose the initial tangent-plane side."""
        dtype = self.center.dtype
        times = jnp.linspace(1.0 / self.H, 1.0, self.H, dtype=dtype)
        direction = goal_position - start_position
        unit = direction / jnp.maximum(jnp.linalg.norm(direction), 1.0e-9)
        z_axis = jnp.array([0.0, 0.0, 1.0], dtype=dtype)
        y_axis = jnp.array([0.0, 1.0, 0.0], dtype=dtype)
        axis = jnp.where(jnp.abs(unit @ z_axis) > 0.9, y_axis, z_axis)
        normal = jnp.cross(unit, axis)
        normal /= jnp.maximum(jnp.linalg.norm(normal), 1.0e-9)
        line = start_position + times[:, None] * direction
        bump = (self.effective_radius + 0.35) * jnp.sin(jnp.pi * times)
        return line + bump[:, None] * normal

    def _build_obstacle_solve_impl(self):
        H, nu = self.H, self.nu
        rho = self.rho
        primal_step = 1.0 / rho
        step_safety = self.pd_step_safety
        extrapolation = self.pd_extrapolation
        center = self.center
        radius = self.effective_radius
        sqp_iterations = self.sqp_iterations
        pd_iterations = self.steps
        pd_mode = self.pd_mode
        feasibility_tolerance = self.pd_primal_tol
        fixed_point_tolerance = self.pd_fixed_point_tol
        a_y_z, g_y, c_y = self.a_y_z, self.g_y, self.c_y
        q_y, r_bar, base_inverse = self.q_y, self.r_bar, self.k_system
        obs_x, gamma_x, decoder_bias = self.obs_x, self.gamma_x, self.decoder_bias
        obstacle_gamma = self.obstacle_gamma
        obstacle_base_obs, obstacle_bias = self.obstacle_base_obs, self.obstacle_bias
        hover = self.u_hover_tiled
        u_min, u_max = self.u_min_tiled, self.u_max_tiled

        def shift_blocks(sequence, width):
            blocks = sequence.reshape(H, width)
            return jnp.concatenate((blocks[1:], blocks[-1:]), axis=0).reshape(-1)

        def solve_impl(
            warm_u,
            warm_upper_dual,
            warm_lower_dual,
            warm_obstacle_dual,
            warm_obstacle_aux,
            reference,
            z0,
            goal_y,
        ):
            error = a_y_z @ z0 + c_y - jnp.tile(goal_y, H)
            gradient = g_y.T @ (q_y @ error) - r_bar @ hover
            base_position = (
                jnp.einsum("hpn,n->hp", obstacle_base_obs, z0) + obstacle_bias
            )

            def sqp_step(_, carry):
                u, upper_dual, lower_dual, obstacle_dual, unused, ref, _ = carry
                difference = ref - center
                norms = jnp.linalg.norm(difference, axis=1, keepdims=True)
                fallback = jnp.broadcast_to(
                    jnp.array([0.0, 1.0, 0.0], dtype=center.dtype), difference.shape
                )
                normals = jnp.where(
                    norms > 1.0e-8, difference / jnp.maximum(norms, 1.0e-8), fallback
                )
                # -n' x_k <= -n' center - radius, x_k = base_k + Gamma_k U.
                obstacle_matrix = -jnp.einsum("hp,hpn->hn", normals, obstacle_gamma)
                obstacle_bound = -(normals @ center + radius) + jnp.einsum(
                    "hp,hp->h", normals, base_position
                )

                # A = [I; -I; A_obs].  The Frobenius norm is a conservative,
                # decomposition-free upper bound on its spectral norm.
                operator_norm_bound_sq = 2.0 * u.size + jnp.sum(obstacle_matrix**2)
                dual_step = step_safety / (primal_step * operator_norm_bound_sq)

                def pd_step(pd_carry):
                    iteration, u_old, u_bar, dual_upper, dual_lower, dual_obs, _, _ = (
                        pd_carry
                    )
                    previous_upper = dual_upper
                    previous_lower = dual_lower
                    previous_obs = dual_obs
                    dual_upper = jnp.maximum(
                        dual_upper + dual_step * (u_bar - u_max), 0.0
                    )
                    dual_lower = jnp.maximum(
                        dual_lower + dual_step * (u_min - u_bar), 0.0
                    )
                    dual_obs = jnp.maximum(
                        dual_obs
                        + dual_step * (obstacle_matrix @ u_bar - obstacle_bound),
                        0.0,
                    )

                    # prox_{tau f}(v) for f(U)=.5 U'HU+gradient'U.  Since
                    # tau=1/rho, rho*(H+rho I)^-1 is the cached proximal map.
                    rhs = u_old - primal_step * (
                        gradient
                        + dual_upper
                        - dual_lower
                        + obstacle_matrix.T @ dual_obs
                    )
                    u_new = rho * (base_inverse @ rhs)
                    u_bar_new = u_new + extrapolation * (u_new - u_old)
                    obstacle_values = obstacle_matrix @ u_new
                    feasibility = jnp.sqrt(
                        jnp.sum(jnp.maximum(u_new - u_max, 0.0) ** 2)
                        + jnp.sum(jnp.maximum(u_min - u_new, 0.0) ** 2)
                        + jnp.sum(
                            jnp.maximum(obstacle_values - obstacle_bound, 0.0) ** 2
                        )
                    )
                    fixed_point = jnp.sqrt(
                        jnp.sum(((u_new - u_old) / primal_step) ** 2)
                        + jnp.sum(((dual_upper - previous_upper) / dual_step) ** 2)
                        + jnp.sum(((dual_lower - previous_lower) / dual_step) ** 2)
                        + jnp.sum(((dual_obs - previous_obs) / dual_step) ** 2)
                    )
                    return (
                        iteration + 1,
                        u_new,
                        u_bar_new,
                        dual_upper,
                        dual_lower,
                        dual_obs,
                        feasibility,
                        fixed_point,
                    )

                initial_pd = (
                    jnp.asarray(0, dtype=jnp.int32),
                    u,
                    u,
                    upper_dual,
                    lower_dual,
                    obstacle_dual,
                    jnp.asarray(jnp.inf, dtype=u.dtype),
                    jnp.asarray(jnp.inf, dtype=u.dtype),
                )
                if pd_mode == "fixed":
                    pd_result = lax.fori_loop(
                        0,
                        pd_iterations,
                        lambda _, value: pd_step(value),
                        initial_pd,
                    )
                else:

                    def not_converged(value):
                        iteration, _, _, _, _, _, feasibility, fixed_point = value
                        residual_too_large = (feasibility > feasibility_tolerance) | (
                            fixed_point > fixed_point_tolerance
                        )
                        return (iteration < pd_iterations) & residual_too_large

                    pd_result = lax.while_loop(not_converged, pd_step, initial_pd)
                pd_count, u, _, upper_dual, lower_dual, obstacle_dual, _, _ = pd_result
                # The clip is an actuator-safety guard for a finite PD iteration cap.
                controls = jnp.clip(u, u_min, u_max).reshape(H, nu)
                inner_obs = obs_x if self.reconstruct_state_plan else obs_x[:, :3]
                inner_gamma = gamma_x if self.reconstruct_state_plan else gamma_x[:, :3]
                inner_bias = (
                    decoder_bias if self.reconstruct_state_plan else decoder_bias[:3]
                )
                plan = (
                    jnp.einsum("hxn,n->hx", inner_obs, z0)
                    + inner_bias
                    + jnp.einsum("hxku,ku->hx", inner_gamma, controls)
                )
                return (
                    u,
                    upper_dual,
                    lower_dual,
                    obstacle_dual,
                    unused,
                    plan[:, :3],
                    pd_count,
                )

            u, upper_dual, lower_dual, obstacle_dual, unused, reference, pd_count = (
                lax.fori_loop(
                    0,
                    sqp_iterations,
                    sqp_step,
                    (
                        warm_u,
                        warm_upper_dual,
                        warm_lower_dual,
                        warm_obstacle_dual,
                        warm_obstacle_aux,
                        reference,
                        jnp.asarray(0, dtype=jnp.int32),
                    ),
                )
            )
            controls = jnp.clip(u, u_min, u_max).reshape(H, nu)
            output_obs = obs_x if self.reconstruct_state_plan else obs_x[:, :3]
            output_gamma = gamma_x if self.reconstruct_state_plan else gamma_x[:, :3]
            output_bias = (
                decoder_bias if self.reconstruct_state_plan else decoder_bias[:3]
            )
            plan = (
                jnp.einsum("hxn,n->hx", output_obs, z0)
                + output_bias
                + jnp.einsum("hxku,ku->hx", output_gamma, controls)
            )
            next_reference = jnp.concatenate((plan[1:, :3], plan[-1:, :3]), axis=0)
            return (
                plan,
                controls,
                shift_blocks(u, nu),
                shift_blocks(upper_dual, nu),
                shift_blocks(lower_dual, nu),
                jnp.concatenate((obstacle_dual[1:], obstacle_dual[-1:])),
                unused,
                next_reference,
                pd_count,
            )

        return solve_impl

    def reset_warm_state(self):
        self.warm_u = self.u_hover_tiled
        self.warm_upper_dual = jnp.zeros_like(self.warm_upper_dual)
        self.warm_lower_dual = jnp.zeros_like(self.warm_lower_dual)
        self.warm_obstacle_dual = jnp.zeros_like(self.warm_obstacle_dual)
        self.warm_obstacle_aux = jnp.zeros_like(self.warm_obstacle_aux)
        self.warm_reference = None

    def initial_warm_state(self, start_position, goal_position):
        return (
            self.warm_u,
            self.warm_upper_dual,
            self.warm_lower_dual,
            self.warm_obstacle_dual,
            self.warm_obstacle_aux,
            self.seed_reference(start_position, goal_position),
        )

    def warmup(self, z0, goal_y, start_position=None, goal_position=None):
        if start_position is None:
            start_position = self.obstacle_base_obs[0] @ z0 + self.obstacle_bias
        if goal_position is None:
            goal_position = goal_y[:3]
        warm = self.initial_warm_state(start_position, goal_position)
        result = self._solve_jit(*warm, z0, goal_y)
        result[1].block_until_ready()
        return result[0]

    def solve(self, z0, goal_y, current_position=None, goal_position=None):
        if self.warm_reference is None:
            if current_position is None or goal_position is None:
                raise ValueError(
                    "The first obstacle solve needs current_position and goal_position"
                )
            self.warm_reference = self.seed_reference(current_position, goal_position)
        start = time.perf_counter()
        result = self._solve_jit(
            self.warm_u,
            self.warm_upper_dual,
            self.warm_lower_dual,
            self.warm_obstacle_dual,
            self.warm_obstacle_aux,
            self.warm_reference,
            z0,
            goal_y,
        )
        result[1].block_until_ready()
        solve_time = time.perf_counter() - start
        (
            plan,
            controls,
            self.warm_u,
            self.warm_upper_dual,
            self.warm_lower_dual,
            self.warm_obstacle_dual,
            self.warm_obstacle_aux,
            self.warm_reference,
            pd_count,
        ) = result
        return plan, controls, solve_time, int(pd_count)
