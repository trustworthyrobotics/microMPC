"""Matrix-free moving-sphere SQP/PD solver for batched IBR responses."""

from __future__ import annotations


import jax
import jax.numpy as jnp
from jax import lax

from . import obstacle_solver as obstacle_base


PD_CHECK_INTERVAL = 5


class MultiRobotPrimalDualMPC(obstacle_base.ObstaclePrimalDualMPC):
    """SQP/PD MPC with a time-varying sphere for every other robot."""

    def __init__(
        self,
        *args,
        num_obstacles: int,
        pd_check_interval: int = PD_CHECK_INTERVAL,
        **kwargs,
    ):
        if num_obstacles < 0:
            raise ValueError("num_obstacles cannot be negative")
        if pd_check_interval <= 0:
            raise ValueError("pd_check_interval must be positive")
        super().__init__(*args, **kwargs)
        self.num_obstacles = int(num_obstacles)
        self.pd_check_interval = int(pd_check_interval)
        dtype = self.gamma_x.dtype
        self.warm_obstacle_dual = jnp.zeros((self.H, self.num_obstacles), dtype=dtype)
        self.warm_obstacle_aux = jnp.zeros_like(self.warm_obstacle_dual)
        self._solve_jit = jax.jit(self._build_multi_obstacle_solve_impl())

    def _build_multi_obstacle_solve_impl(self):
        horizon, control_dim = self.H, self.nu
        obstacle_count = self.num_obstacles
        check_interval = self.pd_check_interval
        rho = self.rho
        primal_step = 1.0 / rho
        a_y_z, g_y, c_y = self.a_y_z, self.g_y, self.c_y
        q_y, r_bar, inverse = self.q_y, self.r_bar, self.k_system
        obs_x, gamma_x, bias = self.obs_x, self.gamma_x, self.decoder_bias
        obstacle_gamma = self.obstacle_gamma
        # Used to compute ||A_obs||_F without materializing A_obs.  For each
        # horizon step this is Gamma Gamma^T in physical position space.
        obstacle_gamma_gram = jnp.einsum("hpn,hqn->hpq", obstacle_gamma, obstacle_gamma)
        obstacle_obs, obstacle_bias = self.obstacle_base_obs, self.obstacle_bias
        hover, u_min, u_max = self.u_hover_tiled, self.u_min_tiled, self.u_max_tiled

        h_index = jnp.arange(horizon, dtype=gamma_x.dtype)[:, None]
        o_index = jnp.arange(obstacle_count, dtype=gamma_x.dtype)[None, :]
        angle = 0.31 * (h_index + 1.0) + 0.73 * (o_index + 1.0)
        z = 2.0 * jnp.mod(0.618 * (h_index + o_index + 1.0), 1.0) - 1.0
        radial = jnp.sqrt(jnp.maximum(1.0 - z * z, 0.0))
        fallback_normals = jnp.stack(
            (radial * jnp.cos(angle), radial * jnp.sin(angle), z), axis=-1
        )

        def shift_blocks(sequence, width):
            blocks = sequence.reshape(horizon, width)
            return jnp.concatenate((blocks[1:], blocks[-1:]), axis=0).reshape(
                sequence.shape
            )

        def shift_obstacle_dual(sequence):
            return jnp.concatenate((sequence[1:], sequence[-1:]), axis=0)

        def solve_impl(
            warm_u,
            warm_upper_dual,
            warm_lower_dual,
            warm_obstacle_dual,
            warm_obstacle_aux,
            reference,
            obstacle_centers,
            obstacle_radii,
            z0,
            goal_y,
        ):
            error = a_y_z @ z0 + c_y - jnp.tile(goal_y, horizon)
            gradient = g_y.T @ (q_y @ error) - r_bar @ hover
            base_position = jnp.einsum("hpn,n->hp", obstacle_obs, z0) + obstacle_bias

            def sqp_step(_, carry):
                u, upper, lower, obs_dual, unused, ref, _ = carry
                difference = ref[:, None, :] - obstacle_centers
                norms = jnp.linalg.norm(difference, axis=-1, keepdims=True)
                normals = jnp.where(
                    norms > 1.0e-8,
                    difference / jnp.maximum(norms, 1.0e-8),
                    fallback_normals,
                )
                obs_bound = (
                    -jnp.einsum("hop,hop->ho", normals, obstacle_centers)
                    - obstacle_radii[None, :]
                    + jnp.einsum("hop,hp->ho", normals, base_position)
                ).reshape(-1)
                norm_bound_sq = 2.0 * u.size + jnp.einsum(
                    "hop,hpq,hoq->", normals, obstacle_gamma_gram, normals
                )
                dual_step = self.pd_step_safety / (primal_step * norm_bound_sq)

                def apply_obstacles(vector):
                    # A_obs @ vector, retaining its natural (horizon, obstacle) layout.
                    return -jnp.einsum("hop,hpn,n->ho", normals, obstacle_gamma, vector)

                def apply_obstacles_transpose(dual):
                    # A_obs.T @ dual, without the H*O by H*nu temporary matrix.
                    return -jnp.einsum("hop,hpn,ho->n", normals, obstacle_gamma, dual)

                def pd_step(value):
                    iteration, u_old, u_bar, dual_upper, dual_lower, dual_obs, _, _ = (
                        value
                    )
                    previous_upper, previous_lower, previous_obs = (
                        dual_upper,
                        dual_lower,
                        dual_obs,
                    )
                    dual_upper = jnp.maximum(
                        dual_upper + dual_step * (u_bar - u_max), 0.0
                    )
                    dual_lower = jnp.maximum(
                        dual_lower + dual_step * (u_min - u_bar), 0.0
                    )
                    dual_obs_flat = jnp.maximum(
                        dual_obs.reshape(-1)
                        + dual_step * (apply_obstacles(u_bar).reshape(-1) - obs_bound),
                        0.0,
                    )
                    dual_obs = dual_obs_flat.reshape(horizon, obstacle_count)
                    rhs = u_old - primal_step * (
                        gradient
                        + dual_upper
                        - dual_lower
                        + apply_obstacles_transpose(dual_obs)
                    )
                    # Written as RHS @ K^-T so vmap batches all robots/environments
                    # into one shared-matrix GEMM instead of independent GEMVs.
                    u_new = rho * (rhs @ inverse.T)
                    u_bar_new = u_new + self.pd_extrapolation * (u_new - u_old)
                    next_iteration = iteration + 1

                    def convergence_metrics(_):
                        feasibility = jnp.sqrt(
                            jnp.sum(jnp.maximum(u_new - u_max, 0.0) ** 2)
                            + jnp.sum(jnp.maximum(u_min - u_new, 0.0) ** 2)
                            + jnp.sum(
                                jnp.maximum(
                                    apply_obstacles(u_new).reshape(-1) - obs_bound, 0.0
                                )
                                ** 2
                            )
                        )
                        fixed_point = jnp.sqrt(
                            jnp.sum(((u_new - u_old) / primal_step) ** 2)
                            + jnp.sum(((dual_upper - previous_upper) / dual_step) ** 2)
                            + jnp.sum(((dual_lower - previous_lower) / dual_step) ** 2)
                            + jnp.sum(((dual_obs - previous_obs) / dual_step) ** 2)
                        )
                        return feasibility, fixed_point

                    if self.pd_mode == "fixed":
                        feasibility, fixed_point = value[6], value[7]
                    else:
                        check_now = ((next_iteration % check_interval) == 0) | (
                            next_iteration == self.steps
                        )
                        feasibility, fixed_point = lax.cond(
                            check_now,
                            convergence_metrics,
                            lambda _: (value[6], value[7]),
                            operand=None,
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

                initial = (
                    jnp.asarray(0, jnp.int32),
                    u,
                    u,
                    upper,
                    lower,
                    obs_dual,
                    jnp.asarray(jnp.inf, u.dtype),
                    jnp.asarray(jnp.inf, u.dtype),
                )
                if self.pd_mode == "fixed":
                    result = lax.fori_loop(
                        0, self.steps, lambda _, value: pd_step(value), initial
                    )
                else:

                    def not_converged(value):
                        iteration, _, _, _, _, _, feasibility, fixed_point = value
                        return (iteration < self.steps) & (
                            (feasibility > self.pd_primal_tol)
                            | (fixed_point > self.pd_fixed_point_tol)
                        )

                    result = lax.while_loop(not_converged, pd_step, initial)
                count, u, _, upper, lower, obs_dual, _, _ = result
                controls = jnp.clip(u, u_min, u_max).reshape(horizon, control_dim)
                inner_obs = obs_x if self.reconstruct_state_plan else obs_x[:, :3]
                inner_gamma = gamma_x if self.reconstruct_state_plan else gamma_x[:, :3]
                inner_bias = bias if self.reconstruct_state_plan else bias[:3]
                plan = (
                    jnp.einsum("hxn,n->hx", inner_obs, z0)
                    + inner_bias
                    + jnp.einsum("hxku,ku->hx", inner_gamma, controls)
                )
                return u, upper, lower, obs_dual, unused, plan[:, :3], count

            u, upper, lower, obs_dual, unused, _, count = lax.fori_loop(
                0,
                self.sqp_iterations,
                sqp_step,
                (
                    warm_u,
                    warm_upper_dual,
                    warm_lower_dual,
                    warm_obstacle_dual,
                    warm_obstacle_aux,
                    reference,
                    jnp.asarray(0, jnp.int32),
                ),
            )
            controls = jnp.clip(u, u_min, u_max).reshape(horizon, control_dim)
            output_obs = obs_x if self.reconstruct_state_plan else obs_x[:, :3]
            output_gamma = gamma_x if self.reconstruct_state_plan else gamma_x[:, :3]
            output_bias = bias if self.reconstruct_state_plan else bias[:3]
            plan = (
                jnp.einsum("hxn,n->hx", output_obs, z0)
                + output_bias
                + jnp.einsum("hxku,ku->hx", output_gamma, controls)
            )
            next_reference = jnp.concatenate((plan[1:, :3], plan[-1:, :3]), axis=0)
            return (
                plan,
                controls,
                shift_blocks(u, control_dim),
                shift_blocks(upper, control_dim),
                shift_blocks(lower, control_dim),
                shift_obstacle_dual(obs_dual),
                unused,
                next_reference,
                count,
            )

        return solve_impl
