"""Array-only condensed MPC with proximal primal-dual iterations."""

import time

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import numpy as np
from jax import lax

from .condensation import (
    CACHE_DIR,
    load_or_build_prediction_maps,
    load_or_build_qp_factor,
    load_or_build_qp_inverse,
)


def validate_cost_parameters(q_state, q_terminal, state_dim):
    """Validate costs for every physical-state component in decoder order."""
    state = np.asarray(q_state, dtype=np.float64).flatten()
    terminal = np.asarray(q_terminal, dtype=np.float64).flatten()
    if state.shape != (state_dim,) or terminal.shape != (state_dim,):
        raise ValueError(
            f"Q_STATE and Q_TERMINAL must each contain {state_dim} weights."
        )
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(terminal)):
        raise ValueError("State-cost weights must be finite.")
    if np.any(state < 0.0) or np.any(terminal < 0.0):
        raise ValueError("Q_STATE and Q_TERMINAL weights must be nonnegative.")
    return state, terminal


class PrimalDualMPC:
    """Condense affine decoded dynamics and solve control-bound QPs with PD.

    The terminal weight replaces the last running-state weight. The proximal
    system is built in float64; online iterations use the requested dtype.
    Warm-start state belongs to this solver instance.
    """

    def __init__(
        self,
        A,
        B,
        decoder_weight,
        decoder_bias,
        r_val,
        u_min,
        u_max,
        horizon,
        rho,
        n_pd_steps,
        device,
        q_state=None,
        q_terminal=None,
        cache_dir=CACHE_DIR,
        warm_start=True,
        termination="converged",
        primal_tol=1e-2,
        fixed_point_tol=1e-2,
        step_safety=0.99,
        extrapolation=1.0,
        factorize=False,
        control_reference=None,
        reconstruct_state_plan=False,
        dtype=np.float32,
    ):
        if np.dtype(dtype) not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("dtype must be float32 or float64")
        if not 0 < r_val < float("inf"):
            raise ValueError("control cost must be positive and finite")
        if not np.isfinite(rho):
            raise ValueError("rho must be finite")
        lower, upper = np.asarray(u_min), np.asarray(u_max)
        if lower.shape != (B.shape[1],) or upper.shape != lower.shape:
            raise ValueError("control bounds must match the control dimension")
        if (
            not np.isfinite(lower).all()
            or not np.isfinite(upper).all()
            or (lower > upper).any()
        ):
            raise ValueError("control bounds must be finite and ordered")
        self.numpy_dtype = np.dtype(dtype).type
        self.dtype = jnp.dtype(dtype)
        state_dim = int(decoder_weight.shape[0])
        q_state = np.zeros(state_dim) if q_state is None else q_state
        q_terminal = np.ones(state_dim) if q_terminal is None else q_terminal
        if horizon < 1 or n_pd_steps < 1 or rho <= 0:
            raise ValueError("horizon, PD steps, and rho must be positive")
        if termination not in {"fixed", "converged"}:
            raise ValueError("termination must be fixed or converged")
        if primal_tol <= 0 or fixed_point_tol <= 0:
            raise ValueError("PD tolerances must be positive")
        if not 0 < step_safety < 1 or not 0 <= extrapolation <= 1:
            raise ValueError("invalid PD step safety or extrapolation")
        self.reconstruct_state_plan = bool(reconstruct_state_plan)
        self.warm_start = bool(warm_start)
        self.termination = termination
        self.primal_tol, self.fixed_point_tol = (
            float(primal_tol),
            float(fixed_point_tol),
        )
        self.step_safety, self.extrapolation = float(step_safety), float(extrapolation)
        self.H = int(horizon)
        self.nz = int(A.shape[0])
        self.nu = int(B.shape[1])
        self.n_controls = self.H * self.nu
        self.rho = float(rho)
        self.steps = int(n_pd_steps)
        self.device = device
        self.factorize = bool(factorize)

        maps, maps_path, maps_hit, model_key = load_or_build_prediction_maps(
            A, B, decoder_weight, decoder_bias, self.H, cache_dir
        )
        self.n_states = maps["decoder_bias"].shape[0]
        state_weights, terminal_weights = validate_cost_parameters(
            q_state, q_terminal, self.n_states
        )
        self.a_y_z = maps["obs_x"].reshape(self.H * self.n_states, self.nz)
        self.g_y = maps["gamma_x"].reshape(self.H * self.n_states, self.n_controls)
        self.c_y = np.tile(maps["decoder_bias"], self.H)

        # Q_STATE is the running cost for x_1 ... x_{H-1}; Q_TERMINAL
        # replaces it at x_H, matching the standard finite-horizon objective.
        q_diagonal = np.tile(state_weights, self.H)
        q_diagonal[-self.n_states :] = terminal_weights
        q_y = np.diag(q_diagonal).astype(self.numpy_dtype)
        r_scalar = r_val if r_val > 0 else 1.0
        r_bar = np.eye(self.n_controls, dtype=self.numpy_dtype) * r_scalar
        if self.factorize:
            k_system, system_path, system_hit = load_or_build_qp_factor(
                np.asarray(self.g_y, dtype=self.numpy_dtype),
                q_y,
                r_bar,
                self.rho,
                model_key,
                self.H,
                cache_dir,
            )
            system_kind = "cholesky"
        else:
            k_system, system_path, system_hit = load_or_build_qp_inverse(
                np.asarray(self.g_y, dtype=self.numpy_dtype),
                q_y,
                r_bar,
                self.rho,
                model_key,
                self.H,
                cache_dir,
            )
            system_kind = "inverse"

        self.cache_info = {
            "maps_path": maps_path,
            "maps_hit": maps_hit,
            "system_kind": system_kind,
            "system_path": system_path,
            "system_hit": system_hit,
        }
        self.obs_x = jax.device_put(
            jnp.asarray(maps["obs_x"], dtype=self.dtype), device
        )
        self.gamma_x = jax.device_put(
            jnp.asarray(maps["gamma_x"], dtype=self.dtype), device
        )
        self.decoder_bias = jax.device_put(
            jnp.asarray(maps["decoder_bias"], dtype=self.dtype), device
        )
        self.a_y_z = jax.device_put(jnp.asarray(self.a_y_z, dtype=self.dtype), device)
        self.g_y = jax.device_put(jnp.asarray(self.g_y, dtype=self.dtype), device)
        self.c_y = jax.device_put(jnp.asarray(self.c_y, dtype=self.dtype), device)
        self.q_y = jax.device_put(jnp.asarray(q_y, dtype=self.dtype), device)
        self.r_bar = jax.device_put(jnp.asarray(r_bar, dtype=self.dtype), device)
        self.k_system = jax.device_put(jnp.asarray(k_system, dtype=self.dtype), device)

        u_hover = np.asarray(
            control_reference if control_reference is not None else np.zeros(self.nu),
            dtype=self.numpy_dtype,
        )
        self.u_hover_tiled = jax.device_put(jnp.tile(u_hover, self.H), device)
        self.u_min_tiled = jax.device_put(
            jnp.tile(np.asarray(u_min, dtype=self.numpy_dtype), self.H), device
        )
        self.u_max_tiled = jax.device_put(
            jnp.tile(np.asarray(u_max, dtype=self.numpy_dtype), self.H), device
        )
        self.warm_u = jax.device_put(
            jnp.zeros(self.n_controls, dtype=self.dtype), device
        )
        self.warm_upper_dual = jax.device_put(
            jnp.zeros(self.n_controls, dtype=self.dtype), device
        )
        self.warm_lower_dual = jax.device_put(
            jnp.zeros(self.n_controls, dtype=self.dtype), device
        )
        self._solve_jit = jax.jit(self._build_solve_impl())

    def _build_solve_impl(self):
        H, nu = self.H, self.nu
        rho, steps = self.rho, self.steps
        factorize = self.factorize
        a_y_z, g_y, c_y = self.a_y_z, self.g_y, self.c_y
        q_y, r_bar, k_system = self.q_y, self.r_bar, self.k_system
        obs_x, gamma_x, decoder_bias = self.obs_x, self.gamma_x, self.decoder_bias
        u_hover_tiled = self.u_hover_tiled
        u_min_tiled, u_max_tiled = self.u_min_tiled, self.u_max_tiled

        def shift(sequence):
            blocks = sequence.reshape(H, nu)
            return jnp.concatenate([blocks[1:], blocks[-1:]], axis=0).reshape(-1)

        def solve_impl(warm_u, warm_upper_dual, warm_lower_dual, z0, y_goal):
            error = a_y_z @ z0 + c_y - jnp.tile(y_goal, H)
            gradient = g_y.T @ (q_y @ error) - r_bar @ u_hover_tiled

            if self.warm_start:
                u, s, eta = (
                    shift(warm_u),
                    shift(warm_upper_dual),
                    shift(warm_lower_dual),
                )
            else:
                u, s, eta = (
                    jnp.zeros_like(warm_u),
                    jnp.zeros_like(warm_upper_dual),
                    jnp.zeros_like(warm_lower_dual),
                )

            tau = 1.0 / rho
            # For box constraints A = [I; -I], ||A||_2^2 = 2 exactly.
            # A Frobenius bound (2 * n_controls) needlessly shrinks the dual
            # step as the horizon grows and slows PD convergence.
            sigma = self.step_safety / (tau * 2.0)

            def pd_update(carry):
                i, old_u, extrapolated, upper, lower, _, _ = carry
                next_upper = jnp.maximum(
                    upper + sigma * (extrapolated - u_max_tiled), 0.0
                )
                next_lower = jnp.maximum(
                    lower + sigma * (u_min_tiled - extrapolated), 0.0
                )
                rhs = rho * old_u - gradient - next_upper + next_lower
                if factorize:
                    intermediate = jsp_linalg.solve_triangular(
                        k_system, rhs, lower=True
                    )
                    next_u = jsp_linalg.solve_triangular(
                        k_system.T, intermediate, lower=False
                    )
                else:
                    next_u = k_system @ rhs
                primal = jnp.sqrt(
                    jnp.sum(
                        jnp.maximum(next_u - u_max_tiled, 0.0) ** 2
                        + jnp.maximum(u_min_tiled - next_u, 0.0) ** 2
                    )
                )
                fixed_point = jnp.sqrt(
                    jnp.sum(
                        ((next_u - old_u) / tau) ** 2
                        + ((next_upper - upper) / sigma) ** 2
                        + ((next_lower - lower) / sigma) ** 2
                    )
                )
                return (
                    i + 1,
                    next_u,
                    next_u + self.extrapolation * (next_u - old_u),
                    next_upper,
                    next_lower,
                    primal,
                    fixed_point,
                )

            initial = (0, u, u, s, eta, jnp.inf, jnp.inf)
            if self.termination == "converged":

                def condition(carry):
                    i, _, _, _, _, primal, fixed_point = carry
                    return (i < steps) & (
                        (primal > self.primal_tol)
                        | (fixed_point > self.fixed_point_tol)
                    )

                result = lax.while_loop(condition, pd_update, initial)
            else:
                result = lax.fori_loop(
                    0, steps, lambda _, carry: pd_update(carry), initial
                )
            count, u, _, upper, lower, primal, fixed_point = result
            # Store feasibility and the step-scaled fixed-point residual separately.
            s = jnp.clip(u, u_min_tiled, u_max_tiled)
            # Return the projected iterate so fixed-iteration operation never
            # emits a control outside the hard bounds.
            u_sequence = s.reshape(H, nu)
            x_plan = None
            if self.reconstruct_state_plan:
                x_plan = (
                    jnp.einsum("hxn,n->hx", obs_x, z0)
                    + decoder_bias[None, :]
                    + jnp.einsum("hxku,ku->hx", gamma_x, u_sequence)
                )
            return x_plan, u_sequence, u, upper, lower, count, primal, fixed_point

        return solve_impl

    def warmup(self, z0, y_goal):
        x_plan, u_sequence, u, s, eta, count, primal, fixed_point = self._solve_jit(
            self.warm_u, self.warm_upper_dual, self.warm_lower_dual, z0, y_goal
        )
        u_sequence.block_until_ready()
        self.warm_u, self.warm_upper_dual, self.warm_lower_dual = u, s, eta
        return x_plan

    def solve(self, z0, y_goal):
        start = time.perf_counter()
        x_plan, u_sequence, u, s, eta, count, primal, fixed_point = self._solve_jit(
            self.warm_u, self.warm_upper_dual, self.warm_lower_dual, z0, y_goal
        )
        u_sequence.block_until_ready()
        solve_time = time.perf_counter() - start
        if self.warm_start:
            self.warm_u, self.warm_upper_dual, self.warm_lower_dual = u, s, eta
        self._diagnostics = (count, primal, fixed_point)
        return x_plan, u_sequence, solve_time

    @property
    def diagnostics(self):
        """Transfer convergence scalars only when a caller requests them."""
        count, primal, fixed_point = self._diagnostics
        values = np.asarray(jax.device_get((count, primal, fixed_point)))
        return {
            "iterations": int(values[0]),
            "primal_residual": float(values[1]),
            "fixed_point_residual": float(values[2]),
        }
