"""Batched box-constrained proximal primal-dual MPC."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from .cache import save_array_cache
from .condensation import build_prediction_maps

SOLVER_DTYPE = jnp.float32


def linear(x, params):
    """Apply an affine layer stored as NumPy/JAX weight and bias arrays."""
    weight, bias = params
    return x @ weight.T + bias


CACHE_SCHEMA_VERSION = 2
CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache" / "solver"


def _hash_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256(f"batched-pd-v{CACHE_SCHEMA_VERSION}".encode())
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.dtype.str.encode())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()[:20]


class BatchedPrimalDualMPC:
    """FP32 proximal PD using a cached QP inverse built in FP64."""

    def __init__(
        self,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
        c_bias: np.ndarray,
        lift_params: tuple,
        device: jax.Device,
        *,
        horizon: int,
        position_dim: int,
        q_terminal: float,
        q_stage: float,
        r_control: float,
        regularization: float,
        q_velocity: float,
        q_acceleration: float,
        rho: float,
        steps: int,
        mode: str,
        abs_tol: float,
        rel_tol: float,
        action_mean: np.ndarray,
        action_std: np.ndarray,
        cache_dir: Path = CACHE_DIR,
    ) -> None:
        if mode not in {"fixed", "converged"}:
            raise ValueError("PD mode must be 'fixed' or 'converged'.")
        if horizon < 1 or steps < 1 or not np.isfinite(rho) or rho <= 0:
            raise ValueError("horizon, steps, and rho must be positive")
        if not 0 < abs_tol < float("inf") or not 0 < rel_tol < float("inf"):
            raise ValueError("PD tolerances must be positive and finite")
        if any(
            not 0 <= weight < float("inf")
            for weight in (
                q_terminal,
                q_stage,
                r_control,
                regularization,
                q_velocity,
                q_acceleration,
            )
        ):
            raise ValueError("objective weights must be finite and nonnegative")
        if r_control + regularization <= 0:
            raise ValueError("control cost plus regularization must be positive")
        if not 1 <= position_dim <= c.shape[0]:
            raise ValueError("position_dim must lie within the decoded state")
        self.device = device
        self.embed_dim = int(position_dim)
        self.state_dim = int(c.shape[0])
        self.action_dim = int(b.shape[1])
        self.horizon = horizon
        self.rho = rho
        self.steps = steps
        self.mode = mode
        self.abs_tol = abs_tol
        self.rel_tol = rel_tol
        mean = np.asarray(action_mean, dtype=np.float64)
        std = np.asarray(action_std, dtype=np.float64)
        if (
            mean.shape != (self.action_dim,)
            or std.shape != (self.action_dim,)
            or bool((std <= 0).any())
        ):
            raise ValueError("Invalid saved action normalization.")
        lower_bound = np.tile((-1.0 - mean) / std, horizon)
        upper_bound = np.tile((1.0 - mean) / std, horizon)
        self.warm_controls: jax.Array | None = None
        self.warm_dual: jax.Array | None = None
        self.last_pd_diagnostics: dict[str, Any] = {}
        self.last_solve_ms = 0.0

        a, b, c, c_bias = (
            np.asarray(value, dtype=np.float64) for value in (a, b, c, c_bias)
        )
        maps = build_prediction_maps(a, b, c, c_bias, horizon)
        gamma = maps["gamma_x"].reshape(
            horizon, self.state_dim, horizon * self.action_dim
        )
        weights = np.full((horizon,), q_stage, dtype=np.float64)
        weights[-1] = q_terminal
        state_weights = np.ones(self.state_dim, dtype=np.float64)
        if self.state_dim >= 2 * self.embed_dim:
            state_weights[self.embed_dim : 2 * self.embed_dim] = q_velocity
        if self.state_dim > 2 * self.embed_dim:
            state_weights[2 * self.embed_dim :] = q_acceleration
        weighted_gamma = weights[:, None, None] * state_weights[None, :, None] * gamma
        hessian = np.einsum("hen,hem->nm", gamma, weighted_gamma)
        size = horizon * self.action_dim
        hessian += (r_control + regularization) * np.eye(size, dtype=np.float64)
        hessian = 0.5 * (hessian + hessian.T)

        free_map = maps["obs_x"]

        cache_key = _hash_arrays(
            a,
            b,
            c,
            c_bias,
            gamma,
            weights,
            state_weights,
            lower_bound,
            upper_bound,
            np.asarray([rho, r_control, regularization], dtype=np.float64),
        )
        cache_path = Path(cache_dir) / f"batched_qp_inverse_{cache_key}.npz"
        cache_hit = False
        if cache_path.exists():
            try:
                with np.load(cache_path, allow_pickle=False) as cached:
                    cached_key = str(cached["cache_key"].item())
                    inverse = cached["inverse"].copy()
                    cached_gamma = cached["gamma"].copy()
                    cached_free_map = cached["free_map"].copy()
                cache_hit = (
                    cached_key == cache_key
                    and inverse.dtype == np.float64
                    and cached_gamma.dtype == np.float64
                    and cached_free_map.dtype == np.float64
                    and inverse.shape == (size, size)
                    and cached_gamma.shape == gamma.shape
                    and cached_free_map.shape == free_map.shape
                )
                if cache_hit:
                    gamma, free_map = cached_gamma, cached_free_map
            except (KeyError, OSError, ValueError):
                cache_hit = False
        if not cache_hit:
            inverse = np.linalg.inv(hessian + rho * np.eye(size, dtype=np.float64))
            save_array_cache(
                cache_path,
                cache_key=np.asarray(cache_key),
                inverse=inverse.astype(np.float64, copy=False),
                gamma=gamma.astype(np.float64, copy=False),
                free_map=free_map.astype(np.float64, copy=False),
                decoder_bias=c_bias.astype(np.float64, copy=False),
                weights=weights.astype(np.float64, copy=False),
                state_weights=state_weights.astype(np.float64, copy=False),
                lower_bound=lower_bound.astype(np.float64, copy=False),
                upper_bound=upper_bound.astype(np.float64, copy=False),
            )
        self.cache_info = {
            "path": str(cache_path),
            "hit": cache_hit,
            "build_dtype": "float64",
        }

        def fp32(array: np.ndarray) -> jax.Array:
            return jax.device_put(jnp.asarray(array, dtype=SOLVER_DTYPE), device)

        self.gamma = fp32(gamma)
        self.free_map = fp32(free_map)
        self.decoder_bias = fp32(c_bias)
        self.weights = fp32(weights)
        self.state_weights = fp32(state_weights)
        self.inverse = fp32(inverse)
        self.lower_bound = fp32(lower_bound)
        self.upper_bound = fp32(upper_bound)
        self.lift_params = tuple(
            (fp32(weight), fp32(bias)) for weight, bias in lift_params
        )
        self._solve_jit = jax.jit(self._build_solve_impl())

    def _build_solve_impl(self):
        horizon, action_dim = self.horizon, self.action_dim
        size, rho, steps = horizon * action_dim, self.rho, self.steps
        mode, abs_tol, rel_tol = self.mode, self.abs_tol, self.rel_tol
        gamma, free_map, decoder_bias = self.gamma, self.free_map, self.decoder_bias
        weights, state_weights, inverse = self.weights, self.state_weights, self.inverse
        lower_bound, upper_bound, lift_params = (
            self.lower_bound,
            self.upper_bound,
            self.lift_params,
        )

        def shift_controls(values):
            blocks = values.reshape(values.shape[0], horizon, action_dim)
            return jnp.concatenate((blocks[:, 1:], blocks[:, -1:]), axis=1).reshape(
                values.shape
            )

        def shift_dual(values):
            blocks = values.reshape(values.shape[0], horizon, action_dim)
            return jnp.concatenate(
                (blocks[:, 1:], jnp.zeros_like(blocks[:, -1:])), axis=1
            ).reshape(values.shape)

        def solve_impl(warm_controls, warm_dual, current_state, goal_embedding):
            z = current_state.astype(SOLVER_DTYPE)
            for index, params in enumerate(lift_params):
                z = linear(z, params)
                if index + 1 < len(lift_params):
                    z = jax.nn.gelu(z, approximate=False)
            free = jnp.einsum("hez,bz->bhe", free_map, z) + decoder_bias[None, None, :]
            goal = jnp.pad(
                goal_embedding.astype(SOLVER_DTYPE),
                ((0, 0), (0, self.state_dim - self.embed_dim)),
            )
            error = free - goal[:, None, :]
            weighted_error = (
                weights[None, :, None] * state_weights[None, None, :] * error
            )
            gradient = jnp.einsum("hen,bhe->bn", gamma, weighted_error)

            controls = shift_controls(warm_controls)
            dual = shift_dual(warm_dual)
            slack = jnp.clip(controls, lower_bound, upper_bound)
            initial = (
                jnp.asarray(0, dtype=jnp.int32),
                controls,
                slack,
                dual,
                jnp.full((gradient.shape[0],), jnp.inf, dtype=SOLVER_DTYPE),
                jnp.full((gradient.shape[0],), jnp.inf, dtype=SOLVER_DTYPE),
                jnp.zeros((gradient.shape[0],), dtype=jnp.bool_),
            )

            def update(carry):
                iteration, old_controls, old_slack, old_dual, _, _, _ = carry
                # Proximal PD: tau=1/rho, sigma=0.49*rho, A=I.
                # tau*sigma*||A||^2 < 1, independent of horizon.
                controls = (rho * old_controls - gradient - old_dual) @ inverse.T
                sigma = 0.49 * rho
                extrapolated = 2.0 * controls - old_controls
                candidate = old_dual + sigma * extrapolated
                dual = candidate - sigma * jnp.clip(
                    candidate / sigma, lower_bound, upper_bound
                )
                slack = jnp.clip(controls, lower_bound, upper_bound)
                primal = jnp.linalg.norm(controls - slack, axis=1)
                primal_change = rho * jnp.linalg.norm(controls - old_controls, axis=1)
                dual_change = jnp.linalg.norm(dual - old_dual, axis=1) / sigma
                dual_residual = jnp.maximum(primal_change, dual_change)
                primal_tol = size**0.5 * abs_tol + rel_tol * jnp.linalg.norm(
                    controls, axis=1
                )
                dual_tol = size**0.5 * abs_tol + rel_tol * jnp.maximum(
                    rho * jnp.linalg.norm(controls, axis=1),
                    jnp.linalg.norm(dual, axis=1) / sigma,
                )
                converged = (primal <= primal_tol) & (dual_residual <= dual_tol)
                return (
                    iteration + 1,
                    controls,
                    slack,
                    dual,
                    primal,
                    dual_residual,
                    converged,
                )

            if mode == "converged":

                def condition(carry):
                    return (carry[0] < steps) & (~jnp.all(carry[-1]))

                result = lax.while_loop(condition, update, initial)
            else:
                result = lax.fori_loop(
                    0, steps, lambda _, carry: update(carry), initial
                )
            iteration, controls, slack, dual, primal, dual_residual, converged = result
            first_action = slack.reshape(-1, horizon, action_dim)[:, 0]
            return (
                first_action,
                controls,
                dual,
                iteration,
                primal,
                dual_residual,
                converged,
            )

        return solve_impl

    def first_action(
        self, current_state: jax.Array, goal_embedding: jax.Array
    ) -> jax.Array:
        batch = len(current_state)
        size = self.horizon * self.action_dim
        if self.warm_controls is None or self.warm_controls.shape != (batch, size):
            self.warm_controls = jax.device_put(
                jnp.zeros((batch, size), dtype=SOLVER_DTYPE), self.device
            )
            self.warm_dual = jnp.zeros_like(self.warm_controls)
        assert self.warm_dual is not None
        # Finish state construction before measuring only the compiled QP solve.
        jax.block_until_ready(
            (self.warm_controls, self.warm_dual, current_state, goal_embedding)
        )
        solve_started_ns = time.perf_counter_ns()
        action, controls, dual, iterations, primal, dual_residual, converged = (
            self._solve_jit(
                self.warm_controls, self.warm_dual, current_state, goal_embedding
            )
        )
        action.block_until_ready()
        self.last_solve_ms = (time.perf_counter_ns() - solve_started_ns) / 1e6
        self.warm_controls, self.warm_dual = controls, dual
        self.last_pd_diagnostics = {
            "iterations": int(iterations),
            "converged_count": int(jnp.sum(converged)),
            "batch_size": batch,
            "max_primal_residual": float(jnp.max(primal)),
            "max_dual_residual": float(jnp.max(dual_residual)),
        }
        return action

    def warmup(self, current_state: jax.Array, goal_embedding: jax.Array) -> None:
        """Compile the solve without advancing the persistent PD warm start."""
        batch = len(current_state)
        size = self.horizon * self.action_dim
        zeros = jax.device_put(
            jnp.zeros((batch, size), dtype=SOLVER_DTYPE), self.device
        )
        result = self._solve_jit(zeros, zeros, current_state, goal_embedding)
        result[0].block_until_ready()
