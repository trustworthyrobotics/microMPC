"""Condensed Koopman MPC experiment with a control-only PD solve.

The QP decision variable is the stacked control sequence.  Koopman dynamics
are projected through the affine decoder while building the cached maps, so
the online objective and planned trajectory are expressed in physical space.
"""

from pathlib import Path

from micrompc.solver.primal_dual import PrimalDualMPC as _PrimalDualMPC

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np


def configure_precision(precision):
    """Configure NumPy and JAX dtypes from a single 32/64-bit setting."""
    global NP_DTYPE, JAX_DTYPE
    if precision not in (32, 64):
        raise ValueError(f"PRECISION must be 32 or 64, got {precision!r}.")
    jax.config.update("jax_enable_x64", precision == 64)
    NP_DTYPE = np.float64 if precision == 64 else np.float32
    JAX_DTYPE = jnp.float64 if precision == 64 else jnp.float32
    return NP_DTYPE, JAX_DTYPE


PRECISION = 64
NP_DTYPE, JAX_DTYPE = configure_precision(PRECISION)
# Prediction maps are always built and cached in float64, then cast to the
# configured JAX solver dtype when transferred to the solve device.


# Physical state order is position, attitude, linear velocity, angular velocity.
# Running-state-cost weights for the same four physical-state groups. These
# apply to predicted states 1 through H-1.
Q_STATE_GROUPS = [
    [0.0] * 3,  # position: x, y, z
    [0.0] * 3,  # attitude: yaw, pitch, roll
    [0.0] * 3,  # linear velocity: vx, vy, vz
    [0.0] * 3,  # angular velocity: p, q, r
]
Q_STATE = np.asarray(Q_STATE_GROUPS, dtype=NP_DTYPE).flatten()

# Terminal-cost weights for the final predicted state.
Q_TERMINAL_GROUPS = [
    [15.0] * 3,  # position: x, y, z
    [10.0] * 3,  # attitude: yaw, pitch, roll
    [10.0] * 3,  # linear velocity: vx, vy, vz
    [10.0] * 3,  # angular velocity: p, q, r
]
Q_TERMINAL = np.asarray(Q_TERMINAL_GROUPS, dtype=NP_DTYPE).flatten()


MASS = 1.0
GRAV = -9.81

MIN_DIST = 2.5
LOW_BOUNDS = np.array([0.1, 0.1, 0.1], dtype=NP_DTYPE)
HIGH_BOUNDS = np.array([4.9, 4.9, 4.9], dtype=NP_DTYPE)

CACHE_DIR = Path(__file__).resolve().parents[1] / "plan" / "cache"


def validate_cost_parameters(q_state, q_terminal, state_dim=12):
    """Validate costs for every physical-state component in decoder order."""
    state = np.asarray(q_state, dtype=NP_DTYPE).flatten()
    terminal = np.asarray(q_terminal, dtype=NP_DTYPE).flatten()
    if state.shape != (state_dim,) or terminal.shape != (state_dim,):
        raise ValueError(
            f"Q_STATE and Q_TERMINAL must each contain {state_dim} weights."
        )
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(terminal)):
        raise ValueError("State-cost weights must be finite.")
    if np.any(state < 0.0) or np.any(terminal < 0.0):
        raise ValueError("Q_STATE and Q_TERMINAL weights must be nonnegative.")
    return state, terminal


def to_jax_array(x, dtype=None):
    if dtype is None:
        dtype = JAX_DTYPE
    return jnp.asarray(np.asarray(x, dtype=NP_DTYPE), dtype=dtype)


def select_jax_device(preferred):
    if preferred in {"gpu", "cuda"}:
        try:
            gpu_devices = jax.devices("gpu")
        except RuntimeError:
            gpu_devices = []
        if gpu_devices:
            return gpu_devices[0]
    return jax.devices("cpu")[0]


def normalize_state_jax(x, min_vals, range_vals):
    safe_range_vals = jnp.where(
        range_vals == 0, jnp.full_like(range_vals, 1e-6), range_vals
    )
    return 2.0 * (x - min_vals) / safe_range_vals - 1.0


def denormalize_state_jax(x_norm, min_vals, range_vals):
    safe_range_vals = jnp.where(
        range_vals == 0, jnp.full_like(range_vals, 1e-6), range_vals
    )
    return (x_norm + 1.0) * 0.5 * safe_range_vals + min_vals


def preprocess_state_jax(x, min_vals, range_vals, enable_normalization):
    if enable_normalization:
        return normalize_state_jax(x, min_vals, range_vals)
    return x


def apply_activation_jax(x, activation_fn):
    key = activation_fn.replace("_", "").lower()
    if key == "gelu":
        return jax.nn.gelu(x, approximate=False)
    if key == "relu":
        return jax.nn.relu(x)
    if key == "tanh":
        return jnp.tanh(x)
    raise ValueError(f"Unsupported activation function: {activation_fn!r}")


def encoder_forward_jax(x, encoder_params, activation_fn):
    h = x
    for weight, bias in encoder_params[:-1]:
        h = apply_activation_jax(h @ weight.T + bias, activation_fn)
    weight, bias = encoder_params[-1]
    return h @ weight.T + bias


def load_checkpoint_jax(model_path, device):
    import torch

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model_config = checkpoint.get("model_config", checkpoint.get("config"))
    if model_config is None:
        raise KeyError("Checkpoint is missing 'model_config' or 'config'.")

    state_dict = checkpoint["state_dict"]
    depth = model_config["depth"]
    encoder_params = []
    for layer_idx in range(depth + 1):
        module_idx = 2 * layer_idx
        encoder_params.append(
            (
                jax.device_put(
                    to_jax_array(state_dict[f"encoder.net.{module_idx}.weight"]), device
                ),
                jax.device_put(
                    to_jax_array(state_dict[f"encoder.net.{module_idx}.bias"]), device
                ),
            )
        )

    norm_stats = checkpoint["normalization_stats"]
    training_config = checkpoint.get("training_config", {})
    model = {
        "state_dim": int(model_config["state_dim"]),
        "control_dim": int(model_config["control_dim"]),
        "latent_dim": int(model_config["latent_dim"]),
        "activation_fn": model_config["activation_fn"],
        "encoder_params": tuple(encoder_params),
        "C_weight": jax.device_put(to_jax_array(state_dict["C.weight"]), device),
        "C_bias": jax.device_put(to_jax_array(state_dict["C.bias"]), device),
        "A": jax.device_put(to_jax_array(state_dict["A.weight"]), device),
        "B": jax.device_put(to_jax_array(state_dict["B.weight"]), device),
    }
    stats = {
        "min": jax.device_put(to_jax_array(norm_stats["min"]), device),
        "range": jax.device_put(to_jax_array(norm_stats["range"]), device),
    }
    enable_normalization = bool(training_config.get("enable_normalization", True))
    return model, stats, enable_normalization


def physical_decoder(model, normalizer, enable_normalization):
    """Return x = C z + c in physical (denormalized) state units."""
    c_weight = np.asarray(model["C_weight"], dtype=NP_DTYPE)
    c_bias = np.asarray(model["C_bias"], dtype=NP_DTYPE)
    if not enable_normalization:
        return c_weight, c_bias

    min_vals = np.asarray(normalizer["min"], dtype=NP_DTYPE)
    range_vals = np.asarray(normalizer["range"], dtype=NP_DTYPE)
    safe_range = np.where(range_vals == 0.0, 1e-6, range_vals)
    scale = 0.5 * safe_range
    return scale[:, None] * c_weight, scale * (c_bias + 1.0) + min_vals


class PrimalDualMPC(_PrimalDualMPC):
    """Quadrotor defaults for the shared numerical solver."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("q_state", Q_STATE)
        kwargs.setdefault("q_terminal", Q_TERMINAL)
        kwargs.setdefault("control_reference", [-GRAV * MASS, 0.0, 0.0, 0.0])
        kwargs.setdefault("dtype", NP_DTYPE)
        super().__init__(*args, **kwargs)


def sample_start_goal(rng, lower=LOW_BOUNDS, upper=HIGH_BOUNDS, min_distance=MIN_DIST):
    start = np.zeros(12, dtype=NP_DTYPE)
    goal = np.zeros(12, dtype=NP_DTYPE)
    start_pos = rng.uniform(low=lower, high=upper)
    while True:
        goal_pos = rng.uniform(low=lower, high=upper)
        if np.linalg.norm(goal_pos - start_pos) > min_distance:
            break
    start[:3] = start_pos
    goal[:3] = goal_pos
    return start, goal


def quadrotor_dynamics_jax(x, u, dt):
    psi, theta, phi = x[3], x[4], x[5]
    x_dot, y_dot, z_dot = x[6], x[7], x[8]
    p, q, r = x[9], x[10], x[11]
    u1, u2, u3, u4 = u
    cos_theta = jnp.cos(theta)
    safe_cos_theta = jnp.where(
        jnp.abs(cos_theta) < 1e-6,
        jnp.where(cos_theta == 0, 1e-6, 1e-6 * jnp.sign(cos_theta)),
        cos_theta,
    )
    xdot = jnp.array(
        [
            x_dot,
            y_dot,
            z_dot,
            q * jnp.sin(phi) / safe_cos_theta + r * jnp.cos(phi) / safe_cos_theta,
            q * jnp.cos(phi) - r * jnp.sin(phi),
            p + q * jnp.sin(phi) * jnp.tan(theta) + r * jnp.cos(phi) * jnp.tan(theta),
            u1
            / MASS
            * (
                jnp.sin(phi) * jnp.sin(psi)
                + jnp.cos(phi) * jnp.cos(psi) * jnp.sin(theta)
            ),
            u1
            / MASS
            * (
                jnp.cos(psi) * jnp.sin(phi)
                - jnp.cos(phi) * jnp.sin(psi) * jnp.sin(theta)
            ),
            GRAV + u1 / MASS * (jnp.cos(phi) * jnp.cos(theta)),
            ((0.1 - 0.3) / 0.5) * q * r + u2 / 0.5,
            ((0.3 - 0.5) / 0.1) * p * r + u3 / 0.1,
            ((0.5 - 0.1) / 0.3) * p * q + u4 / 0.3,
        ],
        dtype=JAX_DTYPE,
    )
    return x + dt * xdot


def rollout_true_dynamics_jax(x0, u_sequence, dt):
    def step(x, u):
        x_next = quadrotor_dynamics_jax(x, u, dt)
        return x_next, x_next

    _, xs = lax.scan(step, x0, u_sequence)
    return jnp.concatenate([x0[None, :], xs], axis=0)
