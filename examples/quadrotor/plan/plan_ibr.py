"""Ego-frame batched distributed IBR-MPC in a configurable workspace.

Each quadrotor treats the other quadrotors' most recent predicted trajectories
as moving spherical keep-out regions.  All best responses in an IBR sweep are
computed simultaneously (Jacobi IBR), and all environments/robots are flattened
into one JAX batch.  Robot count is therefore static for each invocation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from micrompc.examples.quadrotor.shared.runtime import configure_jax_logging

configure_jax_logging()

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
from tqdm import tqdm

from micrompc.examples.quadrotor.shared import planning as base


NP_DTYPE, JAX_DTYPE = np.float32, jnp.float32


from micrompc.examples.quadrotor.shared.multi_robot_solver import (
    MultiRobotPrimalDualMPC,
)


from micrompc.examples.quadrotor.shared.config import (
    add_planning_args,
    parse_configured_args,
    validate_planning_args,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-robots", type=int)
    parser.add_argument("--ibr-iterations", type=int)
    parser.add_argument("--ibr-relaxation", type=float)
    parser.add_argument("--quad-radius", type=float)
    parser.add_argument("--model-error-buffer", type=float)
    parser.add_argument(
        "--endpoint-clearance",
        type=float,
        help="minimum distance among every sampled start and goal [m]",
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--max-mpc-steps", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--precision", type=int, choices=(32, 64))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--sqp-iterations", type=int)
    parser.add_argument("--pd-iterations", type=int)
    parser.add_argument("--pd-mode", choices=("fixed", "converged"))
    parser.add_argument("--pd-primal-tol", type=float)
    parser.add_argument("--pd-fixed-point-tol", type=float)
    parser.add_argument("--pd-check-interval", type=int)
    parser.add_argument(
        "--timing-warmup-cycles",
        type=int,
        help="initial MPC cycles excluded from steady-state timing statistics",
    )
    parser.add_argument("--results-path", type=Path)
    parser.add_argument("--video", action=argparse.BooleanOptionalAction)
    parser.add_argument("--video-path", type=Path)
    parser.add_argument("--video-stride", type=int)
    parser.add_argument(
        "--collision-plot-path",
        type=Path,
        help="output PNG for swept minimum pairwise distance over time",
    )
    parser.add_argument(
        "--save-collision-plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save the collision-clearance PNG",
    )
    parser.add_argument(
        "--npz-path",
        type=Path,
        help="compressed run archive for post-hoc plots and videos",
    )
    parser.add_argument(
        "--save-npz",
        action=argparse.BooleanOptionalAction,
        help="save the full compressed run archive",
    )
    parser.add_argument("--disable-progress", action=argparse.BooleanOptionalAction)
    parser.add_argument("--mode", type=str, choices=("random", "left-to-right"))
    parser.add_argument("--max-yz-goal-displacement", type=float)
    parser.add_argument("--ko-window-size", type=float)
    parser.add_argument("--ibr-convergence-tol", type=float)
    parser.add_argument("--debug-numerics", action=argparse.BooleanOptionalAction)
    parser.add_argument("--debug-numerics-log", type=Path)
    parser.add_argument("--numerical-abs-limit", type=float)
    add_planning_args(parser)
    args = parse_configured_args(parser, Path(__file__).with_suffix(".yaml"))
    validate_planning_args(parser, args)
    positive = (
        args.batch_size,
        args.num_robots,
        args.ibr_iterations,
        args.horizon,
        args.max_mpc_steps,
        args.sqp_iterations,
        args.pd_iterations,
        args.pd_check_interval,
    )
    if any(value <= 0 for value in positive):
        parser.error(
            "batch, robot, IBR, horizon, MPC, SQP, and PD counts must be positive"
        )
    if args.timing_warmup_cycles < 0:
        parser.error("--timing-warmup-cycles cannot be negative")
    if (
        args.quad_radius <= 0.0
        or args.model_error_buffer < 0.0
        or args.endpoint_clearance <= 0.0
    ):
        parser.error(
            "quad radius and endpoint clearance must be positive; model-error buffer must be nonnegative"
        )
    if not 0.0 < args.ibr_relaxation <= 1.0:
        parser.error("--ibr-relaxation must be in (0, 1]")
    if args.ko_window_size <= 0 or any(
        np.asarray(args.workspace_upper) - args.workspace_lower < args.ko_window_size
    ):
        parser.error("KO window must be positive and fit in every workspace dimension")
    if (
        min(args.ibr_convergence_tol, args.numerical_abs_limit) <= 0
        or args.max_yz_goal_displacement < 0
    ):
        parser.error("invalid IBR tolerance, numerical limit, or goal displacement")
    if args.video_stride <= 0:
        parser.error("--video-stride must be positive")
    return args


def _sample_separated_points(rng, count, clearance, lower, upper, forbidden=()):
    lower = np.asarray(lower) + clearance / 2.0
    upper = np.asarray(upper) - clearance / 2.0
    points = [np.asarray(point, dtype=NP_DTYPE) for point in forbidden]
    initial_count = len(points)
    for _ in range(100_000):
        if len(points) == initial_count + count:
            return np.stack(points[initial_count:])
        candidate = rng.uniform(lower, upper).astype(NP_DTYPE)
        if all(np.linalg.norm(candidate - point) >= clearance for point in points):
            points.append(candidate)
    raise RuntimeError(
        "could not sample separated endpoints; reduce robot count or endpoint clearance"
    )


def sample_environments(
    rng,
    batch_size,
    robot_count,
    clearance,
    *,
    mode="random",
    lower=(0.0, 0.0, 0.0),
    upper=(25.0, 25.0, 25.0),
    max_displacement=5.0,
):
    if mode == "left-to-right":
        return sample_left_to_right_environments(
            rng, batch_size, robot_count, clearance, lower, upper, max_displacement
        )
    starts = np.zeros((batch_size, robot_count, 12), dtype=NP_DTYPE)
    goals = np.zeros_like(starts)
    for env in range(batch_size):
        starts[env, :, :3] = _sample_separated_points(
            rng, robot_count, clearance, lower, upper
        )
        goals[env, :, :3] = _sample_separated_points(
            rng, robot_count, clearance, lower, upper, forbidden=starts[env, :, :3]
        )
    return starts, goals


def _sample_grid_yz(rng, count, clearance, lower, upper):
    """Randomly select cells from the smallest square grid that fits `count`."""
    lower = np.asarray(lower)[1:]
    upper = np.asarray(upper)[1:]
    side = int(np.ceil(np.sqrt(count)))
    spacing = np.min((upper - lower) / max(side - 1, 1))
    if count > 1 and spacing < clearance:
        raise RuntimeError(
            "Y-Z grid spacing is smaller than endpoint clearance; "
            "reduce robot count or clearance"
        )
    y = np.linspace(lower[0], upper[0], side, dtype=NP_DTYPE)
    z = np.linspace(lower[1], upper[1], side, dtype=NP_DTYPE)
    grid = np.stack(np.meshgrid(y, z, indexing="ij"), axis=-1).reshape(-1, 2)
    return grid[rng.permutation(len(grid))[:count]]


def _locally_permute_goals(rng, starts_yz, max_displacement):
    """Randomize unique goal cells while preserving a displacement bound."""
    goals_yz = starts_yz.copy()
    count = len(starts_yz)
    for _ in range(20 * count):
        first, second = rng.integers(0, count, size=2)
        if first == second:
            continue
        if (
            np.linalg.norm(goals_yz[second] - starts_yz[first]) <= max_displacement
            and np.linalg.norm(goals_yz[first] - starts_yz[second]) <= max_displacement
        ):
            goals_yz[[first, second]] = goals_yz[[second, first]]
    return goals_yz


def sample_left_to_right_environments(
    rng, batch_size, robot_count, clearance, lower, upper, max_displacement
):
    starts = np.zeros((batch_size, robot_count, 12), dtype=NP_DTYPE)
    goals = np.zeros_like(starts)
    for environment in range(batch_size):
        starts_yz = _sample_grid_yz(rng, robot_count, clearance, lower, upper)
        starts[environment, :, 0] = lower[0]
        starts[environment, :, 1:3] = starts_yz
        goals[environment, :, 0] = upper[0]
        goals[environment, :, 1:3] = _locally_permute_goals(
            rng, starts_yz, max_displacement
        )
    return starts, goals


def build_obstacles(plans, active, states):
    env_count, robot_count, horizon = plans.shape[:3]
    centers = np.empty(
        (env_count, robot_count, horizon, robot_count - 1, 3), dtype=NP_DTYPE
    )
    for robot in range(robot_count):
        others = [index for index in range(robot_count) if index != robot]
        if others:
            other_plans = plans[:, others, :, :3].copy()
            for env in range(env_count):
                for slot, other in enumerate(others):
                    if not active[env, other]:
                        other_plans[env, slot] = states[env, other, :3]
            centers[:, robot] = np.moveaxis(other_plans, 1, 2)
    return centers


def build_obstacles_jax(plans, active, states, other_indices):
    """Gather every robot's moving obstacles without leaving the accelerator."""
    other_plans = plans[:, other_indices, :, :3]
    other_active = active[:, other_indices]
    stationary = states[:, other_indices, :3][..., None, :]
    other_plans = jnp.where(other_active[..., None, None], other_plans, stationary)
    return jnp.swapaxes(other_plans, 2, 3)


def swept_pairwise_minimum_jax(previous, current, valid_robots=None):
    """Return swept distances, optionally excluding invalid numerical rows."""
    robot_count = previous.shape[1]
    if robot_count < 2:
        per_robot = jnp.full(previous.shape[:2], jnp.inf, dtype=previous.dtype)
        return per_robot[:, 0], per_robot
    first, second = jnp.triu_indices(robot_count, 1)
    relative_start = previous[:, first, :3] - previous[:, second, :3]
    relative_delta = (current[:, first, :3] - previous[:, first, :3]) - (
        current[:, second, :3] - previous[:, second, :3]
    )
    denominator = jnp.sum(relative_delta**2, axis=-1)
    alpha = jnp.clip(
        -jnp.sum(relative_start * relative_delta, axis=-1)
        / jnp.maximum(denominator, 1.0e-12),
        0.0,
        1.0,
    )
    distance = jnp.linalg.norm(
        relative_start + alpha[..., None] * relative_delta, axis=-1
    )
    if valid_robots is not None:
        valid_pairs = valid_robots[:, first] & valid_robots[:, second]
        distance = jnp.where(valid_pairs, distance, jnp.inf)
    per_robot = jnp.full(previous.shape[:2], jnp.inf, dtype=previous.dtype)
    per_robot = per_robot.at[:, first].min(distance)
    per_robot = per_robot.at[:, second].min(distance)
    return jnp.min(distance, axis=1), per_robot


def swept_pairwise_minimum(previous, current):
    if previous.shape[1] < 2:
        return np.full(previous.shape[0], np.inf, dtype=NP_DTYPE)
    result = np.full(previous.shape[0], np.inf, dtype=NP_DTYPE)
    for i in range(previous.shape[1]):
        for j in range(i + 1, previous.shape[1]):
            relative_start = previous[:, i, :3] - previous[:, j, :3]
            relative_delta = (current[:, i, :3] - previous[:, i, :3]) - (
                current[:, j, :3] - previous[:, j, :3]
            )
            denominator = np.sum(relative_delta**2, axis=-1)
            alpha = np.clip(
                -np.sum(relative_start * relative_delta, axis=-1)
                / np.maximum(denominator, 1.0e-12),
                0.0,
                1.0,
            )
            distance = np.linalg.norm(
                relative_start + alpha[:, None] * relative_delta, axis=-1
            )
            result = np.minimum(result, distance)
    return result


def distribution(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {
            "mean": None,
            "median": None,
            "std": None,
            "robust_std": None,
            "p10": None,
            "p90": None,
            "min": None,
            "max": None,
            "count": 0,
        }
    result = {
        name: float(function(finite))
        for name, function in (
            ("mean", np.mean),
            ("median", np.median),
            ("std", np.std),
            ("min", np.min),
            ("max", np.max),
        )
    }
    # Gaussian-consistent MAD: representative of ordinary cycle-to-cycle
    # jitter without silently discarding occasional runtime/driver outliers.
    result["robust_std"] = float(1.4826 * np.median(np.abs(finite - result["median"])))
    result["p10"], result["p90"] = (
        float(value) for value in np.percentile(finite, (10, 90))
    )
    result["count"] = int(finite.size)
    return result


def save_3d_video(
    position_history, goals, quad_radius, path, frame_stride, *, lower, upper, dt
):
    lower, upper = np.asarray(lower), np.asarray(upper)
    """Save environment zero with physical-radius drone spheres in a 10 m cube."""
    import matplotlib.animation as mpl_animation
    import matplotlib.pyplot as plt

    positions = np.asarray(position_history)[:, 0, :, :3]
    frame_indices = np.arange(0, len(positions), frame_stride)
    if frame_indices[-1] != len(positions) - 1:
        frame_indices = np.append(frame_indices, len(positions) - 1)
    positions = positions[frame_indices]

    figure = plt.figure(figsize=(8, 8))
    axis = figure.add_subplot(111, projection="3d")
    axis.set(
        xlim=(lower[0], upper[0]),
        ylim=(lower[1], upper[1]),
        zlim=(lower[2], upper[2]),
        xlabel="X [m]",
        ylabel="Y [m]",
        zlabel="Z [m]",
    )
    axis.set_box_aspect(upper - lower)
    axis.set_title("Quadrotor MPC")
    colors = plt.colormaps["tab20"](np.linspace(0.0, 1.0, positions.shape[1]))
    axis.scatter(
        goals[0, :, 0],
        goals[0, :, 1],
        goals[0, :, 2],
        marker="x",
        c=colors,
        s=35,
        alpha=0.7,
    )
    trails = [
        axis.plot([], [], [], color=colors[index], linewidth=1.0, alpha=0.7)[0]
        for index in range(positions.shape[1])
    ]
    azimuth = np.linspace(0.0, 2.0 * np.pi, 12)
    elevation = np.linspace(0.0, np.pi, 7)
    sphere_x = quad_radius * np.outer(np.cos(azimuth), np.sin(elevation))
    sphere_y = quad_radius * np.outer(np.sin(azimuth), np.sin(elevation))
    sphere_z = quad_radius * np.outer(np.ones_like(azimuth), np.cos(elevation))
    surfaces = []

    def animate(frame):
        nonlocal surfaces
        for surface in surfaces:
            surface.remove()
        surfaces = []
        for robot, color in enumerate(colors):
            center = positions[frame, robot]
            surfaces.append(
                axis.plot_surface(
                    sphere_x + center[0],
                    sphere_y + center[1],
                    sphere_z + center[2],
                    color=color,
                    linewidth=0.0,
                    shade=True,
                    alpha=0.9,
                )
            )
            trail = positions[: frame + 1, robot]
            trails[robot].set_data(trail[:, 0], trail[:, 1])
            trails[robot].set_3d_properties(trail[:, 2])
        return (*trails, *surfaces)

    movie = mpl_animation.FuncAnimation(
        figure,
        animate,
        frames=len(positions),
        interval=1000.0 * dt * frame_stride,
        blit=False,
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fps = max(1, int(round(1.0 / (dt * frame_stride))))
    writer = mpl_animation.FFMpegWriter(
        fps=fps, codec="libx264", extra_args=["-preset", "fast", "-pix_fmt", "yuv420p"]
    )
    movie.save(path, writer=writer)
    plt.close(figure)
    print(f"Saved video to {path}")


def save_collision_clearance_plot(min_distances, quad_radius, path, *, dt):
    """Plot swept minimum robot separation for every environment over time."""
    import matplotlib.pyplot as plt

    distances = np.asarray(min_distances, dtype=float)
    if distances.size == 0:
        print("Could not save collision plot: no dynamics steps were simulated")
        return
    if distances.ndim == 1:
        distances = distances[:, None]

    times = dt * np.arange(1, distances.shape[0] + 1)
    collision_distance = 2.0 * quad_radius
    figure, axis = plt.subplots(figsize=(9, 5))
    for environment in range(distances.shape[1]):
        axis.plot(
            times,
            distances[:, environment],
            linewidth=1.0,
            alpha=0.4,
            label=(f"environment {environment}" if distances.shape[1] <= 10 else None),
        )
    if distances.shape[1] > 1:
        axis.plot(
            times,
            np.min(distances, axis=1),
            color="black",
            linewidth=2.0,
            label="minimum across environments",
        )
    axis.axhline(
        collision_distance,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"collision threshold ({collision_distance:.3f} m)",
    )
    axis.fill_between(times, 0.0, collision_distance, color="red", alpha=0.08)
    axis.set(
        xlabel="Simulation time [s]",
        ylabel="Swept minimum pairwise distance [m]",
        title="Inter-robot collision clearance over time",
    )
    axis.set_ylim(bottom=0.0)
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    print(f"Saved collision-clearance plot to {path}")


def _append_debug_event(path, event):
    """Append one immediately flushed JSONL event, permitting NaN/Inf markers."""

    def json_default(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(
            f"Object of type {type(value).__name__} is not JSON serializable"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(event, allow_nan=True, default=json_default) + "\n")
        stream.flush()


def _array_debug_summary(name, value):
    """Copy only scalar diagnostics (not large planner arrays) from the device."""
    array = jnp.asarray(value)
    flat = array.reshape(-1)
    finite = jnp.isfinite(flat)
    invalid = ~finite
    diagnostics = jnp.stack(
        (
            jnp.sum(finite),
            jnp.sum(jnp.isnan(flat)),
            jnp.sum(jnp.isinf(flat)),
            jnp.argmax(invalid),
        )
    ).astype(jnp.int32)
    finite_abs = jnp.where(finite, jnp.abs(flat), -jnp.inf)
    max_abs = jnp.max(finite_abs, initial=-jnp.inf)
    minimum = jnp.min(jnp.where(finite, flat, jnp.inf), initial=jnp.inf)
    maximum = jnp.max(jnp.where(finite, flat, -jnp.inf), initial=-jnp.inf)
    count, nan_count, inf_count, first_flat = (
        int(item) for item in np.asarray(diagnostics)
    )
    result = {
        "name": name,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "size": int(array.size),
        "finite_count": count,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "finite_min": float(np.asarray(minimum)),
        "finite_max": float(np.asarray(maximum)),
        "finite_max_abs": float(np.asarray(max_abs)),
    }
    if count != array.size:
        result["first_invalid_flat_index"] = first_flat
        result["first_invalid_index"] = list(np.unravel_index(first_flat, array.shape))
        result["first_invalid_value"] = float(np.asarray(flat[first_flat]))
    return result


def main():
    global NP_DTYPE, JAX_DTYPE
    args = parse_args()
    base.validate_cost_parameters(args.q_state, args.q_terminal)
    NP_DTYPE, JAX_DTYPE = base.configure_precision(args.precision)
    device = base.select_jax_device(args.device)
    if args.device == "cuda" and device.platform != "gpu":
        raise SystemExit("CUDA requested, but JAX did not find a GPU device")
    model, normalizer, normalized = base.load_checkpoint_jax(args.model_path, device)
    decoder_weight, decoder_bias = base.physical_decoder(model, normalizer, normalized)
    obstacle_count = args.num_robots - 1
    pairwise_clearance = 2.0 * args.quad_radius + args.model_error_buffer
    mpc = MultiRobotPrimalDualMPC(
        model["A"],
        model["B"],
        decoder_weight,
        decoder_bias,
        r_val=args.control_weight,
        u_min=np.asarray(args.u_min),
        u_max=np.asarray(args.u_max),
        horizon=args.horizon,
        rho=args.pd_rho,
        n_pd_steps=args.pd_iterations,
        device=device,
        cache_dir=args.cache_dir,
        control_reference=args.control_reference,
        q_state=np.asarray(args.q_state),
        q_terminal=np.asarray(args.q_terminal),
        obstacle_center=np.zeros(3, dtype=NP_DTYPE),
        obstacle_radius=np.finfo(NP_DTYPE).eps,
        model_error_buffer=0.0,
        sqp_iterations=args.sqp_iterations,
        pd_mode=args.pd_mode,
        pd_primal_tol=args.pd_primal_tol,
        pd_fixed_point_tol=args.pd_fixed_point_tol,
        pd_check_interval=args.pd_check_interval,
        pd_step_safety=args.pd_step_safety,
        pd_extrapolation=args.pd_extrapolation,
        num_obstacles=obstacle_count,
    )

    def encode_one(state):
        return base.encoder_forward_jax(
            base.preprocess_state_jax(
                state, normalizer["min"], normalizer["range"], normalized
            ),
            model["encoder_params"],
            model["activation_fn"],
        )

    encode_batch = jax.jit(jax.vmap(encode_one))
    # The complete IBR cycle below is jitted, so keep these transformations
    # un-jitted here to avoid nested executable boundaries.
    solve_batch = jax.vmap(mpc._build_multi_obstacle_solve_impl())
    seed_batch = jax.jit(jax.vmap(mpc.seed_reference))
    dynamics_batch = jax.jit(
        jax.vmap(
            lambda state, control: base.quadrotor_dynamics_jax(state, control, args.dt)
        )
    )

    rng = np.random.default_rng(args.seed)
    starts, goals_np = sample_environments(
        rng,
        args.batch_size,
        args.num_robots,
        args.endpoint_clearance,
        mode=args.mode,
        lower=args.workspace_lower,
        upper=args.workspace_upper,
        max_displacement=args.max_yz_goal_displacement,
    )
    states = jax.device_put(jnp.asarray(starts, dtype=JAX_DTYPE), device)
    env_count, robot_count = states.shape[:2]
    flat_count = env_count * robot_count
    goals = jax.device_put(
        jnp.asarray(goals_np.reshape(flat_count, 12), dtype=JAX_DTYPE), device
    )
    flat_starts = jax.device_put(
        jnp.asarray(starts.reshape(flat_count, 12), dtype=JAX_DTYPE), device
    )
    initial_offsets = jnp.clip(
        flat_starts[:, :3] - (args.ko_window_size / 2.0),
        np.asarray(args.workspace_lower),
        np.asarray(args.workspace_upper) - args.ko_window_size,
    )
    initial_local_starts = flat_starts[:, :3] - initial_offsets
    initial_local_goals = jnp.clip(
        goals[:, :3] - initial_offsets, 0.0, args.ko_window_size
    )
    reference = (
        seed_batch(initial_local_starts, initial_local_goals)
        + initial_offsets[:, None, :]
    )
    plans = jnp.zeros((env_count, robot_count, args.horizon, 3), dtype=JAX_DTYPE)
    plans = plans.at[..., :3].set(
        reference.reshape(env_count, robot_count, args.horizon, 3)
    )

    warm_u = jnp.broadcast_to(mpc.warm_u, (flat_count, mpc.n_controls))
    warm_upper_dual = jnp.broadcast_to(
        mpc.warm_upper_dual, (flat_count, mpc.n_controls)
    )
    warm_lower_dual = jnp.broadcast_to(
        mpc.warm_lower_dual, (flat_count, mpc.n_controls)
    )
    warm_obstacle_dual = jnp.broadcast_to(
        mpc.warm_obstacle_dual, (flat_count, args.horizon, obstacle_count)
    )
    warm_obstacle_aux = jnp.zeros_like(warm_obstacle_dual)
    active = np.ones((env_count, robot_count), dtype=bool)
    reached = np.zeros_like(active)
    numerically_failed = np.zeros_like(active)
    numerical_failure_step = np.full(active.shape, -1, dtype=np.int32)
    collided_robots = np.zeros((env_count, robot_count), dtype=bool)
    steps = np.zeros_like(active, dtype=np.int32)
    solve_times, sweep_history, min_distances, robot_min_distances = [], [], [], []
    retain_states = args.save_npz or (args.plot and args.video)
    retain_clearances = args.save_npz or (args.plot and args.save_collision_plot)
    state_history = [starts.copy()] if retain_states else []
    active_history = [active.copy()] if args.save_npz else []
    reached_history = [reached.copy()] if args.save_npz else []

    radii = jnp.full((flat_count, obstacle_count), pairwise_clearance, dtype=JAX_DTYPE)
    other_indices = np.asarray(
        [
            [other for other in range(robot_count) if other != robot]
            for robot in range(robot_count)
        ],
        dtype=np.int32,
    )
    other_indices = jax.device_put(jnp.asarray(other_indices), device)
    sweep_count = 1 if robot_count == 1 else args.ibr_iterations
    relaxation = 1.0 if robot_count == 1 else args.ibr_relaxation

    @jax.jit
    def run_mpc_cycle(
        states,
        plans,
        warm_u,
        warm_upper_dual,
        warm_lower_dual,
        warm_obstacle_dual,
        warm_obstacle_aux,
        reference,
        active,
        collision_valid,
    ):
        def rows_are_valid(*arrays):
            valid = jnp.ones(flat_count, dtype=bool)
            for array in arrays:
                rows = array.reshape(flat_count, -1)
                valid &= jnp.all(jnp.isfinite(rows), axis=1)
                valid &= jnp.all(jnp.abs(rows) <= args.numerical_abs_limit, axis=1)
            return valid.reshape(env_count, robot_count)

        if not args.warm_start:
            warm_u = jnp.broadcast_to(mpc.u_hover_tiled, warm_u.shape)
            warm_upper_dual = jnp.zeros_like(warm_upper_dual)
            warm_lower_dual = jnp.zeros_like(warm_lower_dual)
            warm_obstacle_dual = jnp.zeros_like(warm_obstacle_dual)
            warm_obstacle_aux = jnp.zeros_like(warm_obstacle_aux)
        flat_states = states.reshape(flat_count, 12)
        # Give every KO rollout its own translated 5 m observation window.
        # Only position is translated; velocity, attitude, and controls remain
        # in the common world frame.
        ego_offsets = jnp.clip(
            flat_states[:, :3] - (args.ko_window_size / 2.0),
            np.asarray(args.workspace_lower),
            np.asarray(args.workspace_upper) - args.ko_window_size,
        )
        local_states = flat_states.at[:, :3].add(-ego_offsets)
        local_goals = goals.at[:, :3].set(
            jnp.clip(goals[:, :3] - ego_offsets, 0.0, args.ko_window_size)
        )
        z0 = encode_batch(local_states)
        goal_y = local_goals
        active_mask = active[..., None, None]
        current_plans = jnp.where(active_mask, plans, states[:, :, None, :3])
        current_controls = warm_u.reshape(env_count, robot_count, args.horizon, 4)
        current_controls = jnp.where(active_mask, current_controls, 0.0)

        # Result slots have their production shapes from the outset, which lets
        # XLA keep the complete convergence-controlled IBR loop on device.
        initial = (
            jnp.asarray(0, jnp.int32),
            jnp.asarray(jnp.inf, JAX_DTYPE),
            current_plans,
            current_controls,
            warm_upper_dual,
            warm_lower_dual,
            warm_obstacle_dual,
            warm_obstacle_aux,
            reference,
            jnp.zeros(flat_count, jnp.int32),
            active,
        )

        def continue_ibr(value):
            iteration, max_change = value[0], value[1]
            return (iteration < sweep_count) & (max_change > args.ibr_convergence_tol)

        def ibr_sweep(value):
            iteration, _, current_plans, current_controls = value[:4]
            healthy = value[10]
            before = current_plans[..., :3]
            centers = build_obstacles_jax(current_plans, healthy, states, other_indices)
            offsets = ego_offsets.reshape(env_count, robot_count, 3)
            local_reference = current_plans[..., :3] - offsets[:, :, None, :]
            local_centers = centers - offsets[:, :, None, None, :]
            result = solve_batch(
                current_controls.reshape(flat_count, -1),
                warm_upper_dual,
                warm_lower_dual,
                warm_obstacle_dual,
                warm_obstacle_aux,
                local_reference.reshape(flat_count, args.horizon, 3),
                local_centers.reshape(flat_count, args.horizon, obstacle_count, 3),
                radii,
                z0,
                goal_y,
            )
            candidates = result[0].reshape(env_count, robot_count, args.horizon, 3)
            candidates = candidates.at[..., :3].add(offsets[:, :, None, :])
            controls = result[1].reshape(env_count, robot_count, args.horizon, 4)
            candidate_healthy = rows_are_valid(
                candidates,
                controls,
                result[3],
                result[4],
                result[5],
                result[6],
                result[7],
            )
            healthy &= candidate_healthy
            healthy_mask = healthy[..., None, None]
            flat_healthy = healthy.reshape(-1)
            stationary_plans = jnp.broadcast_to(
                states[:, :, None, :3], current_plans.shape
            )
            relaxed_plans = (1.0 - relaxation) * current_plans + relaxation * candidates
            relaxed_controls = (
                1.0 - relaxation
            ) * current_controls + relaxation * controls
            current_plans = jnp.where(healthy_mask, relaxed_plans, stationary_plans)
            current_controls = jnp.where(healthy_mask, relaxed_controls, 0.0)
            next_s = jnp.where(flat_healthy[:, None], result[3], mpc.warm_upper_dual)
            next_eta = jnp.where(flat_healthy[:, None], result[4], mpc.warm_lower_dual)
            next_obs_s = jnp.where(
                flat_healthy[:, None, None], result[5], mpc.warm_obstacle_dual
            )
            next_obs_eta = jnp.where(
                flat_healthy[:, None, None], result[6], mpc.warm_obstacle_aux
            )
            stationary_reference = jnp.broadcast_to(
                flat_states[:, None, :3], (flat_count, args.horizon, 3)
            )
            next_reference = jnp.where(
                flat_healthy[:, None, None], result[7], stationary_reference
            )
            changes = jnp.linalg.norm(current_plans[..., :3] - before, axis=-1)
            max_change = jnp.max(jnp.where(healthy[..., None], changes, -jnp.inf))
            return (
                iteration + 1,
                max_change,
                current_plans,
                current_controls,
                next_s,
                next_eta,
                next_obs_s,
                next_obs_eta,
                next_reference,
                result[8],
                healthy,
            )

        result = lax.while_loop(continue_ibr, ibr_sweep, initial)
        iterations, max_change, current_plans, current_controls = result[:4]
        final_s, final_eta, final_obs_s, final_obs_eta = result[4:8]
        pd_counts = result[9]
        healthy = result[10]
        shifted_controls = jnp.concatenate(
            (current_controls[:, :, 1:], current_controls[:, :, -1:]), axis=2
        )
        shifted_reference = jnp.concatenate(
            (current_plans[:, :, 1:, :3], current_plans[:, :, -1:, :3]), axis=2
        )
        flat_healthy = healthy.reshape(-1)
        warm_u = jnp.where(
            flat_healthy[:, None], shifted_controls.reshape(flat_count, -1), mpc.warm_u
        )
        warm_upper_dual = jnp.where(flat_healthy[:, None], final_s, mpc.warm_upper_dual)
        warm_lower_dual = jnp.where(
            flat_healthy[:, None], final_eta, mpc.warm_lower_dual
        )
        warm_obstacle_dual = jnp.where(
            flat_healthy[:, None, None], final_obs_s, mpc.warm_obstacle_dual
        )
        warm_obstacle_aux = jnp.where(
            flat_healthy[:, None, None], final_obs_eta, mpc.warm_obstacle_aux
        )
        stationary_reference = jnp.broadcast_to(
            flat_states[:, None, :3], (flat_count, args.horizon, 3)
        )
        reference = jnp.where(
            flat_healthy[:, None, None],
            shifted_reference.reshape(flat_count, args.horizon, 3),
            stationary_reference,
        )
        proposed = dynamics_batch(
            flat_states, current_controls.reshape(flat_count, args.horizon, 4)[:, 0]
        )
        proposed = proposed.reshape(env_count, robot_count, 12)
        dynamics_healthy = rows_are_valid(proposed)
        healthy &= dynamics_healthy
        newly_failed = active & ~healthy
        next_states = jnp.where(healthy[..., None], proposed, states)
        current_plans = jnp.where(
            healthy[..., None, None],
            current_plans,
            jnp.broadcast_to(states[:, :, None, :3], current_plans.shape),
        )
        collision_valid &= ~newly_failed
        minimum, robot_minimum = swept_pairwise_minimum_jax(
            states, next_states, collision_valid
        )
        return (
            next_states,
            current_plans,
            warm_u,
            warm_upper_dual,
            warm_lower_dual,
            warm_obstacle_dual,
            warm_obstacle_aux,
            reference,
            minimum,
            robot_minimum,
            iterations,
            max_change,
            pd_counts,
            newly_failed,
        )

    # Compile the exact production shape before timing.
    compiled = run_mpc_cycle(
        states,
        plans,
        warm_u,
        warm_upper_dual,
        warm_lower_dual,
        warm_obstacle_dual,
        warm_obstacle_aux,
        reference,
        jnp.asarray(active),
        jnp.asarray(~numerically_failed),
    )
    compiled[0].block_until_ready()
    print(
        f"Running {env_count} environments x {robot_count} robots ({flat_count} solves/sweep) on {device}"
    )
    print(
        f"Physical radius={args.quad_radius:.3f} m; planning clearance={pairwise_clearance:.3f} m"
    )

    progress = tqdm(
        range(args.max_mpc_steps),
        desc="Batched quadrotor IBR-MPC",
        disable=args.disable_progress,
    )
    for mpc_step in progress:
        states_np = np.asarray(states)
        distances = np.linalg.norm(goals_np[..., :3] - states_np[..., :3], axis=-1)
        newly_reached = active & (distances <= args.goal_tol)
        reached |= newly_reached
        active &= ~newly_reached
        if not active.any():
            break

        cycle_start = time.perf_counter()
        result = run_mpc_cycle(
            states,
            plans,
            warm_u,
            warm_upper_dual,
            warm_lower_dual,
            warm_obstacle_dual,
            warm_obstacle_aux,
            reference,
            jnp.asarray(active),
            jnp.asarray(~numerically_failed),
        )
        result[0].block_until_ready()
        solve_times.append(1.0e3 * (time.perf_counter() - cycle_start))
        (
            states,
            plans,
            warm_u,
            warm_upper_dual,
            warm_lower_dual,
            warm_obstacle_dual,
            warm_obstacle_aux,
            reference,
            minimum_device,
            robot_minimum_device,
            sweep_device,
            max_change_device,
            pd_counts_device,
            newly_failed_device,
        ) = result
        newly_failed = np.asarray(newly_failed_device)
        if newly_failed.any():
            numerically_failed |= newly_failed
            numerical_failure_step[newly_failed] = mpc_step
            active &= ~newly_failed
            failed_indices = np.argwhere(newly_failed)
            if args.debug_numerics:
                _append_debug_event(
                    args.debug_numerics_log,
                    {
                        "event": "robots_quarantined",
                        "mpc_step": mpc_step,
                        "seed": args.seed,
                        "failed_environment_robot_indices": failed_indices,
                        "max_change": float(np.asarray(max_change_device)),
                        "pd_iterations": np.asarray(pd_counts_device),
                        "arrays": [
                            _array_debug_summary("states_after_quarantine", states),
                            _array_debug_summary("plans_after_quarantine", plans),
                            _array_debug_summary("warm_u_after_quarantine", warm_u),
                            _array_debug_summary(
                                "reference_after_quarantine", reference
                            ),
                        ],
                    },
                )
        if retain_states:
            state_history.append(np.asarray(states).copy())
        sweep = int(np.asarray(sweep_device))
        sweep_history.append(sweep)
        if retain_clearances:
            min_distances.append(np.asarray(minimum_device))
        robot_minimum = np.asarray(robot_minimum_device)
        if args.save_npz:
            robot_min_distances.append(robot_minimum)
        collided_robots |= robot_minimum < 2.0 * args.quad_radius - 1.0e-6
        steps[active] += 1
        if args.save_npz:
            active_history.append(active.copy())
            reached_history.append(reached.copy())
        progress.set_postfix(
            active=int(active.sum()),
            reached=f"{100 * reached.mean():.1f}%",
            collided_robots=int(collided_robots.sum()),
            sweeps=sweep,
            batch_ms=f"{solve_times[-1]:.1f}",
        )

    states_np = np.asarray(states)
    final_distances = np.linalg.norm(states_np[..., :3] - goals_np[..., :3], axis=-1)
    # Include robots that enter the goal ball on the final permitted dynamics step.
    reached |= final_distances <= args.goal_tol
    collided_per_environment = collided_robots.sum(axis=1)
    success = reached.all(axis=1) & (collided_per_environment == 0)
    steady_state_solve_times = solve_times[args.timing_warmup_cycles :]
    summary = {
        "batch_size": env_count,
        "num_robots": robot_count,
        "ibr_iterations": args.ibr_iterations,
        "ibr_relaxation": args.ibr_relaxation,
        "pd_check_interval": args.pd_check_interval,
        "model_path": str(args.model_path),
        "horizon": args.horizon,
        "max_mpc_steps": args.max_mpc_steps,
        "sqp_iterations": args.sqp_iterations,
        "pd_iterations": args.pd_iterations,
        "pd_mode": args.pd_mode,
        "pd_primal_tol": args.pd_primal_tol,
        "pd_fixed_point_tol": args.pd_fixed_point_tol,
        "timing_warmup_cycles_excluded": args.timing_warmup_cycles,
        "requested_device": args.device,
        "jax_device": str(device),
        "precision": args.precision,
        "seed": args.seed,
        "quad_radius": args.quad_radius,
        "model_error_buffer": args.model_error_buffer,
        "endpoint_clearance": args.endpoint_clearance,
        "dt": args.dt,
        "goal_tolerance": args.goal_tol,
        "workspace_lower": np.asarray(args.workspace_lower).tolist(),
        "workspace_upper": np.asarray(args.workspace_upper).tolist(),
        "ko_ego_window_size": args.ko_window_size,
        "planning_pairwise_clearance": pairwise_clearance,
        "plot": args.plot,
        "video": args.plot and args.video,
        "video_path": str(args.video_path),
        "video_stride": args.video_stride,
        "collision_plot_path": str(args.collision_plot_path),
        "save_collision_plot": args.plot and args.save_collision_plot,
        "npz_path": str(args.npz_path),
        "save_npz": args.save_npz,
        "numerical_quarantine": True,
        "numerical_sanitization_abs_limit": args.numerical_abs_limit,
        "goal_reaching_rate_percent": float(100.0 * reached.mean()),
        "environment_success_rate_percent": float(100.0 * success.mean()),
        "collided_robots_per_environment": collided_per_environment.tolist(),
        "max_collided_robots_per_environment": int(
            collided_per_environment.max(initial=0)
        ),
        "whole_batch_ibr_cycle_ms": distribution(steady_state_solve_times),
        "whole_batch_ibr_cycle_including_warmup_ms": distribution(solve_times),
        "solve_time_samples_ms": steady_state_solve_times,
        "ibr_sweeps_per_step": distribution(sweep_history),
        "final_goal_distance": distribution(final_distances),
        "raw": {
            "goal_reached": reached.tolist(),
            "collided_robots": collided_robots.tolist(),
            "numerically_failed_robots": numerically_failed.tolist(),
            "numerical_failure_step": numerical_failure_step.tolist(),
            "success": success.tolist(),
            "robot_steps": steps.tolist(),
            "final_goal_distance": final_distances.tolist(),
            "ibr_sweeps": sweep_history,
            "minimum_pairwise_distance": [value.tolist() for value in min_distances],
        },
    }
    args.results_path.parent.mkdir(parents=True, exist_ok=True)
    args.results_path.write_text(json.dumps(summary, indent=2) + "\n")
    if args.save_npz:
        args.npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.npz_path,
            states=np.asarray(state_history),
            starts=starts,
            goals=goals_np,
            final_states=states_np,
            final_plans=np.asarray(plans),
            minimum_pairwise_distance=np.asarray(min_distances),
            per_robot_minimum_pairwise_distance=np.asarray(robot_min_distances),
            active=np.asarray(active_history),
            reached=np.asarray(reached_history),
            collided_robots=collided_robots,
            numerically_failed_robots=numerically_failed,
            numerical_failure_step=numerical_failure_step,
            robot_steps=steps,
            solve_times_ms=np.asarray(solve_times),
            ibr_sweeps=np.asarray(sweep_history),
            dt=np.asarray(args.dt),
            quad_radius=np.asarray(args.quad_radius),
            planning_pairwise_clearance=np.asarray(pairwise_clearance),
            workspace_lower=np.asarray(args.workspace_lower),
            workspace_upper=np.asarray(args.workspace_upper),
            q_state=np.asarray(args.q_state),
            q_terminal=np.asarray(args.q_terminal),
            metadata_json=np.asarray(json.dumps(summary)),
        )
        print(f"Saved run archive to {args.npz_path}")
    if args.plot and args.save_collision_plot:
        try:
            save_collision_clearance_plot(
                min_distances, args.quad_radius, args.collision_plot_path, dt=args.dt
            )
        except Exception as exc:
            print(f"Could not save collision plot: {exc}")
    if args.plot and args.video:
        try:
            save_3d_video(
                state_history,
                goals_np,
                args.quad_radius,
                args.video_path,
                args.video_stride,
                lower=args.workspace_lower,
                upper=args.workspace_upper,
                dt=args.dt,
            )
        except Exception as exc:
            print(f"Could not save video: {exc}")
    print(f"Goal reaching: {summary['goal_reaching_rate_percent']:.2f}%")
    print(f"Environment success: {summary['environment_success_rate_percent']:.2f}%")
    print(
        f"Max collided robots/environment: {summary['max_collided_robots_per_environment']}"
    )
    timing = summary["whole_batch_ibr_cycle_ms"]
    if timing["median"] is not None:
        print(
            f"Median steady-state solve time: {timing['median']:.3f} ± "
            f"{timing['robust_std']:.3f} ms"
        )
    print(f"Saved summary to {args.results_path}")


if __name__ == "__main__":
    main()
