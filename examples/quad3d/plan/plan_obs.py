"""Nominal condensed Koopman MPC with SQP-linearized obstacle avoidance.

Each SQP iteration
replaces the spherical keep-out set by tangent halfspaces along the current
nominal trajectory, then solves the resulting control-condensed QP with PD.
The obstacle is not part of the simulated dynamics; collision checks are
therefore geometric only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from micrompc.examples.quad3d.shared.runtime import configure_jax_logging

configure_jax_logging()

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from micrompc.examples.quad3d.shared import planning as base
from micrompc.examples.quad3d.shared import visualization as viz


NP_DTYPE, JAX_DTYPE = np.float32, jnp.float32


def sample_start_goal(rng: np.random.Generator, lower, upper):
    lower, upper = np.asarray(lower), np.asarray(upper)
    width = min(1.0, float(upper[0] - lower[0]))
    """Sample from opposite x faces with full workspace coverage in y and z."""
    start = np.zeros(12, dtype=NP_DTYPE)
    goal = np.zeros(12, dtype=NP_DTYPE)
    start[:3] = [
        rng.uniform(lower[0], lower[0] + width),
        rng.uniform(lower[1], upper[1]),
        rng.uniform(lower[2], upper[2]),
    ]
    goal[:3] = [
        rng.uniform(upper[0] - width, upper[0]),
        rng.uniform(lower[1], upper[1]),
        rng.uniform(lower[2], upper[2]),
    ]
    return start, goal


def segment_sphere_collision(p0, p1, center, radius):
    """Include swept segments so a discrete dynamics step cannot tunnel through."""
    p0 = np.asarray(p0)
    p1 = np.asarray(p1)
    delta = p1 - p0
    denom = np.sum(delta * delta, axis=-1)
    numer = np.sum((np.asarray(center) - p0) * delta, axis=-1)
    alpha = np.clip(numer / np.maximum(denom, 1.0e-12), 0.0, 1.0)
    closest = p0 + alpha[..., None] * delta
    return np.linalg.norm(closest - np.asarray(center), axis=-1) <= radius


def add_obstacle_to_axis(
    ax,
    radius,
    center,
):
    theta = np.linspace(0.0, 2.0 * np.pi, 32)
    phi = np.linspace(0.0, np.pi, 20)
    unit = np.stack(
        (
            np.outer(np.cos(theta), np.sin(phi)),
            np.outer(np.sin(theta), np.sin(phi)),
            np.outer(np.ones_like(theta), np.cos(phi)),
        )
    )
    sphere = np.asarray(center)[:, None, None] + radius * unit
    ax.plot_surface(*sphere, color="crimson", alpha=0.25, linewidth=0)


from micrompc.examples.quad3d.shared.obstacle_solver import ObstaclePrimalDualMPC


from micrompc.examples.quad3d.shared.config import (
    add_planning_args,
    parse_configured_args,
    validate_planning_args,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--max-mpc-steps", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--precision", type=int, choices=(32, 64))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--obstacle-radius", type=float)
    parser.add_argument("--model-error-buffer", type=float)
    parser.add_argument("--sqp-iterations", type=int)
    parser.add_argument("--pd-iterations", type=int)
    parser.add_argument("--pd-mode", choices=("fixed", "converged"))
    parser.add_argument("--pd-primal-tol", type=float)
    parser.add_argument("--pd-fixed-point-tol", type=float)
    parser.add_argument("--pd-step-safety", type=float)
    parser.add_argument("--pd-extrapolation", type=float)
    parser.add_argument("--animation", action=argparse.BooleanOptionalAction)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction)
    parser.add_argument("--video-path", type=Path)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--obstacle-center", type=float, nargs=3)
    parser.add_argument(
        "--reconstruct-state-plan", action=argparse.BooleanOptionalAction
    )
    add_planning_args(parser)
    args = parse_configured_args(parser, Path(__file__).with_suffix(".yaml"))
    if (
        args.obstacle_radius <= 0
        or args.model_error_buffer < 0
        or args.sqp_iterations < 1
    ):
        parser.error(
            "obstacle radius and SQP count must be positive; buffer must be nonnegative"
        )
    return validate_planning_args(parser, args)


def main():
    global NP_DTYPE, JAX_DTYPE
    args = parse_args()
    base.validate_cost_parameters(args.q_state, args.q_terminal)
    NP_DTYPE, JAX_DTYPE = base.configure_precision(args.precision)
    device = base.select_jax_device(args.device)
    if args.device == "cuda" and device.platform != "gpu":
        raise SystemExit("CUDA was requested, but JAX did not find a GPU device.")
    model, normalizer, normalized = base.load_checkpoint_jax(args.model_path, device)
    decoder_weight, decoder_bias = base.physical_decoder(model, normalizer, normalized)
    encode = jax.jit(
        lambda x: base.encoder_forward_jax(
            base.preprocess_state_jax(
                x, normalizer["min"], normalizer["range"], normalized
            ),
            model["encoder_params"],
            model["activation_fn"],
        )
    )
    dynamics = jax.jit(lambda x, u: base.quadrotor_dynamics_jax(x, u, args.dt))
    rng = np.random.default_rng(args.seed)
    start_np, goal_np = sample_start_goal(
        rng, args.workspace_lower, args.workspace_upper
    )
    state = jax.device_put(jnp.asarray(start_np, dtype=JAX_DTYPE), device)
    goal_y = jax.device_put(jnp.asarray(goal_np, dtype=JAX_DTYPE), device)
    mpc = ObstaclePrimalDualMPC(
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
        reconstruct_state_plan=args.reconstruct_state_plan,
        q_state=np.asarray(args.q_state),
        q_terminal=np.asarray(args.q_terminal),
        obstacle_center=args.obstacle_center,
        obstacle_radius=args.obstacle_radius,
        model_error_buffer=args.model_error_buffer,
        sqp_iterations=args.sqp_iterations,
        pd_mode=args.pd_mode,
        pd_primal_tol=args.pd_primal_tol,
        pd_fixed_point_tol=args.pd_fixed_point_tol,
        pd_step_safety=args.pd_step_safety,
        pd_extrapolation=args.pd_extrapolation,
    )
    z0 = encode(state)
    mpc.warmup(z0, goal_y, state[:3], jnp.asarray(goal_np[:3], dtype=JAX_DTYPE))
    dynamics(
        state, jnp.zeros(model["control_dim"], dtype=JAX_DTYPE)
    ).block_until_ready()
    print(f"Start: {start_np[:3]}; goal: {goal_np[:3]}")
    print(
        f"Obstacle: center={np.asarray(args.obstacle_center)}, physical radius={args.obstacle_radius:.2f} m, "
        f"planning radius={args.obstacle_radius + args.model_error_buffer:.2f} m"
    )
    print(
        f"PD: mode={args.pd_mode}, max iterations={args.pd_iterations}, "
        f"primal tol={args.pd_primal_tol:.2e}, "
        f"fixed-point tol={args.pd_fixed_point_tol:.2e}"
    )

    trajectory = [start_np.copy()]
    plans, controls, solve_times = [], [], []
    collided = False
    progress = tqdm(range(args.max_mpc_steps), desc="Obstacle MPC")
    for _ in progress:
        z = encode(state)
        if not args.warm_start:
            mpc.reset_warm_state()
        plan, control_plan, solve_time, pd_count = mpc.solve(
            z, goal_y, state[:3], goal_np[:3]
        )
        solve_times.append(solve_time)
        progress.set_postfix(solve_ms=f"{1.0e3 * solve_time:.2f}", pd=pd_count)
        if args.reconstruct_state_plan:
            plans.append(np.asarray(plan))
        controls.append(np.asarray(control_plan[0]))
        previous = np.asarray(state[:3])
        state = dynamics(state, control_plan[0])
        current = np.asarray(state)
        collided |= bool(
            segment_sphere_collision(
                previous,
                current[:3],
                np.asarray(args.obstacle_center),
                args.obstacle_radius,
            )
        )
        trajectory.append(current.copy())
        if np.linalg.norm(current[:3] - goal_np[:3]) <= args.goal_tol:
            break

    trajectory = np.asarray(trajectory)
    reached = np.linalg.norm(trajectory[-1, :3] - goal_np[:3]) <= args.goal_tol
    print(f"Reached goal: {reached}; geometric collision: {collided}")
    if solve_times:
        print(
            f"Solve time: mean={1e3 * np.mean(solve_times):.3f} ms, median={1e3 * np.median(solve_times):.3f} ms"
        )

    if not args.plot:
        return
    final_plan = plans[-1] if plans else None
    fig = viz.plot_3d(final_plan, trajectory, start_np, goal_np, plans)
    add_obstacle_to_axis(
        fig.axes[0],
        args.obstacle_radius,
        center=args.obstacle_center,
    )
    if controls:
        viz.plot_controls(np.asarray(controls))
    anim = None
    if args.animation and (args.save_video or not args.no_show):
        anim_fig, anim = viz.show_3d_animation(
            start_np,
            goal_np,
            float(np.mean(solve_times)),
            plans,
            trajectory,
            dt=args.dt,
        )
        add_obstacle_to_axis(
            anim_fig.axes[0],
            args.obstacle_radius,
            center=args.obstacle_center,
        )
    if args.save_video and anim is not None:
        viz.save_animation(anim, args.video_path, dt=args.dt)
    if not args.no_show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
