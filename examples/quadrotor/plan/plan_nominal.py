"""Nominal Quad3D Koopman MPC using a condensed primal-dual QP."""

import argparse
from pathlib import Path
import numpy as np
from micrompc.examples.quadrotor.shared.runtime import configure_jax_logging

configure_jax_logging()

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm
from micrompc.examples.quadrotor.shared import planning as base
from micrompc.examples.quadrotor.shared import visualization as viz

from micrompc.examples.quadrotor.shared.config import (
    add_planning_args,
    parse_configured_args,
    validate_planning_args,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reconstruct-state-plan", action=argparse.BooleanOptionalAction
    )
    add_planning_args(parser)
    parser.add_argument("--animation", action=argparse.BooleanOptionalAction)
    parser.add_argument("--save-video", action=argparse.BooleanOptionalAction)
    parser.add_argument("--video-path", type=Path)
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--factorize", action=argparse.BooleanOptionalAction)
    parser.add_argument("--min-distance", type=float)
    args = parse_configured_args(parser, Path(__file__).with_suffix(".yaml"))
    if args.min_distance < 0 or args.min_distance >= np.linalg.norm(
        np.asarray(args.workspace_upper) - args.workspace_lower
    ):
        parser.error(
            "min_distance must be nonnegative and smaller than the workspace diagonal"
        )
    return validate_planning_args(parser, args)


def main():
    args = parse_args()
    base.configure_precision(args.precision)
    base.validate_cost_parameters(args.q_state, args.q_terminal)
    device = base.select_jax_device(args.device)
    print(f"Using JAX device: {device}")
    try:
        model, normalizer, enable_normalization = base.load_checkpoint_jax(
            args.model_path, device
        )
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {args.model_path}")
        return

    encode_jit = jax.jit(
        lambda x: base.encoder_forward_jax(
            base.preprocess_state_jax(
                x, normalizer["min"], normalizer["range"], enable_normalization
            ),
            model["encoder_params"],
            model["activation_fn"],
        )
    )
    dynamics_step_jit = jax.jit(lambda x, u: base.quadrotor_dynamics_jax(x, u, args.dt))

    rng = np.random.default_rng(args.seed)
    start_state_np, goal_state_np = base.sample_start_goal(
        rng, args.workspace_lower, args.workspace_upper, args.min_distance
    )
    start_state = jax.device_put(jnp.asarray(start_state_np), device)
    goal_y = jax.device_put(jnp.asarray(goal_state_np, dtype=base.JAX_DTYPE), device)
    z0 = encode_jit(start_state)
    z0.block_until_ready()

    decoder_weight, decoder_bias = base.physical_decoder(
        model, normalizer, enable_normalization
    )
    print(f"Start position: {start_state_np[:3]}")
    print(f"Goal position:  {goal_state_np[:3]}")
    print(f"Normalization enabled: {enable_normalization}")
    print("Initializing condensed JAX PD experiment...")
    mpc = base.PrimalDualMPC(
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
        q_state=np.asarray(args.q_state),
        q_terminal=np.asarray(args.q_terminal),
        cache_dir=args.cache_dir,
        control_reference=args.control_reference,
        reconstruct_state_plan=args.reconstruct_state_plan,
        warm_start=args.warm_start,
        termination=args.pd_mode,
        primal_tol=args.pd_primal_tol,
        fixed_point_tol=args.pd_fixed_point_tol,
        step_safety=args.pd_step_safety,
        extrapolation=args.pd_extrapolation,
        factorize=args.factorize,
    )
    print(
        f"Projected-map cache: {'hit' if mpc.cache_info['maps_hit'] else 'built'} ({mpc.cache_info['maps_path']})"
    )
    system_label = (
        "QP-Cholesky" if mpc.cache_info["system_kind"] == "cholesky" else "QP-inverse"
    )
    print(
        f"{system_label} cache: "
        f"{'hit' if mpc.cache_info['system_hit'] else 'built'} "
        f"({mpc.cache_info['system_path']})"
    )

    mpc.warmup(z0, goal_y)
    dynamics_step_jit(
        start_state, jnp.zeros(model["control_dim"], dtype=base.JAX_DTYPE)
    ).block_until_ready()
    print("Solver ready. Running MPC loop...")

    current_state = start_state
    true_traj = [start_state_np.copy()]
    applied_controls, planned_trajs, solve_times = [], [], []
    progress = tqdm(range(args.max_mpc_steps), desc="MPC Steps")
    for _ in progress:
        z_current = encode_jit(current_state)
        z_current.block_until_ready()
        x_plan, u_plan, solve_time = mpc.solve(z_current, goal_y)
        solve_times.append(solve_time)
        progress.set_postfix({"solve_ms": f"{solve_time * 1000.0:.4f}"})
        u_plan_np = np.asarray(u_plan[0])
        if args.reconstruct_state_plan:
            planned_trajs.append(np.asarray(x_plan))

        applied_controls.append(u_plan_np)
        current_state = dynamics_step_jit(current_state, u_plan[0])
        current_state_np = np.asarray(current_state)
        true_traj.append(current_state_np.copy())
        if np.linalg.norm(current_state_np[:3] - goal_state_np[:3]) <= args.goal_tol:
            break

    true_traj = np.asarray(true_traj, dtype=base.NP_DTYPE)
    applied_controls = np.asarray(applied_controls, dtype=base.NP_DTYPE)
    print(f"Final position: {true_traj[-1, :3]}")
    print(f"Goal position:  {goal_state_np[:3]}")
    avg_solve = float(np.mean(solve_times)) if solve_times else 0.0
    if solve_times:
        print(f"Average solve time: {avg_solve * 1000.0:.4f} ms")
        print(f"Median solve time:  {np.median(solve_times) * 1000.0:.4f} ms")
        print(f"Min solve time:     {np.min(solve_times) * 1000.0:.4f} ms")
        print(f"Max solve time:     {np.max(solve_times) * 1000.0:.4f} ms")

    if not args.plot:
        return
    last_plan = planned_trajs[-1] if planned_trajs else None
    anim = None
    if args.animation:
        _, anim = viz.show_3d_animation(
            start_state_np,
            goal_state_np,
            avg_solve,
            planned_trajs,
            true_traj,
            dt=args.dt,
        )
    if args.animation and args.save_video and anim is not None:
        viz.save_animation(anim, args.video_path, dt=args.dt)
    viz.plot_3d(last_plan, true_traj, start_state_np, goal_state_np, planned_trajs)
    if len(applied_controls):
        viz.plot_controls(applied_controls)
    if not args.no_show:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
