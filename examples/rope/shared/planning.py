#!/usr/bin/env python3
"""Evaluate Rope Koopman MPC in parallel MJLab/MuJoCo-Warp environments."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Any

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import h5py
from .runtime import configure_jax_logging

configure_jax_logging()

import jax
import numpy as np
import torch
from tqdm.auto import tqdm

from micrompc.examples.rope.shared.simulation import (
    ArticulatedRopeConfig,
    BatchedCudaIKSolver,
    BatchedRopeSimulation,
)
from micrompc.examples.rope.shared.checkpoints import (
    latest_checkpoint,
    load_self_contained_model,
    require_device,
    to_jsonable,
)
from micrompc.examples.rope.shared.perception import (
    JAXVisionEncoder,
    make_markov_state_jax,
    jax_to_torch,
    select_jax_device,
)
from micrompc.examples.rope.shared.visualization import (
    measured_task_targets,
    choose_video_envs,
    replay_and_save_videos,
)
from micrompc.examples.rope.shared.solver import make_solver
from micrompc.examples.rope.shared.lab_env import (
    BaseEnvConfig,
    LabEnv,
    TaskBounds,
    TaskState,
)


DEFAULT_MODEL_DIR = "rope/models/KO_rope"
DEFAULT_DATASET_PATH = "rope/data/test/rope_random_cubic_spline.h5"
DEFAULT_OUT_DIR = "rope/plan/results/plan_nominal"

NUM_ENVS = 1
MAX_MPC_STEPS = 100
ENCODER_BATCH_SIZE = 256
GOAL_THRESHOLD = 0.05
SAVE_SUCCESS = 25
SAVE_FAILURE = 5
HORIZON = 5
Q_TERMINAL = 10.0
Q_STAGE = 0.05
R_CONTROL = 0.1
QP_REGULARIZATION = 1e-3
Q_VELOCITY = 1.0
Q_ACCELERATION = 1.0
PD_RHO = 0.1
PD_STEPS = 150
# ``fixed`` always runs PD_STEPS. ``converged`` stops when every batch
# member meets the standard absolute/relative PD residual tolerances.
PD_MODE = "converged"
PD_ABS_TOL = 1e-2
PD_REL_TOL = 1e-2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(DEFAULT_MODEL_DIR))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, default=Path(DEFAULT_DATASET_PATH))
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS)
    parser.add_argument("--max-mpc-steps", type=int, default=MAX_MPC_STEPS)
    parser.add_argument("--encoder-batch-size", type=int, default=ENCODER_BATCH_SIZE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--goal-threshold", type=float, default=GOAL_THRESHOLD)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--q-terminal", type=float, default=Q_TERMINAL)
    parser.add_argument("--q-stage", type=float, default=Q_STAGE)
    parser.add_argument("--r-control", type=float, default=R_CONTROL)
    parser.add_argument("--q-velocity", type=float, default=Q_VELOCITY)
    parser.add_argument("--q-acceleration", type=float, default=Q_ACCELERATION)
    parser.add_argument("--qp-regularization", type=float, default=QP_REGULARIZATION)
    parser.add_argument("--pd-rho", type=float, default=PD_RHO)
    parser.add_argument("--pd-steps", type=int, default=PD_STEPS)
    parser.add_argument("--pd-mode", choices=("fixed", "converged"), default=PD_MODE)
    parser.add_argument("--pd-abs-tol", type=float, default=PD_ABS_TOL)
    parser.add_argument("--pd-rel-tol", type=float, default=PD_REL_TOL)
    parser.add_argument("--disable-progress", action="store_true")
    parser.add_argument("--disable-videos", action="store_true")
    from micrompc.examples.rope.shared.config import parse_config

    parser.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    args = parse_config(parser, "plan/plan_nominal.yaml")
    args.disable_videos = args.disable_videos or not args.plot
    positive = (
        args.num_envs,
        args.max_mpc_steps,
        args.encoder_batch_size,
        args.horizon,
    )
    if min(positive) < 1:
        parser.error(
            "environment count, step counts, batch sizes, and horizon must be positive."
        )
    if (
        min(
            args.q_terminal,
            args.q_stage,
            args.r_control,
            args.q_velocity,
            args.q_acceleration,
        )
        < 0
    ):
        parser.error("QP cost weights cannot be negative.")
    if args.qp_regularization <= 0 or args.pd_rho <= 0 or args.pd_steps < 1:
        parser.error("QP regularization, PD rho, and PD steps must be positive.")
    if args.pd_abs_tol < 0 or args.pd_rel_tol < 0:
        parser.error("PD tolerances cannot be negative.")
    if args.goal_threshold <= 0:
        parser.error("--goal-threshold must be positive.")
    return args


def decode_attr(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def distribution(values: np.ndarray | list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not finite.size:
        return {
            "median": None,
            "std": None,
            "mean": None,
            "min": None,
            "max": None,
            "count": 0,
        }
    return {
        "median": float(np.median(finite)),
        "std": float(np.std(finite)),
        "mean": float(np.mean(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "count": int(finite.size),
    }


def read_evaluation_batch(
    dataset_path: Path, *, num_envs: int, seed: int
) -> tuple[
    dict[str, Any],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Sample complete episodes and load only their start/goal records."""
    with h5py.File(dataset_path, "r") as h5:
        if decode_attr(h5.attrs.get("simulation_backend", "")) != "mjlab_mujoco_warp":
            raise ValueError("Evaluation requires an MJLab/MuJoCo-Warp Rope dataset.")
        ep_len = np.asarray(h5["ep_len"][:], dtype=np.int64)
        valid = np.flatnonzero(ep_len >= 2)
        if not valid.size:
            raise ValueError("Need at least one episode containing two frames.")
        rng = np.random.default_rng(seed)
        episode_indices = rng.choice(
            valid, size=num_envs, replace=num_envs > len(valid)
        )
        qpos = np.empty((num_envs, h5["qpos"].shape[1]), dtype=np.float32)
        qvel = np.empty((num_envs, h5["qvel"].shape[1]), dtype=np.float32)
        start_targets = np.empty((num_envs, 3), dtype=np.float32)
        goal_targets = np.empty_like(start_targets)
        start_frames = np.empty((num_envs, *h5["pixels"].shape[1:]), dtype=np.uint8)
        goal_frames = np.empty_like(start_frames)
        for env_idx, episode_idx in enumerate(episode_indices):
            offset = int(h5["ep_offset"][episode_idx])
            goal_row = offset + int(h5["ep_len"][episode_idx]) - 1
            qpos[env_idx] = h5["qpos"][offset]
            qvel[env_idx] = h5["qvel"][offset]
            start_targets[env_idx] = h5["task_target"][offset]
            goal_targets[env_idx] = h5["task_target"][goal_row]
            start_frames[env_idx] = h5["pixels"][offset]
            goal_frames[env_idx] = h5["pixels"][goal_row]
        bounds = json.loads(decode_attr(h5.attrs["task_bounds"]))
        config = {
            "camera": decode_attr(h5.attrs.get("camera", "video_cam")),
            "disable_shadows": bool(h5.attrs.get("disable_shadows", False)),
            "physics_timestep": float(h5.attrs.get("physics_timestep", 0.002)),
            "control_decimation": int(h5.attrs.get("control_decimation", 25)),
            "goal_tolerance": float(h5.attrs.get("goal_tolerance", 1e-4)),
            "height": int(h5["pixels"].shape[1]),
            "width": int(h5["pixels"].shape[2]),
            "task_bounds": bounds,
            "nominal_task_state": json.loads(
                decode_attr(h5.attrs["nominal_task_state"])
            ),
            "rope_segments": int(h5.attrs.get("rope_segments", 22)),
            "rope_rest_length": float(h5.attrs["rope_rest_length"]),
            "rope_radius": float(h5.attrs.get("rope_radius", 0.008)),
            "rope_link_mass": float(h5.attrs.get("rope_link_mass", 0.0015)),
            "rope_joint_damping": float(h5.attrs.get("rope_joint_damping", 0.02)),
            "rope_joint_stiffness": float(h5.attrs.get("rope_joint_stiffness", 0.005)),
            "rope_joint_armature": float(h5.attrs.get("rope_joint_armature", 0.001)),
            "rope_endpoint_timeconst": float(
                h5.attrs.get("rope_endpoint_timeconst", 0.02)
            ),
            "rope_endpoint_damping_ratio": float(
                h5.attrs.get("rope_endpoint_damping_ratio", 0.09)
            ),
        }
    return (
        config,
        episode_indices,
        qpos,
        qvel,
        start_targets,
        goal_targets,
        start_frames,
        goal_frames,
    )


def main() -> None:
    args = parse_args()
    device = require_device(args.device)
    jax_device = select_jax_device(device)
    model_dir = args.model_dir.expanduser().resolve()
    dataset_path = args.dataset_path.expanduser().resolve()
    checkpoint = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else latest_checkpoint(model_dir).resolve()
    )
    # Torch is used only to deserialize the checkpoint and run MJLab/IK. All
    # learned-model inference is transferred to JAX below.
    model, planning, action_mean, action_std = load_self_contained_model(
        checkpoint, torch.device("cpu")
    )
    vision = JAXVisionEncoder(model, jax_device, int(planning["img_size"]))
    if int(planning["frameskip"]) != 1 or int(planning["action_dim"]) != 3:
        raise ValueError("Batched Rope MPC requires frameskip=1 and action_dim=3.")

    (
        dataset,
        episode_indices,
        qpos,
        qvel,
        start_targets_np,
        goal_targets_np,
        start_frames,
        goal_frames,
    ) = read_evaluation_batch(dataset_path, num_envs=args.num_envs, seed=args.seed)
    bounds_data = dataset["task_bounds"]
    bounds = TaskBounds(
        reach=tuple(bounds_data["reach"]),
        height=tuple(bounds_data["height"]),
        width=tuple(bounds_data["width"]),
    )
    ik_env = LabEnv(
        base_config=BaseEnvConfig(
            task_bounds=bounds,
            nominal_task_state=TaskState.from_array(dataset["nominal_task_state"]),
        )
    )
    rope_config = ArticulatedRopeConfig(
        segments=dataset["rope_segments"],
        length=dataset["rope_rest_length"],
        radius=dataset["rope_radius"],
        link_mass=dataset["rope_link_mass"],
        joint_damping=dataset["rope_joint_damping"],
        joint_stiffness=dataset["rope_joint_stiffness"],
        joint_armature=dataset["rope_joint_armature"],
        endpoint_timeconst=dataset["rope_endpoint_timeconst"],
        endpoint_damping_ratio=dataset["rope_endpoint_damping_ratio"],
    )
    simulation = BatchedRopeSimulation(
        ik_env=ik_env,
        batch_size=args.num_envs,
        width=dataset["width"],
        height=dataset["height"],
        camera_name=dataset["camera"],
        disable_shadows=dataset["disable_shadows"],
        device=str(device),
        rope_config=rope_config,
    )
    if not np.isclose(simulation.timestep, dataset["physics_timestep"]):
        raise ValueError("Simulator and dataset physics timesteps do not match.")
    ik_solver = BatchedCudaIKSolver(
        ik_env, batch_size=args.num_envs, device=str(device)
    )
    simulation.reset(qpos, qpos)
    simulation.sim.data.qvel[:, simulation.robot_dofadr] = torch.as_tensor(
        qvel, device=device
    )
    simulation.sim.forward()

    start_embeddings = vision.encode(start_frames, args.encoder_batch_size)
    goal_embeddings = vision.encode(goal_frames, args.encoder_batch_size)
    markov_order = int(planning["markov_order"])
    history: deque[jax.Array] = deque(
        [start_embeddings for _ in range(markov_order + 1)],
        maxlen=markov_order + 1,
    )
    current_state = make_markov_state_jax(history, markov_order)
    solver = make_solver(args, model, planning, jax_device, action_mean, action_std)
    solver.warmup(current_state, goal_embeddings)
    mean = torch.as_tensor(action_mean, dtype=torch.float32, device=device)
    std = torch.as_tensor(action_std, dtype=torch.float32, device=device)
    lower = torch.tensor(
        [bounds.reach[0], bounds.height[0], bounds.width[0]], device=device
    )
    upper = torch.tensor(
        [bounds.reach[1], bounds.height[1], bounds.width[1]], device=device
    )
    commanded_targets = torch.as_tensor(start_targets_np, device=device)
    goal_targets = torch.as_tensor(goal_targets_np, device=device)
    tolerance = args.goal_threshold
    # Match the serial Rope evaluator: task-goal success is defined in the
    # commanded three-dimensional task space. Physical attachment error is
    # retained as a separate diagnostic below.
    distances = torch.linalg.vector_norm(commanded_targets - goal_targets, dim=1)
    successes = distances <= tolerance
    best_distances = distances.clone()
    reached_distances = torch.where(
        successes, distances, torch.full_like(distances, torch.nan)
    )
    executed_steps = torch.zeros(args.num_envs, dtype=torch.int32, device=device)
    solve_times_ms: list[float] = []
    perception_times_ms: list[float] = []
    mpc_total_times_ms: list[float] = []
    pd_iterations: list[int] = []
    completed_steps = 0
    run_started = time.perf_counter()

    progress = tqdm(
        total=args.max_mpc_steps,
        desc="Batched Rope Koopman MPC",
        unit="step",
        disable=args.disable_progress,
    )
    for step in range(args.max_mpc_steps):
        if bool(successes.all()):
            break
        normalized_actions = solver.first_action(current_state, goal_embeddings)
        solve_times_ms.append(solver.last_solve_ms)
        pd_iterations.append(int(solver.last_pd_diagnostics["iterations"]))
        raw_actions = (jax_to_torch(normalized_actions) * std + mean).clamp(-1.0, 1.0)
        raw_actions[successes] = 0.0
        desired_targets = torch.clamp(commanded_targets + raw_actions, lower, upper)
        controls = ik_solver.solve(simulation.robot_qpos_tensor(), desired_targets)
        simulation.step(controls, dataset["control_decimation"])
        commanded_targets = desired_targets

        frames = simulation.render()
        embeddings, vit_ms = vision.encode(frames, args.encoder_batch_size, timed=True)
        perception_times_ms.append(vit_ms)
        mpc_total_times_ms.append(solve_times_ms[-1] + perception_times_ms[-1])
        history.append(embeddings)
        current_state = make_markov_state_jax(history, markov_order)
        distances = torch.linalg.vector_norm(commanded_targets - goal_targets, dim=1)
        best_distances = torch.minimum(best_distances, distances)
        newly_successful = (~successes) & (distances <= tolerance)
        executed_steps[newly_successful] = step + 1
        reached_distances[newly_successful] = distances[newly_successful]
        successes |= newly_successful
        completed_steps = step + 1
        progress.update()
        progress.set_postfix(
            # success=f"{int(successes.sum())}/{args.num_envs}",
            success=f"{100.0 * float(successes.float().mean()):.1f}%",
            task_goal=f"{float(torch.median(distances)):.4f}",
            mpc_total_ms=f"{mpc_total_times_ms[-1]:.1f}",
            perception_ms=f"{perception_times_ms[-1]:.1f}",
            solve_ms=f"{solve_times_ms[-1]:.1f}",
            pd=pd_iterations[-1],
        )
    progress.close()

    elapsed = time.perf_counter() - run_started
    executed_steps[~successes] = completed_steps
    result_distances = torch.where(successes, reached_distances, distances)
    successes_np = successes.cpu().numpy()
    executed_steps_np = executed_steps.cpu().numpy()
    final_distances_np = result_distances.cpu().numpy()
    best_distances_np = best_distances.cpu().numpy()
    physical_distances_np = (
        torch.linalg.vector_norm(
            measured_task_targets(simulation) - goal_targets, dim=1
        )
        .cpu()
        .numpy()
    )
    solve_times_np = np.asarray(solve_times_ms, dtype=np.float64)
    perception_times_np = np.asarray(perception_times_ms, dtype=np.float64)
    mpc_total_times_np = np.asarray(mpc_total_times_ms, dtype=np.float64)
    success_count = int(successes_np.sum())
    task_stats = distribution(final_distances_np)
    solve_stats = distribution(solve_times_np)
    perception_stats = distribution(perception_times_np)
    mpc_total_stats = distribution(mpc_total_times_np)
    amortized_stats = distribution(solve_times_np / args.num_envs)
    amortized_perception_stats = distribution(perception_times_np / args.num_envs)
    amortized_mpc_total_stats = distribution(mpc_total_times_np / args.num_envs)

    timestamp = int(time.time())
    out_dir = args.out_dir.expanduser().resolve() / str(timestamp)
    while out_dir.exists():
        timestamp += 1
        out_dir = args.out_dir.expanduser().resolve() / str(timestamp)
    out_dir.mkdir(parents=True)
    summary = {
        "success_count": success_count,
        "failure_count": args.num_envs - success_count,
        "num_envs": args.num_envs,
        "success_rate": success_count / args.num_envs,
        "success_metric": "minimum Euclidean commanded-task distance <= goal_threshold",
        "task_coordinates": ["reach", "height", "width"],
        "goal_threshold_m": tolerance,
        "task_goal_distance_m": task_stats,
        "best_task_goal_distance_m": distribution(best_distances_np),
        "physical_attachment_goal_distance_m": distribution(physical_distances_np),
        "batch_mpc_total_ms": mpc_total_stats,
        "batch_vit_forward_ms": perception_stats,
        "batch_qp_solve_ms": solve_stats,
        "batch_perception_ms": perception_stats,
        "batch_solve_ms": solve_stats,
        "amortized_per_env_mpc_total_ms": amortized_mpc_total_stats,
        "amortized_per_env_perception_ms": amortized_perception_stats,
        "amortized_per_env_solve_ms": amortized_stats,
        "timing_scopes": {
            "perception": "synchronized ViT encoder plus projector forward pass only; rendering, transfer, resize, and normalization excluded",
            "solve": "synchronized compiled JAX QP only; pending input work and PD diagnostic host conversions excluded",
            "mpc_total": "sum of the ViT-forward and QP-solve measurements",
        },
        "elapsed_seconds": elapsed,
        "completed_mpc_steps": completed_steps,
        "episode_indices": episode_indices.tolist(),
        "dataset_path": str(dataset_path),
        "checkpoint_path": str(checkpoint),
        "device": str(device),
        "jax_device": str(jax_device),
        "precision": {
            "perception": "bfloat16",
            "solver": "float32",
            "cache_build": "float64",
        },
        "seed": args.seed,
        "video_config": {
            "enabled": not args.disable_videos,
            "save_success": SAVE_SUCCESS,
            "save_failure": SAVE_FAILURE,
            "selected_env_indices": [],
        },
        "saved_videos": [],
        "pd": {
            "mode": args.pd_mode,
            "rho": args.pd_rho,
            "max_steps": args.pd_steps,
            "absolute_tolerance": args.pd_abs_tol,
            "relative_tolerance": args.pd_rel_tol,
            "iteration_counts": distribution(pd_iterations),
            "last_diagnostics": solver.last_pd_diagnostics,
        },
        "qp": {
            "horizon": args.horizon,
            "q_terminal": args.q_terminal,
            "q_stage": args.q_stage,
            "r_control": args.r_control,
            "q_velocity": args.q_velocity,
            "q_acceleration": args.q_acceleration,
            "regularization": args.qp_regularization,
            "cache": solver.cache_info,
        },
    }
    np.savez_compressed(
        out_dir / "results.npz",
        episode_indices=episode_indices,
        successes=successes_np,
        executed_steps=executed_steps_np,
        final_task_goal_distances_m=final_distances_np,
        best_task_goal_distances_m=best_distances_np,
        physical_attachment_goal_distances_m=physical_distances_np,
        mpc_total_times_ms=mpc_total_times_np,
        perception_times_ms=perception_times_np,
        solve_times_ms=solve_times_np,
        pd_iterations=np.asarray(pd_iterations, dtype=np.int32),
    )

    selected_envs, video_labels = (
        (np.empty(0, dtype=np.int64), [])
        if args.disable_videos
        else choose_video_envs(successes_np, args.seed)
    )
    summary["video_config"]["selected_env_indices"] = selected_envs.tolist()
    video_records = []
    if not args.disable_videos:
        del simulation, ik_solver, history, current_state, goal_embeddings, solver
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        video_records = replay_and_save_videos(
            args=args,
            out_dir=out_dir,
            model=model,
            vision=vision,
            planning=planning,
            device=device,
            jax_device=jax_device,
            dataset=dataset,
            ik_env=ik_env,
            rope_config=rope_config,
            episode_indices=episode_indices[selected_envs],
            qpos=qpos[selected_envs],
            qvel=qvel[selected_envs],
            start_targets=start_targets_np[selected_envs],
            start_frames=start_frames[selected_envs],
            goal_frames=goal_frames[selected_envs],
            target_steps=executed_steps_np[selected_envs],
            labels=video_labels,
            action_mean=action_mean,
            action_std=action_std,
        )
    summary["saved_videos"] = video_records
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(summary), handle, indent=2)

    print(
        f"\nSuccess rate: {success_count}/{args.num_envs} ({100.0 * success_count / args.num_envs:.1f}%)"
    )
    print(
        f"Task goal:   median={task_stats['median']:.6f} m, 1 std={task_stats['std']:.6f} m"
    )
    if mpc_total_times_np.size:
        print(
            f"ViT + QP:    median={mpc_total_stats['median']:.3f} ms, "
            f"1 std={mpc_total_stats['std']:.3f} ms"
        )
        print(
            f"ViT forward: median={perception_stats['median']:.3f} ms, "
            f"1 std={perception_stats['std']:.3f} ms"
        )
        print(
            f"QP solve:    median={solve_stats['median']:.3f} ms, 1 std={solve_stats['std']:.3f} ms"
        )
    print(f"PD:        mode={args.pd_mode}, max steps={args.pd_steps}")
    print(f"Videos:      {len(video_records)}")
    print(f"Summary:     {out_dir / 'summary.json'}")
    print(f"Results:     {out_dir / 'results.npz'}")


if __name__ == "__main__":
    main()
