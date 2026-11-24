"""Optional video replay, separate from the timed planning loop."""

from __future__ import annotations
import argparse
from collections import deque
from pathlib import Path
from typing import Any
import imageio.v2 as imageio
import jax
import numpy as np
import torch
from tqdm.auto import tqdm
from micrompc.examples.rope.shared.perception import (
    JAXVisionEncoder,
    make_markov_state_jax,
    jax_to_torch,
)
from micrompc.examples.rope.shared.solver import make_solver
from micrompc.examples.rope.shared.simulation import (
    ArticulatedRopeConfig,
    BatchedCudaIKSolver,
    BatchedRopeSimulation,
)
from micrompc.examples.rope.shared.lab_env import LabEnv

SAVE_SUCCESS = 25
SAVE_FAILURE = 5


def measured_task_targets(simulation: BatchedRopeSimulation) -> torch.Tensor:
    """Convert the two physical gripper attachment positions to Rope task coordinates."""
    left = simulation.sim.data.site_xpos[:, simulation.arm1_site_id]
    right = simulation.sim.data.site_xpos[:, simulation.arm2_site_id]
    return torch.stack(
        (
            0.5 * (left[:, 0] + right[:, 0]),
            0.5 * (left[:, 2] + right[:, 2]),
            left[:, 1] - right[:, 1],
        ),
        dim=1,
    )


def choose_video_envs(successes: np.ndarray, seed: int) -> tuple[np.ndarray, list[str]]:
    """Select non-repeating examples after their outcomes are known."""
    rng = np.random.default_rng(seed + 1)
    success_pool = np.flatnonzero(successes)
    failure_pool = np.flatnonzero(~successes)
    success_envs = rng.choice(
        success_pool, size=min(SAVE_SUCCESS, len(success_pool)), replace=False
    )
    failure_envs = rng.choice(
        failure_pool, size=min(SAVE_FAILURE, len(failure_pool)), replace=False
    )
    selected = np.concatenate((success_envs, failure_envs)).astype(np.int64, copy=False)
    labels = ["success"] * len(success_envs) + ["failure"] * len(failure_envs)
    return selected, labels


def replay_and_save_videos(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    model: torch.nn.Module,
    vision: JAXVisionEncoder,
    planning: dict[str, object],
    device: torch.device,
    jax_device: jax.Device,
    dataset: dict[str, Any],
    ik_env: LabEnv,
    rope_config: ArticulatedRopeConfig,
    episode_indices: np.ndarray,
    qpos: np.ndarray,
    qvel: np.ndarray,
    start_targets: np.ndarray,
    start_frames: np.ndarray,
    goal_frames: np.ndarray,
    target_steps: np.ndarray,
    labels: list[str],
    action_mean: np.ndarray,
    action_std: np.ndarray,
) -> list[dict[str, Any]]:
    """Replay only selected outcomes, avoiding RGB storage for the full sweep."""
    if not len(episode_indices):
        return []
    batch_size = len(episode_indices)
    simulation = BatchedRopeSimulation(
        ik_env=ik_env,
        batch_size=batch_size,
        width=dataset["width"],
        height=dataset["height"],
        camera_name=dataset["camera"],
        disable_shadows=dataset["disable_shadows"],
        device=str(device),
        rope_config=rope_config,
    )
    ik_solver = BatchedCudaIKSolver(ik_env, batch_size=batch_size, device=str(device))
    simulation.reset(qpos, qpos)
    simulation.sim.data.qvel[:, simulation.robot_dofadr] = torch.as_tensor(
        qvel, device=device
    )
    simulation.sim.forward()
    current_embeddings = vision.encode(start_frames, args.encoder_batch_size)
    goal_embeddings = vision.encode(goal_frames, args.encoder_batch_size)
    markov_order = int(planning["markov_order"])
    history: deque[jax.Array] = deque(
        [current_embeddings for _ in range(markov_order + 1)],
        maxlen=markov_order + 1,
    )
    current_state = make_markov_state_jax(history, markov_order)
    solver = make_solver(args, model, planning, jax_device, action_mean, action_std)
    solver.warmup(current_state, goal_embeddings)
    mean = torch.as_tensor(action_mean, dtype=torch.float32, device=device)
    std = torch.as_tensor(action_std, dtype=torch.float32, device=device)
    bounds_data = dataset["task_bounds"]
    lower = torch.tensor(
        [bounds_data["reach"][0], bounds_data["height"][0], bounds_data["width"][0]],
        device=device,
    )
    upper = torch.tensor(
        [bounds_data["reach"][1], bounds_data["height"][1], bounds_data["width"][1]],
        device=device,
    )
    commanded_targets = torch.as_tensor(start_targets, device=device)
    fps = max(1, round(1.0 / (simulation.timestep * dataset["control_decimation"])))
    writers = []
    records: list[dict[str, Any]] = []
    for video_idx, (label, episode_idx, steps) in enumerate(
        zip(labels, episode_indices, target_steps, strict=True), start=1
    ):
        stem = f"{label}_{video_idx:02d}_episode_{int(episode_idx):05d}"
        video_path = out_dir / f"{stem}.mp4"
        goal_path = out_dir / f"{stem}_goal.png"
        imageio.imwrite(goal_path, np.ascontiguousarray(goal_frames[video_idx - 1]))
        writer = imageio.get_writer(video_path, fps=fps, quality=8, macro_block_size=1)
        writer.append_data(np.ascontiguousarray(start_frames[video_idx - 1]))
        writers.append(writer)
        records.append(
            {
                "outcome": label,
                "episode_idx": int(episode_idx),
                "steps": int(steps),
                "path": str(video_path),
                "goal_image_path": str(goal_path),
            }
        )
    try:
        max_steps = int(np.max(target_steps)) if len(target_steps) else 0
        progress = tqdm(
            range(max_steps),
            desc="Saving selected videos",
            unit="step",
            disable=args.disable_progress,
        )
        for step in progress:
            active = torch.as_tensor(step < target_steps, device=device)
            normalized_actions = solver.first_action(current_state, goal_embeddings)
            raw_actions = (jax_to_torch(normalized_actions) * std + mean).clamp(
                -1.0, 1.0
            )
            raw_actions[~active] = 0.0
            desired_targets = torch.clamp(commanded_targets + raw_actions, lower, upper)
            controls = ik_solver.solve(simulation.robot_qpos_tensor(), desired_targets)
            simulation.step(controls, dataset["control_decimation"])
            commanded_targets = desired_targets
            frames = simulation.render()
            current_embeddings = vision.encode(frames, args.encoder_batch_size)
            history.append(current_embeddings)
            current_state = make_markov_state_jax(history, markov_order)
            for env_idx, writer in enumerate(writers):
                if step < target_steps[env_idx]:
                    writer.append_data(np.ascontiguousarray(frames[env_idx]))
    finally:
        for writer in writers:
            writer.close()
    return records
