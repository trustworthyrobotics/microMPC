"""Bounded visual on-policy data aggregation for Rope MJLab Koopman training."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import h5py
import lightning as pl
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback
from torch.utils.data import Dataset

from micrompc.examples.rope.shared.inverse_qp import DifferentiableInverseQP


class DaggerReplayDataset(Dataset):
    """A compressed, fixed-capacity HDF5 ring buffer of visual windows."""

    def __init__(
        self,
        path: Path,
        *,
        capacity: int,
        num_frames: int,
        horizon: int,
        height: int,
        width: int,
        action_dim: int,
    ) -> None:
        if capacity < 1:
            raise ValueError("DAgger replay capacity must be positive.")
        self.path = Path(path)
        self.capacity = int(capacity)
        self.num_frames = int(num_frames)
        self.horizon = int(horizon)
        self.height = int(height)
        self.width = int(width)
        self.action_dim = int(action_dim)
        self._h5: h5py.File | None = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with h5py.File(self.path, "w") as h5:
                self._initialize_file(h5)
        else:
            with h5py.File(self.path, "r") as h5:
                expected = (self.capacity, self.num_frames, self.height, self.width, 3)
                if tuple(h5["pixels"].shape) != expected:
                    raise ValueError(
                        f"Existing DAgger replay has shape {h5['pixels'].shape}, expected {expected}."
                    )

    def _initialize_file(self, h5: h5py.File) -> None:
        h5.create_dataset(
            "pixels",
            shape=(self.capacity, self.num_frames, self.height, self.width, 3),
            dtype=np.uint8,
            chunks=(1, self.num_frames, self.height, self.width, 3),
            # Match the source dataset's fast, lossless codec. The capacity is
            # bounded by raw bytes, so compression is only an additional saving.
            compression="lzf",
        )
        h5.create_dataset(
            "action",
            shape=(self.capacity, self.horizon, self.action_dim),
            dtype=np.float32,
            chunks=(min(64, self.capacity), self.horizon, self.action_dim),
        )
        h5.attrs["count"] = 0
        h5.attrs["write_pos"] = 0
        h5.attrs["capacity"] = self.capacity

    def _file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.path, "r")
        return self._h5

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __len__(self) -> int:
        if self._h5 is not None:
            return int(self._h5.attrs["count"])
        with h5py.File(self.path, "r") as h5:
            return int(h5.attrs["count"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        h5 = self._file()
        count = int(h5.attrs["count"])
        if not 0 <= index < count:
            raise IndexError(index)
        pixels = torch.from_numpy(np.asarray(h5["pixels"][index], dtype=np.uint8))
        action = torch.from_numpy(np.asarray(h5["action"][index], dtype=np.float32))
        return {"pixels": pixels.permute(0, 3, 1, 2).contiguous(), "action": action}

    def append(self, pixels: np.ndarray, actions: np.ndarray) -> int:
        """Append windows, overwriting the oldest physical slots when full."""
        if len(pixels) != len(actions):
            raise ValueError("Replay pixels and actions must have equal batch sizes.")
        if not len(pixels):
            return 0
        if pixels.shape[1:] != (self.num_frames, self.height, self.width, 3):
            raise ValueError(f"Unexpected replay pixel shape {pixels.shape}.")
        if actions.shape[1:] != (self.horizon, self.action_dim):
            raise ValueError(f"Unexpected replay action shape {actions.shape}.")
        if len(pixels) > self.capacity:
            pixels = pixels[-self.capacity :]
            actions = actions[-self.capacity :]

        # Epoch-end callbacks run before DataLoader workers exit. Those workers
        # may still hold the current file open for reading, which makes an r+
        # open fail under HDF5 locking. Build the next generation separately and
        # atomically replace the path; existing readers safely retain the old
        # inode and newly spawned workers open the updated one.
        self.close()
        temporary_path = self.path.with_name(f".{self.path.name}.tmp.{os.getpid()}")
        temporary_path.unlink(missing_ok=True)
        try:
            with (
                h5py.File(self.path, "r") as old_h5,
                h5py.File(temporary_path, "w") as new_h5,
            ):
                self._initialize_file(new_h5)
                count = int(old_h5.attrs["count"])
                write_pos = int(old_h5.attrs["write_pos"])
                # Copy only initialized slots, in small blocks, to bound RAM.
                for start in range(0, count, 8):
                    stop = min(count, start + 8)
                    new_h5["pixels"][start:stop] = old_h5["pixels"][start:stop]
                    new_h5["action"][start:stop] = old_h5["action"][start:stop]

                first = min(len(pixels), self.capacity - write_pos)
                new_h5["pixels"][write_pos : write_pos + first] = pixels[:first]
                new_h5["action"][write_pos : write_pos + first] = actions[:first]
                remaining = len(pixels) - first
                if remaining:
                    new_h5["pixels"][:remaining] = pixels[first:]
                    new_h5["action"][:remaining] = actions[first:]
                new_h5.attrs["write_pos"] = (write_pos + len(pixels)) % self.capacity
                new_h5.attrs["count"] = min(self.capacity, count + len(pixels))
                new_h5.flush()
            os.replace(temporary_path, self.path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return len(pixels)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class DaggerMixtureDataset(Dataset):
    """Keep epoch length stable while randomly mixing original and replay data."""

    def __init__(
        self, original: Dataset, replay: DaggerReplayDataset, original_fraction: float
    ):
        if not 0.0 <= original_fraction <= 1.0:
            raise ValueError("dagger_original_fraction must be in [0, 1].")
        self.original = original
        self.replay = replay
        self.original_fraction = float(original_fraction)

    def __len__(self) -> int:
        return len(self.original)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        replay_len = len(self.replay)
        if replay_len == 0 or torch.rand(()) < self.original_fraction:
            return self.original[index]
        replay_index = int(torch.randint(replay_len, ()).item())
        return self.replay[replay_index]


def replay_capacity(
    replay_mb: int,
    *,
    num_frames: int,
    height: int,
    width: int,
    horizon: int,
    action_dim: int,
) -> int:
    """Capacity whose uncompressed payload is no larger than the requested budget."""
    if replay_mb < 1:
        raise ValueError("dagger_replay_mb must be positive.")
    bytes_per_window = num_frames * height * width * 3 + horizon * action_dim * 4
    return max(1, (replay_mb * 1024 * 1024) // bytes_per_window)


from micrompc.trainer.replay import should_collect_dagger as should_collect


def _preprocess_pixels(pixels: torch.Tensor, img_size: int) -> torch.Tensor:
    pixels = pixels.float().div_(255.0)
    if pixels.shape[-2:] != (img_size, img_size):
        pixels = F.interpolate(
            pixels, size=(img_size, img_size), mode="bilinear", align_corners=False
        )
    mean = pixels.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = pixels.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (pixels - mean) / std


def _build_markov_sequence(embeddings: torch.Tensor, order: int) -> torch.Tensor:
    states = []
    for step in range(order, embeddings.shape[1]):
        differences = embeddings[:, step - order : step + 1]
        components = [differences[:, -1]]
        for _ in range(order):
            differences = differences[:, 1:] - differences[:, :-1]
            components.append(differences[:, -1])
        states.append(torch.cat(components, dim=-1))
    return torch.stack(states, dim=1)


@torch.inference_mode()
def _encode_reference_frames(
    model: torch.nn.Module,
    frames: np.ndarray,
    *,
    device: torch.device,
    img_size: int,
    encoder_batch_size: int,
) -> torch.Tensor:
    batch_size, time_steps = frames.shape[:2]
    flat = frames.reshape(-1, *frames.shape[2:])
    encoded = []
    for start in range(0, len(flat), encoder_batch_size):
        pixels = torch.from_numpy(
            np.ascontiguousarray(flat[start : start + encoder_batch_size])
        )
        pixels = pixels.permute(0, 3, 1, 2).to(device)
        pixels = _preprocess_pixels(pixels, img_size)
        output = model.encoder(pixels, interpolate_pos_encoding=True)
        encoded.append(model.projector(output.last_hidden_state[:, 0]))
    return torch.cat(encoded).reshape(batch_size, time_steps, -1)


@torch.inference_mode()
def collect_visual_dagger(
    *,
    model: torch.nn.Module,
    source_dataset: object,
    source_indices: Sequence[int],
    count: int,
    horizon: int,
    markov_order: int,
    regularization: float,
    plan_batch_size: int,
    encoder_batch_size: int,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Plan against demonstration images, then render learner rollouts in MJWarp."""
    if count < 1:
        return np.empty((0,), dtype=np.uint8), np.empty((0,), dtype=np.float32)
    if plan_batch_size < 1 or encoder_batch_size < 1:
        raise ValueError("DAgger batch sizes must be positive.")
    if not source_indices:
        raise ValueError("No source training indices are available for DAgger.")

    from micrompc.examples.rope.shared.simulation import (
        ArticulatedRopeConfig,
        BatchedCudaIKSolver,
        BatchedRopeSimulation,
    )
    from micrompc.examples.rope.shared.lab_env import (
        BaseEnvConfig,
        LabEnv,
        TaskBounds,
        TaskState,
    )

    rng = np.random.default_rng(seed)
    selected = rng.choice(
        np.asarray(source_indices), size=count, replace=count > len(source_indices)
    )
    inverse_qp = DifferentiableInverseQP(horizon, regularization)
    mean = torch.as_tensor(action_mean.reshape(-1), device=device, dtype=torch.float32)
    std = torch.as_tensor(action_std.reshape(-1), device=device, dtype=torch.float32)
    all_pixels: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    was_training = model.training
    model.eval()

    try:
        for start in range(0, count, plan_batch_size):
            refs = source_dataset.load_dagger_references(
                selected[start : start + plan_batch_size]
            )
            frames = refs["pixels"]
            embeddings = _encode_reference_frames(
                model,
                frames,
                device=device,
                img_size=int(model.planning_config["img_size"]),
                encoder_batch_size=encoder_batch_size,
            )
            markov = _build_markov_sequence(embeddings, markov_order)
            normalized = inverse_qp.solve(model.predictor, markov[:, 0], markov[:, 1:])
            requested_actions = torch.nan_to_num(
                normalized * std + mean, nan=0.0, posinf=1.0, neginf=-1.0
            ).clamp(-1.0, 1.0)

            config = refs["sim_config"]
            bounds_dict = json.loads(config["task_bounds"])
            bounds = TaskBounds(
                reach=tuple(bounds_dict["reach"]),
                height=tuple(bounds_dict["height"]),
                width=tuple(bounds_dict["width"]),
            )
            nominal = TaskState.from_array(json.loads(config["nominal_task_state"]))
            ik_env = LabEnv(
                base_config=BaseEnvConfig(
                    task_bounds=bounds, nominal_task_state=nominal
                )
            )
            rope_config = ArticulatedRopeConfig(
                segments=config["rope_segments"],
                length=config["rope_rest_length"],
                radius=config["rope_radius"],
                link_mass=config["rope_link_mass"],
                joint_damping=config["rope_joint_damping"],
                joint_stiffness=config["rope_joint_stiffness"],
                joint_armature=config["rope_joint_armature"],
                endpoint_timeconst=config["rope_endpoint_timeconst"],
                endpoint_damping_ratio=config["rope_endpoint_damping_ratio"],
            )
            simulation = BatchedRopeSimulation(
                ik_env=ik_env,
                batch_size=len(frames),
                width=frames.shape[3],
                height=frames.shape[2],
                camera_name=config["camera"],
                disable_shadows=config["disable_shadows"],
                device=str(device),
                rope_config=rope_config,
            )
            if not np.isclose(simulation.timestep, config["physics_timestep"]):
                raise ValueError(
                    "DAgger simulator timestep does not match the source dataset: "
                    f"{simulation.timestep} != {config['physics_timestep']}."
                )
            ik_solver = BatchedCudaIKSolver(
                ik_env, batch_size=len(frames), device=str(device)
            )
            initial_qpos = torch.as_tensor(
                refs["qpos"], device=device, dtype=torch.float32
            )
            simulation.reset(initial_qpos, initial_qpos)
            simulation.sim.data.qvel[:, simulation.robot_dofadr] = torch.as_tensor(
                refs["qvel"], device=device, dtype=torch.float32
            )
            simulation.sim.forward()
            current_targets = torch.as_tensor(
                refs["task_target"], device=device, dtype=torch.float32
            )
            lower = torch.tensor(
                [bounds.reach[0], bounds.height[0], bounds.width[0]],
                device=device,
                dtype=torch.float32,
            )
            upper = torch.tensor(
                [bounds.reach[1], bounds.height[1], bounds.width[1]],
                device=device,
                dtype=torch.float32,
            )
            applied_actions = []
            rendered = []
            for step in range(horizon):
                desired_targets = torch.clamp(
                    current_targets + requested_actions[:, step], lower, upper
                )
                applied_actions.append(desired_targets - current_targets)
                controls = ik_solver.solve(
                    simulation.robot_qpos_tensor(), desired_targets
                )
                simulation.step(controls, config["control_decimation"])
                current_targets = desired_targets
                rendered.append(simulation.render())
            generated = np.stack(rendered, axis=1)
            # The dataset stores robot state but not the articulated rope's full
            # generalized state. Preserve the true reference image at t=0 and
            # use the reconstructed MJLab rope only for learner future frames.
            replay_frames = np.concatenate(
                (frames[:, : markov_order + 1], generated), axis=1
            )
            applied_normalized = (torch.stack(applied_actions, dim=1) - mean) / std
            finite = torch.isfinite(applied_normalized).all(dim=(1, 2))
            all_pixels.extend(replay_frames[finite.cpu().numpy()])
            all_actions.extend(applied_normalized[finite].cpu().numpy())
            del simulation
    finally:
        model.train(was_training)

    return np.asarray(all_pixels, dtype=np.uint8), np.asarray(
        all_actions, dtype=np.float32
    )


class VisualDaggerCallback(Callback):
    def __init__(
        self,
        *,
        args: object,
        source_dataset: object,
        source_indices: Sequence[int],
        replay: DaggerReplayDataset,
        action_mean: np.ndarray,
        action_std: np.ndarray,
    ) -> None:
        self.args = args
        self.source_dataset = source_dataset
        self.source_indices = list(source_indices)
        self.replay = replay
        self.action_mean = action_mean
        self.action_std = action_std

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        epoch = trainer.current_epoch + 1
        if not should_collect(
            epoch,
            trainer.max_epochs,
            self.args.dagger_start_epoch,
            self.args.dagger_interval,
        ):
            return
        # In a multi-process run every rank must release its read handle before
        # rank zero updates the shared replay file.
        self.replay.close()
        trainer.strategy.barrier("rope_visual_dagger_close")
        added = 0
        if trainer.is_global_zero:
            pixels, actions = collect_visual_dagger(
                model=pl_module.model,
                source_dataset=self.source_dataset,
                source_indices=self.source_indices,
                count=min(self.args.dagger_trajectories, self.replay.capacity),
                horizon=self.args.num_preds,
                markov_order=self.args.markov_order,
                regularization=self.args.dagger_qp_regularization,
                plan_batch_size=self.args.dagger_plan_batch_size,
                encoder_batch_size=self.args.dagger_encoder_batch_size,
                action_mean=self.action_mean,
                action_std=self.action_std,
                device=pl_module.device,
                seed=int(self.args.seed) + epoch,
            )
            added = self.replay.append(pixels, actions)
        trainer.strategy.barrier("rope_visual_dagger")
        pl_module.log("dagger/replay_windows", float(len(self.replay)), sync_dist=True)
        if trainer.is_global_zero:
            print(
                f"DAgger replay: added {added} windows; buffer contains {len(self.replay)}."
            )
