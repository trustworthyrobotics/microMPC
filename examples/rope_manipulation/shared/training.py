#!/usr/bin/env python3
"""Train a visual Koopman model with differentiable inverse-QP and DAgger."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from functools import partial
from pathlib import Path, PosixPath

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import lightning as pl
import numpy as np
import stable_pretraining as spt
import torch
import torch.nn.functional as F
from micrompc.trainer.training import KoopmanTrainer, LossWeights
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset, random_split

from micrompc.examples.rope_manipulation.shared.models import SketchGaussianRegularizer


DEFAULT_DATASET_PATH = "rope_manipulation/data/train/rope_random_cubic_spline.h5"
DEFAULT_RUN_DIR = "rope_manipulation/runs/koopman"
FIXED_FRAMESKIP = 1

VIT_PRESETS = {
    "tiny": (192, 12, 3, 768),
    "small": (384, 12, 6, 1536),
    "base": (768, 12, 12, 3072),
    "large": (1024, 24, 16, 4096),
    "huge": (1280, 32, 16, 5120),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--finetune-dir", type=Path, default=None)
    parser.add_argument("--output-model-name", default="koopdyn")
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--train-split", type=float, default=1.0)

    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--encoder-scale", choices=tuple(VIT_PRESETS), default="tiny")
    parser.add_argument("--vit-hidden-size", type=int, default=None)
    parser.add_argument("--vit-depth", type=int, default=9)
    parser.add_argument("--vit-num-heads", type=int, default=None)
    parser.add_argument("--vit-mlp-dim", type=int, default=384)
    parser.add_argument("--embed-dim", type=int, default=12)
    parser.add_argument(
        "--markov-order",
        type=int,
        default=2,
        help="Highest finite-difference order included after the position embedding.",
    )
    parser.add_argument(
        "--num-preds", type=int, default=10, help="World-model training horizon."
    )
    parser.add_argument("--action-dim", type=int, default=3)

    parser.add_argument("--lifting-dim", type=int, default=2048)
    parser.add_argument("--lifting-hidden-width", type=int, default=256)
    parser.add_argument("--lifting-depth", type=int, default=2)
    parser.add_argument("--lifting-dropout", type=float, default=0.0)

    parser.add_argument("--prediction-weight", type=float, default=1.0)
    parser.add_argument("--latent-weight", type=float, default=0.1)
    parser.add_argument("--sigreg-weight", type=float, default=0.005)
    parser.add_argument("--sigreg-knots", type=int, default=17)
    parser.add_argument("--sigreg-num-proj", type=int, default=1024)
    parser.add_argument("--control-weight", type=float, default=1e-2)
    parser.add_argument("--qp-regularization", type=float, default=1e-4)

    parser.add_argument("--dagger", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dagger-start-epoch", type=int, default=2)
    parser.add_argument("--dagger-interval", type=int, default=1)
    parser.add_argument("--dagger-trajectories", type=int, default=64)
    parser.add_argument(
        "--dagger-replay-mb",
        type=int,
        default=512,
        help="Uncompressed upper bound for the compressed visual replay ring buffer.",
    )
    parser.add_argument("--dagger-original-fraction", type=float, default=0.5)
    parser.add_argument("--dagger-plan-batch-size", type=int, default=32)
    parser.add_argument("--dagger-encoder-batch-size", type=int, default=64)
    parser.add_argument("--dagger-qp-regularization", type=float, default=1e-4)

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--persistent-workers", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--precision", default="bf16-mixed")
    from micrompc.examples.rope_manipulation.shared.config import parse_config

    args = parse_config(parser, "train/train.yaml")

    if args.markov_order < 0:
        parser.error("--markov-order must be non-negative.")
    if args.num_preds < 1:
        parser.error("--num-preds must be positive.")
    if not 0 < args.train_split <= 1:
        parser.error("--train-split must be in (0, 1].")
    if args.control_weight < 0.0:
        parser.error("--control-weight must be non-negative.")
    if args.qp_regularization <= 0.0 or args.dagger_qp_regularization <= 0.0:
        parser.error("QP regularization values must be positive.")
    if args.dagger_start_epoch < 1 or args.dagger_interval < 1:
        parser.error("DAgger start and interval must be positive.")
    if args.dagger_trajectories < 1:
        parser.error("--dagger-trajectories must be positive.")
    if args.dagger_replay_mb < 1:
        parser.error("--dagger-replay-mb must be positive.")
    if not 0.0 <= args.dagger_original_fraction <= 1.0:
        parser.error("--dagger-original-fraction must be in [0, 1].")
    if args.dagger_plan_batch_size < 1 or args.dagger_encoder_batch_size < 1:
        parser.error("DAgger batch sizes must be positive.")

    preset = VIT_PRESETS[args.encoder_scale]
    args.vit_hidden_size = (
        preset[0] if args.vit_hidden_size is None else args.vit_hidden_size
    )
    args.vit_depth = preset[1] if args.vit_depth is None else args.vit_depth
    args.vit_num_heads = preset[2] if args.vit_num_heads is None else args.vit_num_heads
    args.vit_mlp_dim = preset[3] if args.vit_mlp_dim is None else args.vit_mlp_dim
    if args.vit_hidden_size % args.vit_num_heads:
        parser.error("--vit-hidden-size must be divisible by --vit-num-heads.")

    args.frameskip = FIXED_FRAMESKIP
    args.markov_state_dim = (args.markov_order + 1) * args.embed_dim
    args.effective_action_dim = args.frameskip * args.action_dim
    return args


def build_markov_state(history_emb: torch.Tensor, markov_order: int) -> torch.Tensor:
    """Return [position, first difference, ..., requested finite difference]."""
    context_len = markov_order + 1
    if history_emb.ndim != 3 or history_emb.shape[1] < context_len:
        raise ValueError(
            f"Expected [batch, >= {context_len}, embed_dim], got {tuple(history_emb.shape)}."
        )
    differences = history_emb[:, -context_len:]
    components = [differences[:, -1]]
    for _ in range(markov_order):
        differences = differences[:, 1:] - differences[:, :-1]
        components.append(differences[:, -1])
    return torch.cat(components, dim=-1)


def build_markov_sequence(emb: torch.Tensor, markov_order: int) -> torch.Tensor:
    return torch.stack(
        [
            build_markov_state(emb[:, : step + 1], markov_order)
            for step in range(markov_order, emb.shape[1])
        ],
        dim=1,
    )


def preprocess_pixels(pixels: torch.Tensor, img_size: int) -> torch.Tensor:
    pixels = pixels.float().div_(255.0)
    if pixels.shape[-2:] != (img_size, img_size):
        batch_size, time_steps = pixels.shape[:2]
        pixels = F.interpolate(
            pixels.view(batch_size * time_steps, *pixels.shape[2:]),
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        ).view(batch_size, time_steps, pixels.shape[2], img_size, img_size)
    mean = pixels.new_tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
    std = pixels.new_tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
    return (pixels - mean) / std


from .model_factory import build_model


class KoopmanRopeDataset(Dataset):
    """Pixel/action windows with first-frame padding for missing Markov history."""

    def __init__(
        self,
        dataset_path: Path,
        *,
        markov_order: int,
        num_preds: int,
        frameskip: int,
        action_dim: int,
    ) -> None:
        self.dataset_path = dataset_path
        self.markov_order = int(markov_order)
        self.num_preds = int(num_preds)
        self.frameskip = int(frameskip)
        self.action_dim = int(action_dim)
        self.num_frames = self.markov_order + self.num_preds + 1
        self._h5: h5py.File | None = None

        with h5py.File(self.dataset_path, "r") as h5:
            required = {"pixels", "action", "ep_len", "ep_offset"}
            missing = required - set(h5)
            if missing:
                raise ValueError(f"Dataset is missing {sorted(missing)}.")
            self.ep_len = np.asarray(h5["ep_len"][:], dtype=np.int64)
            self.ep_offset = np.asarray(h5["ep_offset"][:], dtype=np.int64)
            self.height = int(h5["pixels"].shape[1])
            self.width = int(h5["pixels"].shape[2])
            if int(h5["action"].shape[-1]) != self.action_dim:
                raise ValueError(
                    f"Expected action_dim={self.action_dim}, got {h5['action'].shape[-1]}."
                )
            actions = np.asarray(h5["action"][:], dtype=np.float32)
            actions = actions[~np.isnan(actions).any(axis=1)]
            if not actions.size:
                raise ValueError("Dataset contains no finite applied actions.")
            self.action_mean = actions.mean(axis=0, keepdims=True).astype(np.float32)
            self.action_std = np.maximum(
                actions.std(axis=0, keepdims=True), 1e-6
            ).astype(np.float32)

        self.samples: list[tuple[int, int]] = []
        required_future = self.num_preds * self.frameskip
        for ep_idx, ep_len in enumerate(self.ep_len.tolist()):
            max_current = ep_len - 1 - required_future
            self.samples.extend((ep_idx, current) for current in range(max_current + 1))
        if not self.samples:
            raise ValueError(
                "No valid windows found. Check markov-order, num-preds, and episode lengths."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.dataset_path, "r")
        return self._h5

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        h5 = self._file()
        ep_idx, current = self.samples[index]
        ep_base = int(self.ep_offset[ep_idx])
        relative_frames = np.arange(
            -self.markov_order, self.num_preds + 1, dtype=np.int64
        )
        frame_rows = ep_base + np.maximum(current + relative_frames * self.frameskip, 0)
        unique_rows, inverse = np.unique(frame_rows, return_inverse=True)
        pixels_np = np.asarray(h5["pixels"][unique_rows], dtype=np.uint8)[inverse]
        pixels = torch.from_numpy(pixels_np).permute(0, 3, 1, 2).contiguous()
        action_start = ep_base + current
        action_offsets = np.arange(self.num_preds * self.frameskip, dtype=np.int64)
        actions = np.asarray(
            h5["action"][action_start + action_offsets], dtype=np.float32
        )
        actions = (np.nan_to_num(actions, nan=0.0) - self.action_mean) / self.action_std
        return {
            "pixels": pixels,
            "action": torch.from_numpy(actions.reshape(self.num_preds, -1)).float(),
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def load_dagger_references(self, indices: Sequence[int]) -> dict[str, object]:
        """Load reference pixels and MJLab simulator starts for selected windows."""
        pixels = []
        qpos = []
        qvel = []
        task_targets = []
        with h5py.File(self.dataset_path, "r") as h5:
            required = {"qpos", "qvel", "task_target"}
            missing = required - set(h5)
            if missing:
                raise ValueError(f"DAgger requires dataset fields {sorted(missing)}.")
            for index in indices:
                ep_idx, current = self.samples[int(index)]
                ep_base = int(self.ep_offset[ep_idx])
                relative = np.arange(
                    -self.markov_order, self.num_preds + 1, dtype=np.int64
                )
                rows = ep_base + np.maximum(current + relative * self.frameskip, 0)
                unique_rows, inverse = np.unique(rows, return_inverse=True)
                pixels.append(
                    np.asarray(h5["pixels"][unique_rows], dtype=np.uint8)[inverse]
                )
                current_row = ep_base + current
                qpos.append(np.asarray(h5["qpos"][current_row], dtype=np.float32))
                qvel.append(np.asarray(h5["qvel"][current_row], dtype=np.float32))
                task_targets.append(
                    np.asarray(h5["task_target"][current_row], dtype=np.float32)
                )

            def attr_text(name: str, default: str = "") -> str:
                value = h5.attrs.get(name, default)
                return value.decode("utf-8") if isinstance(value, bytes) else str(value)

            backend = attr_text("simulation_backend")
            sim_config = {
                "camera": attr_text("camera", "video_cam"),
                "disable_shadows": bool(h5.attrs.get("disable_shadows", False)),
                "physics_timestep": float(h5.attrs.get("physics_timestep", 0.002)),
                "control_decimation": int(h5.attrs.get("control_decimation", 25)),
                "task_bounds": attr_text("task_bounds"),
                "nominal_task_state": attr_text("nominal_task_state"),
                "rope_segments": int(h5.attrs.get("rope_segments", 22)),
                "rope_rest_length": float(h5.attrs["rope_rest_length"]),
                "rope_radius": float(h5.attrs.get("rope_radius", 0.008)),
                "rope_link_mass": float(h5.attrs.get("rope_link_mass", 0.0015)),
                "rope_joint_damping": float(h5.attrs.get("rope_joint_damping", 0.02)),
                "rope_joint_stiffness": float(
                    h5.attrs.get("rope_joint_stiffness", 0.005)
                ),
                "rope_joint_armature": float(
                    h5.attrs.get("rope_joint_armature", 0.001)
                ),
                "rope_endpoint_timeconst": float(
                    h5.attrs.get("rope_endpoint_timeconst", 0.02)
                ),
                "rope_endpoint_damping_ratio": float(
                    h5.attrs.get("rope_endpoint_damping_ratio", 0.09)
                ),
            }
        if self.frameskip != 1:
            raise ValueError("Visual DAgger currently requires frameskip=1.")
        if backend != "mjlab_mujoco_warp":
            raise ValueError("Visual DAgger requires an MJLab/MJWarp source dataset.")
        return {
            "pixels": np.stack(pixels),
            "qpos": np.stack(qpos),
            "qvel": np.stack(qvel),
            "task_target": np.stack(task_targets),
            "sim_config": sim_config,
        }

    def __del__(self) -> None:
        h5 = getattr(self, "_h5", None)
        if h5 is not None:
            try:
                h5.close()
            except Exception:
                pass


def koopman_forward(
    self, batch: dict[str, torch.Tensor], stage: str, args: argparse.Namespace
):
    from micrompc.examples.rope_manipulation.shared.inverse_qp import DifferentiableInverseQP

    pixels = preprocess_pixels(batch["pixels"], args.img_size)
    trajectory_emb = self.model.encode({"pixels": pixels})["emb"]
    markov_seq = build_markov_sequence(trajectory_emb, args.markov_order)
    current_state = markov_seq[:, 0]
    target_states = markov_seq[:, 1:]
    action = torch.nan_to_num(batch["action"], nan=0.0)

    z_pred, state_pred, z_target = self.model.predictor(
        current_state, action, target_states
    )
    sigreg_loss = self.sigreg(trajectory_emb.transpose(0, 1))
    # Keep the inverse solve in fp32: Cholesky is not reliable under bf16
    # autocast. Detaching visual states makes this term identify dynamics rather
    # than allowing the encoder to reshape its representation to game the QP.
    with torch.autocast(device_type=current_state.device.type, enabled=False):
        inverse_qp = DifferentiableInverseQP(args.num_preds, args.qp_regularization)
        qp_actions = inverse_qp.solve(
            self.model.predictor,
            current_state.detach().float(),
            target_states.detach().float(),
        )
    losses = KoopmanTrainer(
        LossWeights(
            args.prediction_weight,
            args.latent_weight,
            args.control_weight,
        )
    )(
        state_pred,
        target_states,
        z_pred,
        z_target,
        qp_actions,
        action.detach().float(),
        additional_losses={"sigreg": (args.sigreg_weight, sigreg_loss)},
    )
    loss = losses["loss"]
    prediction_loss, latent_loss, control_loss = (
        losses["state"],
        losses["latent"],
        losses["control"],
    )
    output = {
        "loss": loss,
        "prediction_loss": prediction_loss,
        "latent_loss": latent_loss,
        "sigreg_loss": sigreg_loss,
        "control_loss": control_loss,
    }
    logged_output = (
        {key: value.detach() for key, value in output.items()}
        if stage == "fit"
        else {f"{stage}/{key}": value.detach() for key, value in output.items()}
    )
    self.log_dict(
        logged_output,
        on_step=True,
        on_epoch=False,
        sync_dist=True,
        prog_bar=True,
    )
    return output


def sanitize_hparams(args: argparse.Namespace) -> dict[str, object]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }


from .training_metadata import attach_planning_metadata


def make_loader(
    dataset: Dataset,
    args: argparse.Namespace,
    *,
    shuffle: bool,
    drop_last: bool,
    persistent_workers: bool,
) -> DataLoader:
    kwargs = {
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "num_workers": args.num_workers,
        "pin_memory": args.accelerator == "gpu",
        "persistent_workers": args.num_workers > 0 and persistent_workers,
    }
    if args.num_workers > 0:
        kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(dataset, **kwargs)


def main() -> None:
    from micrompc.examples.rope_manipulation.shared.dagger import (
        DaggerMixtureDataset,
        DaggerReplayDataset,
        VisualDaggerCallback,
        replay_capacity,
    )

    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    dataset_path = args.dataset_path.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    finetune_dir = (
        args.finetune_dir.expanduser().resolve() if args.finetune_dir else None
    )

    if finetune_dir is None:
        output_run_dir = run_dir
        ckpt_path = output_run_dir / "last.ckpt"
        if (
            output_run_dir.exists()
            and any(output_run_dir.iterdir())
            and not ckpt_path.is_file()
        ):
            raise FileExistsError(
                f"Run dir already exists but has no resumable checkpoint: {output_run_dir}"
            )
    else:
        ckpt_path = run_dir / "last.ckpt"
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"Finetune source checkpoint not found: {ckpt_path}"
            )
        output_run_dir = finetune_dir
        if output_run_dir.exists():
            raise FileExistsError(
                f"Finetune output dir already exists: {output_run_dir}"
            )

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    output_run_dir.mkdir(parents=True, exist_ok=True)
    spt.set(
        cache_dir=str(output_run_dir),
        default_callbacks={
            "env_dump": False,
            "registry": False,
            "sklearn_checkpoint": False,
            "wandb_checkpoint": False,
            "trackio_checkpoint": False,
            "swanlab_checkpoint": False,
            "hf_checkpoint": False,
        },
    )
    dataset = KoopmanRopeDataset(
        dataset_path,
        markov_order=args.markov_order,
        num_preds=args.num_preds,
        frameskip=args.frameskip,
        action_dim=args.action_dim,
    )
    if len(dataset) < 2:
        raise ValueError(f"Need at least two valid windows, got {len(dataset)}.")
    train_len = min(max(1, int(len(dataset) * args.train_split)), len(dataset))
    val_len = len(dataset) - train_len
    train_set, val_set = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(args.seed),
    )
    replay = None
    train_data: Dataset = train_set
    if args.dagger:
        args.dagger_replay_capacity = replay_capacity(
            args.dagger_replay_mb,
            num_frames=dataset.num_frames,
            height=dataset.height,
            width=dataset.width,
            horizon=args.num_preds,
            action_dim=args.effective_action_dim,
        )
        replay = DaggerReplayDataset(
            output_run_dir / "dagger_replay.h5",
            capacity=args.dagger_replay_capacity,
            num_frames=dataset.num_frames,
            horizon=args.num_preds,
            height=dataset.height,
            width=dataset.width,
            action_dim=args.effective_action_dim,
        )
        train_data = DaggerMixtureDataset(
            train_set, replay, original_fraction=args.dagger_original_fraction
        )
    train_loader = make_loader(
        train_data,
        args,
        shuffle=True,
        drop_last=True,
        # Worker copies must be recreated so they observe each replay update.
        persistent_workers=args.persistent_workers and not args.dagger,
    )
    val_loader = (
        make_loader(
            val_set, args, shuffle=False, drop_last=False, persistent_workers=False
        )
        if val_len
        else None
    )

    world_model = build_model(args)
    attach_planning_metadata(
        world_model,
        args,
        action_mean=dataset.action_mean,
        action_std=dataset.action_std,
    )
    optimizers = {
        "model_opt": {
            "modules": "model",
            "optimizer": {
                "type": "AdamW",
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            },
            "scheduler": {"type": "LinearWarmupCosineAnnealingLR"},
            "interval": "epoch",
        }
    }
    module = spt.Module(
        model=world_model,
        sigreg=SketchGaussianRegularizer(
            knots=args.sigreg_knots, num_proj=args.sigreg_num_proj
        ),
        forward=partial(koopman_forward, args=args),
        optim=optimizers,
        hparams=sanitize_hparams(args),
    )

    config = sanitize_hparams(args)
    config.update(
        {
            "output_run_dir": str(output_run_dir),
            "resume_checkpoint": str(ckpt_path) if ckpt_path.is_file() else None,
        }
    )
    with (output_run_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    callbacks: list[Callback] = []
    if args.dagger:
        assert replay is not None
        callbacks.append(
            VisualDaggerCallback(
                args=args,
                source_dataset=dataset,
                source_indices=train_set.indices,
                replay=replay,
                action_mean=dataset.action_mean,
                action_std=dataset.action_std,
            )
        )
    callbacks.extend(
        [
            ModelCheckpoint(
                dirpath=output_run_dir,
                filename=f"{args.output_model_name}" + "_{epoch:03d}",
                save_last=True,
                save_top_k=0,
                every_n_epochs=1,
            ),
        ]
    )
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        callbacks=callbacks,
        default_root_dir=output_run_dir,
        logger=False,
        num_sanity_val_steps=1 if val_loader is not None else 0,
        enable_checkpointing=True,
    )
    data_module = spt.data.DataModule(train=train_loader, val=val_loader)
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([PosixPath])
    trainer.fit(
        module,
        datamodule=data_module,
        ckpt_path=str(ckpt_path) if ckpt_path.is_file() else None,
    )


if __name__ == "__main__":
    main()
