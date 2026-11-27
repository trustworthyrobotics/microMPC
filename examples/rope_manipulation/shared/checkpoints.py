from __future__ import annotations
from pathlib import Path
import numpy as np
import torch


def latest_checkpoint(model_dir: Path) -> Path:
    """Return the single supported planning and resume checkpoint."""
    path = model_dir / "last.ckpt"
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def load_self_contained_model(checkpoint_path: Path, device: torch.device):
    """Reconstruct the model from Lightning state and saved hyperparameters."""
    from argparse import Namespace
    from .model_factory import build_model
    from .training_metadata import attach_planning_metadata

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = Namespace(**checkpoint["hyper_parameters"])
    state = {
        key.removeprefix("model."): value
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
    model = build_model(args)
    attach_planning_metadata(
        model,
        args,
        action_mean=state["planning_action_mean"],
        action_std=state["planning_action_std"],
    )
    model.load_state_dict(state, strict=True)
    config = model.planning_config
    action_mean, action_std = model.planning_action_mean, model.planning_action_std
    model = model.to(device).eval()
    model.requires_grad_(False)
    mean_np = action_mean.detach().float().cpu().numpy().reshape(-1)
    std_np = action_std.detach().float().cpu().numpy().reshape(-1)
    return model, config, mean_np, std_np


def require_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_arg)


def to_jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value
