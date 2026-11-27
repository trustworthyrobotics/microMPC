"""Planning metadata stored alongside trained visual dynamics."""

import argparse
import numpy as np
import torch


def attach_planning_metadata(
    model: torch.nn.Module,
    args: argparse.Namespace,
    *,
    action_mean: np.ndarray,
    action_std: np.ndarray,
) -> None:
    """Save learned-model settings and normalization needed by an external planner."""
    model.planning_config = {
        "format_version": 2,
        "planner": "linear_koopman_qp",
        "img_size": int(args.img_size),
        "embed_dim": int(args.embed_dim),
        "markov_order": int(args.markov_order),
        "markov_state_dim": int(args.markov_state_dim),
        "action_dim": int(args.action_dim),
        "effective_action_dim": int(args.effective_action_dim),
        "frameskip": int(args.frameskip),
        "training_horizon": int(args.num_preds),
        "pixel_normalization": "imagenet",
        "goal": "final frame of the selected episode",
        "state_cost": "position embedding only",
        "action_normalization": "applied training actions",
        "inverse_qp_control_weight": float(args.control_weight),
        "inverse_qp_regularization": float(args.qp_regularization),
        "dagger": bool(args.dagger),
    }
    model.register_buffer(
        "planning_action_mean",
        torch.as_tensor(action_mean, dtype=torch.float32).reshape(-1).clone(),
    )
    model.register_buffer(
        "planning_action_std",
        torch.as_tensor(action_std, dtype=torch.float32).reshape(-1).clone(),
    )
