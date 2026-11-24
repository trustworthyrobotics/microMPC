"""Reusable Koopman training objective, independent of datasets and runners."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossWeights:
    """Weights for state, lifted-state, and recovered-control supervision."""

    state: float = 1.0
    latent: float = 0.1
    control: float = 0.01

    def __post_init__(self):
        if any(
            not 0 <= value < float("inf")
            for value in (self.state, self.latent, self.control)
        ):
            raise ValueError("loss weights must be finite and nonnegative")


class KoopmanTrainer:
    """Compute the shared objective for a PyTorch or Lightning training runner.

    Model rollout, optimization, logging, and simulator collection belong to the
    runner. This class owns the common training mathematics and retains gradients
    through the recovered controls. Additional losses supply example-specific
    reconstruction or representation regularization.
    """

    def __init__(self, weights: LossWeights):
        self.weights = weights

    def __call__(
        self,
        predicted_states,
        target_states,
        predicted_latents,
        target_latents,
        recovered_controls,
        target_controls,
        *,
        additional_losses: Mapping[str, tuple[float, torch.Tensor]] | None = None,
    ):
        pairs = (
            (predicted_states, target_states),
            (predicted_latents, target_latents),
            (recovered_controls, target_controls),
        )
        if any(pred.shape != target.shape for pred, target in pairs):
            raise ValueError("each prediction and target must have identical shapes")
        losses = {
            name: F.mse_loss(pred, target)
            for name, (pred, target) in zip(("state", "latent", "control"), pairs)
        }
        total = sum(
            getattr(self.weights, name) * value for name, value in losses.items()
        )
        for name, (weight, value) in (additional_losses or {}).items():
            if name in losses or name == "loss":
                raise ValueError(f"reserved loss name: {name}")
            if not 0 <= weight < float("inf"):
                raise ValueError(
                    "additional loss weights must be finite and nonnegative"
                )
            losses[name] = value
            total = total + weight * value
        return {"loss": total, **losses}
