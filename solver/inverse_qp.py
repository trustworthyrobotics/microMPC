"""Differentiable regularized control recovery for affine linear dynamics."""

from __future__ import annotations

import torch


def condensed_control_map(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, horizon: int
) -> torch.Tensor:
    """Return the block-Toeplitz map from controls to decoded future states."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    effect = b
    markov = []
    for _ in range(horizon):
        markov.append(c @ effect)
        effect = a @ effect
    markov = torch.stack(markov)
    steps = torch.arange(horizon, device=a.device)
    lags = steps[:, None] - steps[None, :]
    blocks = markov[lags.clamp_min(0)] * (lags >= 0)[..., None, None]
    return blocks.permute(0, 2, 1, 3).reshape(
        horizon * c.shape[0], horizon * b.shape[1]
    )


class DifferentiableInverseQP:
    """Solve mean squared tracking plus regularized normalized control effort.

    Inputs may be physical or normalized, provided matrices and states use the
    same coordinates. Optional control statistics define U = mean + std * V;
    the returned controls are V. Gradients propagate through the Cholesky solve.
    """

    def __init__(self, horizon: int, regularization: float) -> None:
        if horizon < 1 or not 0 < regularization < float("inf"):
            raise ValueError("horizon and regularization must be positive and finite")
        self.horizon = int(horizon)
        self.regularization = float(regularization)

    def solve(
        self,
        a,
        b,
        c,
        states,
        target_states,
        *,
        control_mean=None,
        control_std=None,
        demonstrated_controls=None,
    ):
        """Recover controls from a free rollout or an existing controlled rollout."""
        if states.shape != target_states.shape or states.ndim != 3:
            raise ValueError(
                "states and targets must have identical [batch, horizon, state] shapes"
            )
        if states.shape[1:] != (self.horizon, c.shape[0]):
            raise ValueError("state dimensions do not match the horizon and decoder")
        gamma = condensed_control_map(a, b, c, self.horizon)
        control_dim = b.shape[1]
        mean = (
            gamma.new_zeros(control_dim)
            if control_mean is None
            else control_mean.to(gamma)
        )
        std = (
            gamma.new_ones(control_dim)
            if control_std is None
            else control_std.to(gamma)
        )
        if mean.shape != (control_dim,) or std.shape != (control_dim,):
            raise ValueError("control statistics must have one entry per control")
        if (
            not torch.isfinite(mean).all()
            or not torch.isfinite(std).all()
            or (std <= 0).any()
        ):
            raise ValueError(
                "control statistics must be finite with positive standard deviations"
            )
        mean_sequence, std_sequence = (
            mean.repeat(self.horizon),
            std.repeat(self.horizon),
        )
        residual = (states - target_states).reshape(states.shape[0], -1)
        if demonstrated_controls is None:
            residual = residual + gamma @ mean_sequence
        else:
            if demonstrated_controls.shape != (
                states.shape[0],
                self.horizon,
                control_dim,
            ):
                raise ValueError(
                    "demonstrated controls must have shape [batch, horizon, control]"
                )
            residual = (
                residual + (mean_sequence - demonstrated_controls.flatten(1)) @ gamma.T
            )
        scaled_gamma = gamma * std_sequence.unsqueeze(0)
        scale = float(gamma.shape[0])
        hessian = scaled_gamma.T @ scaled_gamma / scale
        hessian = 0.5 * (hessian + hessian.T)
        hessian = hessian + self.regularization * torch.eye(
            hessian.shape[0], device=hessian.device, dtype=hessian.dtype
        )
        gradient = residual @ scaled_gamma / scale
        controls = torch.cholesky_solve(-gradient.T, torch.linalg.cholesky(hessian)).T
        return controls.reshape(states.shape[0], self.horizon, control_dim)
