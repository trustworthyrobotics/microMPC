"""State-based Koopman model components used by this example."""

from __future__ import annotations
import torch
from torch import nn
from typing import Type


def create_mlp(
    input_dim: int,
    output_dim: int,
    hidden_width: int,
    depth: int,
    activation_fn: Type[nn.Module] = nn.GELU,
) -> nn.Sequential:
    """Helper function to create a multi-layer perceptron."""
    if depth == 0:
        return nn.Sequential(nn.Linear(input_dim, output_dim))

    layers = [nn.Linear(input_dim, hidden_width), activation_fn()]
    for _ in range(depth - 1):
        layers.extend([nn.Linear(hidden_width, hidden_width), activation_fn()])
    layers.append(nn.Linear(hidden_width, output_dim))
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_width: int,
        depth: int,
        activation_fn: Type[nn.Module],
    ):
        super().__init__()
        self.net = create_mlp(input_dim, latent_dim, hidden_width, depth, activation_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class KoopmanDynamics(nn.Module):
    """
    Koopman model with nonlinear encoder and linear decoder x = C z.
    """

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        latent_dim: int,
        hidden_width: int,
        depth: int,
        activation_fn: Type[nn.Module] | str,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim

        self.encoder = Encoder(
            state_dim, latent_dim, hidden_width, depth, activation_fn
        )
        self.C = nn.Linear(latent_dim, state_dim, bias=True)
        self.A = nn.Linear(latent_dim, latent_dim, bias=False)
        self.B = nn.Linear(control_dim, latent_dim, bias=False)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.C(z)

    def forward(
        self, x_k: torch.Tensor, u_seq: torch.Tensor, x_next_seq: torch.Tensor
    ) -> tuple:
        """
        x_k: [B, STATE_DIM] - Starting state
        u_seq: [B, M, CONTROL_DIM] - Sequence of M control inputs
        x_next_seq: [B, M, STATE_DIM] - Sequence of M target next states
        """
        M = u_seq.shape[1]
        state_dim = x_k.shape[-1]

        z_k = self.encoder(x_k)

        x_next_seq_flat = x_next_seq.view(-1, state_dim)
        z_target_seq_flat = self.encoder(x_next_seq_flat)
        z_target_seq = z_target_seq_flat.view(-1, M, self.latent_dim)

        z_pred_list = []
        z_current_pred = z_k
        for i in range(M):
            u_i = u_seq[:, i, :]
            z_next_pred = self.A(z_current_pred) + self.B(u_i)
            z_pred_list.append(z_next_pred)
            z_current_pred = z_next_pred

        z_pred_seq = torch.stack(z_pred_list, dim=1)
        x_pred_seq = self.decode(z_pred_seq)

        return z_pred_seq, x_pred_seq, z_target_seq
