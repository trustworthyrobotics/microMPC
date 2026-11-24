"""Visual Koopman model components used by this example."""

from __future__ import annotations
import torch
from torch import nn
from einops import rearrange


class SketchGaussianRegularizer(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer used by LE-WM."""

    def __init__(self, knots: int = 17, num_proj: int = 1024) -> None:
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """
        proj: (T, B, D)
        """
        a = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        a = a.div_(a.norm(p=2, dim=0))
        x_t = (proj @ a).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


class ProjectionHead(nn.Module):
    """Simple ProjectionHead with optional normalization and activation."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int | None = None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ) -> None:
        super().__init__()
        norm = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _build_koopman_encoder(
    input_dim: int,
    output_dim: int,
    *,
    hidden_width: int,
    depth: int,
    dropout: float = 0.0,
    activation_fn: type[nn.Module] = nn.GELU,
) -> nn.Sequential:
    if depth < 1:
        raise ValueError("depth must be at least 1.")

    layers: list[nn.Module] = []
    current_dim = input_dim
    for _ in range(depth):
        layers.append(nn.Linear(current_dim, hidden_width))
        layers.append(activation_fn())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        current_dim = hidden_width
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class KoopmanDynamics(nn.Module):
    """Koopman model with nonlinear encoder and linear decoder x = C z."""

    def __init__(
        self,
        *,
        state_dim: int,
        control_dim: int,
        latent_dim: int,
        hidden_width: int = 1024,
        depth: int = 4,
        dropout: float = 0.0,
        activation_fn: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        if state_dim < 1:
            raise ValueError("state_dim must be positive.")
        if control_dim < 1:
            raise ValueError("control_dim must be positive.")
        if latent_dim < 1:
            raise ValueError("latent_dim must be positive.")

        self.state_dim = int(state_dim)
        self.control_dim = int(control_dim)
        self.latent_dim = int(latent_dim)

        self.encoder = _build_koopman_encoder(
            self.state_dim,
            self.latent_dim,
            hidden_width=hidden_width,
            depth=depth,
            dropout=dropout,
            activation_fn=activation_fn,
        )
        self.C = nn.Linear(self.latent_dim, self.state_dim, bias=True)
        self.A = nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        self.B = nn.Linear(self.control_dim, self.latent_dim, bias=False)

    def lift_state(self, state: torch.Tensor) -> torch.Tensor:
        return self.encoder(state)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.C(latent)

    def decode_state(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decode(latent)

    def _rollout_latent(
        self, z_current: torch.Tensor, control: torch.Tensor
    ) -> torch.Tensor:
        z_pred = []
        for step in range(control.shape[1]):
            z_current = self.A(z_current) + self.B(control[:, step])
            z_pred.append(z_current)
        return torch.stack(z_pred, dim=1)

    def forward(
        self,
        x_k: torch.Tensor,
        u_seq: torch.Tensor,
        x_next_seq: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | torch.Tensor:
        if x_k.ndim != 2:
            raise ValueError(
                f"Expected x_k with shape [batch, state_dim], got {x_k.shape}."
            )
        if u_seq.ndim != 3:
            raise ValueError(
                f"Expected u_seq with shape [batch, horizon, control_dim], got {u_seq.shape}."
            )
        if x_k.shape[-1] != self.state_dim:
            raise ValueError(
                f"Expected state_dim={self.state_dim}, got {x_k.shape[-1]}."
            )
        if u_seq.shape[-1] != self.control_dim:
            raise ValueError(
                f"Expected control_dim={self.control_dim}, got {u_seq.shape[-1]}."
            )

        z_k = self.lift_state(x_k)
        z_pred_seq = self._rollout_latent(z_k, u_seq)
        x_pred_seq = self.decode(z_pred_seq)
        if x_next_seq is None:
            return z_pred_seq

        if x_next_seq.ndim != 3:
            raise ValueError(
                f"Expected x_next_seq with shape [batch, horizon, state_dim], got {x_next_seq.shape}."
            )
        x_next_seq_flat = x_next_seq.reshape(-1, self.state_dim)
        z_target_seq = self.lift_state(x_next_seq_flat).reshape(
            x_next_seq.shape[0], x_next_seq.shape[1], -1
        )
        return z_pred_seq, x_pred_seq, z_target_seq


class VisualKoopmanModel(nn.Module):
    """LE-WM joint-embedding predictive architecture."""

    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        action_encoder: nn.Module,
        projector: nn.Module | None = None,
        pred_proj: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()

    def encode(self, info: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        pixels = info["pixels"].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...")
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)
        if "action" in info:
            info["act_emb"] = self.action_encoder(info["action"])
        return info
