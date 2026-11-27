"""Online QP data aggregation shared by the quadrotor Koopman trainers."""

from collections.abc import Sequence

import torch
from torch.utils.data import Dataset

from micrompc.examples.quadrotor.shared.models import KoopmanDynamics
from micrompc.examples.quadrotor.shared.inverse_qp import DifferentiableInverseQP


MASS = 1.0
GRAVITY = -9.81
DT = 0.025
THRUST_MIN = 0.0
THRUST_MAX = -2.0 * GRAVITY * MASS
TORQUE_LIMIT = 0.1


class DaggerReplayBuffer:
    """FIFO trajectory replay; the permanent original data lives elsewhere."""

    def __init__(self, max_trajectories: int) -> None:
        if max_trajectories < 1:
            raise ValueError("dagger_max_buffer must be positive.")
        self.max_trajectories = int(max_trajectories)
        self.states: list[torch.Tensor] = []
        self.controls: list[torch.Tensor] = []

    def __len__(self) -> int:
        return len(self.states)

    def extend(
        self,
        states: Sequence[torch.Tensor],
        controls: Sequence[torch.Tensor],
    ) -> None:
        if len(states) != len(controls):
            raise ValueError(
                "states and controls must contain the same number of trajectories."
            )
        self.states.extend(state.detach().cpu() for state in states)
        self.controls.extend(control.detach().cpu() for control in controls)
        overflow = len(self.states) - self.max_trajectories
        if overflow > 0:
            del self.states[:overflow]
            del self.controls[:overflow]


class RandomMixtureDataset(Dataset):
    """Randomly mix permanent demonstrations with generated replay samples."""

    def __init__(
        self, original: Dataset, generated: Dataset, original_fraction: float
    ) -> None:
        if not 0.0 <= original_fraction <= 1.0:
            raise ValueError("dagger_original_fraction must lie in [0, 1].")
        if len(generated) == 0:
            raise ValueError("The generated dataset must not be empty.")
        self.original = original
        self.generated = generated
        self.original_fraction = float(original_fraction)

    def __len__(self) -> int:
        # Keep the epoch and learning-rate schedule length stable.
        return len(self.original)

    def __getitem__(self, idx: int):
        if torch.rand(()) < self.original_fraction:
            return self.original[idx]
        generated_idx = int(torch.randint(len(self.generated), ()).item())
        return self.generated[generated_idx]


from micrompc.trainer.replay import should_collect_dagger as should_collect_dagger


def _quadrotor_step(
    states: torch.Tensor, controls: torch.Tensor, dt: float
) -> torch.Tensor:
    """Batched forward-Euler step matching quadrotor/data/generate_data.py."""
    psi, theta, phi = states[:, 3], states[:, 4], states[:, 5]
    x_dot, y_dot, z_dot = states[:, 6], states[:, 7], states[:, 8]
    p, q, r = states[:, 9], states[:, 10], states[:, 11]
    thrust, u2, u3, u4 = controls.unbind(dim=1)

    derivatives = torch.stack(
        [
            x_dot,
            y_dot,
            z_dot,
            q * torch.sin(phi) / torch.cos(theta)
            + r * torch.cos(phi) / torch.cos(theta),
            q * torch.cos(phi) - r * torch.sin(phi),
            p
            + q * torch.sin(phi) * torch.tan(theta)
            + r * torch.cos(phi) * torch.tan(theta),
            thrust
            / MASS
            * (
                torch.sin(phi) * torch.sin(psi)
                + torch.cos(phi) * torch.cos(psi) * torch.sin(theta)
            ),
            thrust
            / MASS
            * (
                torch.cos(psi) * torch.sin(phi)
                - torch.cos(phi) * torch.sin(psi) * torch.sin(theta)
            ),
            GRAVITY + thrust / MASS * torch.cos(phi) * torch.cos(theta),
            ((0.1 - 0.3) / 0.5) * q * r + u2 / 0.5,
            ((0.3 - 0.5) / 0.1) * p * r + u3 / 0.1,
            ((0.5 - 0.1) / 0.3) * p * q + u4 / 0.3,
        ],
        dim=1,
    )
    return states + dt * derivatives


def _sample_reference_batch(original_dataset, count: int, horizon: int):
    eligible = [
        idx
        for idx, controls in enumerate(original_dataset.controls)
        if len(controls) >= horizon
    ]
    if not eligible:
        raise ValueError(f"No original trajectory is at least {horizon} steps long.")

    initial_states = []
    target_states = []
    for _ in range(count):
        trajectory_idx = eligible[int(torch.randint(len(eligible), ()).item())]
        states = original_dataset.states[trajectory_idx]
        controls = original_dataset.controls[trajectory_idx]
        start = int(torch.randint(len(controls) - horizon + 1, ()).item())
        initial_states.append(states[start])
        target_states.append(states[start + 1 : start + horizon + 1])
    return torch.stack(initial_states), torch.stack(target_states)


@torch.no_grad()
def collect_dagger_trajectories(
    model: KoopmanDynamics,
    original_dataset,
    count: int,
    horizon: int,
    control_mean: torch.Tensor,
    control_std: torch.Tensor,
    regularization: float,
    device: torch.device,
    plan_batch_size: int = 512,
    dt: float = DT,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Track original references with the QP and execute controls in true dynamics."""
    if count < 1:
        return [], []
    if plan_batch_size < 1:
        raise ValueError("dagger_plan_batch_size must be positive.")

    inverse_qp = DifferentiableInverseQP(
        horizon=horizon,
        control_mean=control_mean,
        control_std=control_std,
        regularization=regularization,
    )
    was_training = model.training
    model.eval()
    collected_states: list[torch.Tensor] = []
    collected_controls: list[torch.Tensor] = []

    for batch_start in range(0, count, plan_batch_size):
        batch_count = min(plan_batch_size, count - batch_start)
        initial_raw, target_raw = _sample_reference_batch(
            original_dataset, batch_count, horizon
        )
        initial_model = original_dataset.preprocess_state(initial_raw).to(device)
        target_model = original_dataset.preprocess_state(target_raw).to(device)
        normalized_controls = inverse_qp.solve(model, initial_model, target_model)
        controls = inverse_qp.denormalize_controls(normalized_controls)
        controls[..., 0].clamp_(THRUST_MIN, THRUST_MAX)
        controls[..., 1:].clamp_(-TORQUE_LIMIT, TORQUE_LIMIT)

        state = initial_raw.to(device)
        rollout = [state]
        for step in range(horizon):
            state = _quadrotor_step(state, controls[:, step], dt)
            rollout.append(state)
        states = torch.stack(rollout, dim=1)

        finite = torch.isfinite(states).all(dim=(1, 2)) & torch.isfinite(controls).all(
            dim=(1, 2)
        )
        for states_i, controls_i in zip(states[finite].cpu(), controls[finite].cpu()):
            collected_states.append(states_i)
            collected_controls.append(controls_i)

    model.train(was_training)
    return collected_states, collected_controls
