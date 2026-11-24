"""Dataset, checkpointing, and replay utilities shared by both trainers."""

from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from .models import KoopmanDynamics
from .dagger import (
    DaggerReplayBuffer,
    RandomMixtureDataset,
    collect_dagger_trajectories,
)


class QuadrotorDataset(Dataset):
    """Dataset for multi-step Koopman training with optional normalization."""

    def __init__(
        self,
        data_path=None,
        multi_step_horizon=1,
        enable_normalization=True,
        *,
        states=None,
        controls=None,
        normalization_stats=None,
    ):
        if states is None or controls is None:
            if data_path is None:
                raise ValueError(
                    "data_path is required when trajectories are not supplied."
                )
            demos = torch.load(data_path, weights_only=False)
            states = [
                torch.as_tensor(demo["states"].T, dtype=torch.float32) for demo in demos
            ]
            controls = [
                torch.as_tensor(demo["controls"].T, dtype=torch.float32)
                for demo in demos
            ]

        if len(states) != len(controls):
            raise ValueError(
                "states and controls must have the same number of trajectories."
            )
        self.states = [torch.as_tensor(state, dtype=torch.float32) for state in states]
        self.controls = [
            torch.as_tensor(control, dtype=torch.float32) for control in controls
        ]

        self.multi_step_horizon = multi_step_horizon
        self.enable_normalization = enable_normalization
        self.x_k, self.u_k, self.x_kp1 = self._process_data()
        if normalization_stats is None:
            self._calculate_normalization_stats()
        else:
            self.min = normalization_stats["min"].detach().clone()
            self.max = normalization_stats["max"].detach().clone()
            self.range = normalization_stats["range"].detach().clone()

    def _process_data(self):
        x_k_list, u_k_list, x_kp1_list = [], [], []
        for episode_states, episode_controls in zip(self.states, self.controls):
            x_k_list.append(episode_states[:-1])
            u_k_list.append(episode_controls)
            x_kp1_list.append(episode_states[1:])

        episode_lengths = [len(ep) for ep in u_k_list]
        x_k = torch.cat(x_k_list)
        u_k = torch.cat(u_k_list)
        x_kp1 = torch.cat(x_kp1_list)

        self.valid_indices = []
        current_start_index = 0
        M = self.multi_step_horizon
        for length in episode_lengths:
            valid_end_in_episode = length - M
            for i in range(valid_end_in_episode + 1):
                self.valid_indices.append(current_start_index + i)
            current_start_index += length

        return x_k, u_k, x_kp1

    def _calculate_normalization_stats(self):
        self.min = self.x_k.min(dim=0).values
        self.max = self.x_k.max(dim=0).values
        self.range = self.max - self.min
        self.range[self.range == 0] = 1e-6

    def normalize(self, x):
        min_dev = self.min.to(x.device)
        range_dev = self.range.to(x.device)
        return 2 * (x - min_dev) / range_dev - 1

    def preprocess_state(self, x):
        if self.enable_normalization:
            return self.normalize(x)
        return x

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_start_idx = self.valid_indices[idx]
        M = self.multi_step_horizon

        x_k = self.x_k[actual_start_idx]
        u_seq = self.u_k[actual_start_idx : actual_start_idx + M]
        x_kp1_seq = self.x_kp1[actual_start_idx : actual_start_idx + M]

        return self.preprocess_state(x_k), u_seq, self.preprocess_state(x_kp1_seq)


def checkpoint_path_for_epoch(model_save_path: str | Path, epoch: int) -> Path:
    path = Path(model_save_path)
    return path.with_name(f"{path.stem}_epoch{epoch:02d}{path.suffix}")


def save_checkpoint(
    path: str | Path,
    model: KoopmanDynamics,
    model_config: dict,
    training_config: dict,
    dataset: QuadrotorDataset,
    activation: str,
    epoch: int | None = None,
) -> None:
    model_config_save = model_config.copy()
    model_config_save["activation_fn"] = activation

    save_data = {
        "model_config": model_config_save,
        "training_config": training_config,
        "state_dict": model.state_dict(),
        "normalization_stats": {
            "min": dataset.min,
            "max": dataset.max,
            "range": dataset.range,
        },
    }
    if epoch is not None:
        save_data["epoch"] = epoch
    torch.save(save_data, path)


def validate_dagger_args(args) -> None:
    if not args.dagger:
        return
    if args.dagger_start_epoch < 0:
        raise ValueError("--dagger_start_epoch must be non-negative.")
    if args.dagger_interval < 1:
        raise ValueError("--dagger_interval must be positive.")
    if args.dagger_trajectories < 1:
        raise ValueError("--dagger_trajectories must be positive.")
    if args.dagger_max_buffer < 1:
        raise ValueError("--dagger_max_buffer must be positive.")
    if not 0.0 <= args.dagger_original_fraction <= 1.0:
        raise ValueError("--dagger_original_fraction must lie in [0, 1].")
    if args.dagger_horizon < 1:
        raise ValueError("--dagger_horizon must be positive.")


def make_dataloader(dataset: Dataset, args) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )


def dagger_config(args) -> dict:
    return {
        "dagger": args.dagger,
        "dagger_start_epoch": args.dagger_start_epoch,
        "dagger_interval": args.dagger_interval,
        "dagger_trajectories": args.dagger_trajectories,
        "dagger_max_buffer": args.dagger_max_buffer,
        "dagger_original_fraction": args.dagger_original_fraction,
        "dagger_horizon": args.dagger_horizon,
        "dagger_plan_batch_size": args.dagger_plan_batch_size,
        "dagger_qp_regularization": args.dagger_qp_regularization,
        "dagger_initial_state_perturbation": False,
        "dagger_eviction": "fifo",
        "dagger_replay_sampling": "random",
    }


def collect_and_rebuild_dataloader(
    args,
    model: KoopmanDynamics,
    original_dataset: QuadrotorDataset,
    replay_buffer: DaggerReplayBuffer,
    control_mean: torch.Tensor,
    control_std: torch.Tensor,
    training_horizon: int,
    device: torch.device,
) -> tuple[DataLoader, int]:
    if args.dagger_horizon < training_horizon:
        raise ValueError(
            f"DAgger horizon {args.dagger_horizon} must be at least the training "
            f"window {training_horizon}."
        )
    states, controls = collect_dagger_trajectories(
        model=model,
        original_dataset=original_dataset,
        count=args.dagger_trajectories,
        horizon=args.dagger_horizon,
        control_mean=control_mean,
        control_std=control_std,
        regularization=args.dagger_qp_regularization,
        device=device,
        plan_batch_size=args.dagger_plan_batch_size,
    )
    replay_buffer.extend(states, controls)
    normalization_stats = {
        "min": original_dataset.min,
        "max": original_dataset.max,
        "range": original_dataset.range,
    }
    generated_dataset = QuadrotorDataset(
        multi_step_horizon=training_horizon,
        enable_normalization=original_dataset.enable_normalization,
        states=replay_buffer.states,
        controls=replay_buffer.controls,
        normalization_stats=normalization_stats,
    )
    mixed_dataset = RandomMixtureDataset(
        original_dataset,
        generated_dataset,
        original_fraction=args.dagger_original_fraction,
    )
    return make_dataloader(mixed_dataset, args), len(states)
