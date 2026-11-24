# koop_train_lindec.py
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from micrompc.trainer import KoopmanTrainer, LossWeights
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from micrompc.examples.quad3d.shared.models import KoopmanDynamics
from micrompc.examples.quad3d.shared.dagger import (
    DaggerReplayBuffer,
    should_collect_dagger,
)
from micrompc.examples.quad3d.shared.utils import (
    parse_train_args,
    get_activation_fn,
    get_activation_name,
    resolve_path,
    print_config,
    get_unique_path,
)


DATA_PATH = "data/expert_data.pt"
MODEL_SAVE_PATH = "runs/koopman/last.pt"
LOG_DIR = "runs/koop_quad3d_lindec"

STATE_DIM = 12
CONTROL_DIM = 4
LATENT_DIM = 128
HIDDEN_WIDTH = 64
HIDDEN_DEPTH = 3
ACTIVATION_FN = nn.GELU

LEARNING_RATE = 1e-4
MIN_LEARNING_RATE = 1e-7
WEIGHT_DECAY = 1e-6
EPOCHS = 250
BATCH_SIZE = 4096
MULTI_STEP_HORIZON = 75
NUM_WORKERS = 24
UPDATE_EVERY = 1
ENABLE_NORMALIZATION = False

LAMBDA_STATE = 1.0
LAMBDA_LATENT = 0.1

DAGGER = False
DAGGER_START_EPOCH = 10
DAGGER_INTERVAL = 1
DAGGER_TRAJECTORIES = 5_000
DAGGER_MAX_BUFFER = 50_000
DAGGER_ORIGINAL_FRACTION = 0.5
DAGGER_HORIZON = 75
DAGGER_PLAN_BATCH_SIZE = 4096
DAGGER_QP_REGULARIZATION = 1e-4


from micrompc.examples.quad3d.shared.training import (
    QuadrotorDataset,
    checkpoint_path_for_epoch,
    save_checkpoint,
    validate_dagger_args,
    make_dataloader,
    dagger_config,
    collect_and_rebuild_dataloader,
)


def train(args):
    if args.update_every < 0:
        raise ValueError(f"--update_every must be >= 0, got {args.update_every}.")
    validate_dagger_args(args)

    data_path = resolve_path(args.data_path)
    model_save_path = resolve_path(args.model_save_path)
    log_dir = resolve_path(args.log_dir)

    model_save_path = get_unique_path(model_save_path)
    log_dir = get_unique_path(log_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    activation_fn = get_activation_fn(args.activation)

    writer = SummaryWriter(log_dir)
    dataset = QuadrotorDataset(
        data_path,
        multi_step_horizon=args.multi_step_horizon,
        enable_normalization=args.enable_normalization,
    )
    dataloader = make_dataloader(dataset, args)
    control_mean = dataset.u_k.mean(dim=0)
    control_std = dataset.u_k.std(dim=0).clamp_min(1e-6)
    replay_buffer = DaggerReplayBuffer(args.dagger_max_buffer) if args.dagger else None

    model_config = {
        "state_dim": args.state_dim,
        "control_dim": args.control_dim,
        "latent_dim": args.latent_dim,
        "hidden_width": args.hidden_width,
        "depth": args.hidden_depth,
        "activation_fn": activation_fn,
    }
    objective = KoopmanTrainer(LossWeights(args.lambda_state, args.lambda_latent, 0.0))
    model = KoopmanDynamics(**model_config).to(device)

    nn.init.eye_(model.A.weight)
    nn.init.xavier_uniform_(model.B.weight)

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * len(dataloader),
        eta_min=args.min_lr,
    )

    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)

    print("Starting training for linear-decoder Deep Koopman model (3D Quadrotor)")
    print_config(
        {
            **model_config,
            "total_latent_dim": args.latent_dim,
        },
        title="Model Configuration",
    )

    training_config = {
        "lr": args.lr,
        "min_lr": args.min_lr,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "multi_step_horizon": args.multi_step_horizon,
        "update_every": args.update_every,
        "enable_normalization": args.enable_normalization,
        "lambda_state": args.lambda_state,
        "lambda_latent": args.lambda_latent,
        **dagger_config(args),
    }
    print_config(training_config, title="Training Configuration")

    global_step = 0
    latest_update_checkpoint_path: Path | None = None
    pbar = tqdm(range(args.epochs), desc="Training Epochs")
    for epoch_idx in pbar:
        for x_k, u_seq, x_kp1_seq in dataloader:
            x_k = x_k.to(device)
            u_seq = u_seq.to(device)
            x_kp1_seq = x_kp1_seq.to(device)

            z_pred_seq, x_pred_seq, z_target_seq = model(x_k, u_seq, x_kp1_seq)

            losses = objective(
                x_pred_seq, x_kp1_seq, z_pred_seq, z_target_seq, u_seq, u_seq
            )
            loss_state, loss_latent, loss = (
                losses["state"],
                losses["latent"],
                losses["loss"],
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if global_step % 10 == 0:
                writer.add_scalar("Loss/Total", loss.item(), global_step)
                writer.add_scalar("Loss/State", loss_state.item(), global_step)
                writer.add_scalar("Loss/Latent", loss_latent.item(), global_step)
                writer.add_scalar(
                    "Hyperparams/Learning_Rate", scheduler.get_last_lr()[0], global_step
                )

            pbar.set_postfix(loss=f"{loss.item():.4f}")
            global_step += 1

        epoch = epoch_idx + 1
        if args.dagger and should_collect_dagger(
            epoch,
            args.epochs,
            args.dagger_start_epoch,
            args.dagger_interval,
        ):
            assert replay_buffer is not None
            print(
                f"Collecting {args.dagger_trajectories} DAgger trajectories at H={args.dagger_horizon}..."
            )
            dataloader, collected = collect_and_rebuild_dataloader(
                args,
                model,
                dataset,
                replay_buffer,
                control_mean,
                control_std,
                args.multi_step_horizon,
                device,
            )
            writer.add_scalar("DAgger/CollectedTrajectories", collected, epoch)
            writer.add_scalar("DAgger/BufferTrajectories", len(replay_buffer), epoch)
            print(
                f"DAgger replay: added {collected}; buffer contains {len(replay_buffer)} trajectories."
            )

        if args.update_every > 0 and epoch % args.update_every == 0:
            epoch_save_path = checkpoint_path_for_epoch(model_save_path, epoch)
            save_checkpoint(
                path=epoch_save_path,
                model=model,
                model_config=model_config,
                training_config=training_config,
                dataset=dataset,
                activation=args.activation,
                epoch=epoch,
            )
            if (
                latest_update_checkpoint_path is not None
                and latest_update_checkpoint_path != epoch_save_path
                and latest_update_checkpoint_path.exists()
            ):
                latest_update_checkpoint_path.unlink()
            latest_update_checkpoint_path = epoch_save_path
            print(f"Saved checkpoint to {epoch_save_path}")

    writer.close()
    save_checkpoint(
        path=model_save_path,
        model=model,
        model_config=model_config,
        training_config=training_config,
        dataset=dataset,
        activation=args.activation,
        epoch=args.epochs,
    )
    if (
        latest_update_checkpoint_path is not None
        and latest_update_checkpoint_path != Path(model_save_path)
        and latest_update_checkpoint_path.exists()
    ):
        latest_update_checkpoint_path.unlink()

    print(f"\nTraining complete. Model and config saved to {model_save_path}")
    print(f"To view logs, run: tensorboard --logdir {log_dir}")


if __name__ == "__main__":
    defaults = {
        "data_path": DATA_PATH,
        "model_save_path": MODEL_SAVE_PATH,
        "log_dir": LOG_DIR,
        "state_dim": STATE_DIM,
        "control_dim": CONTROL_DIM,
        "latent_dim": LATENT_DIM,
        "hidden_width": HIDDEN_WIDTH,
        "hidden_depth": HIDDEN_DEPTH,
        "activation": get_activation_name(ACTIVATION_FN),
        "lr": LEARNING_RATE,
        "min_lr": MIN_LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "multi_step_horizon": MULTI_STEP_HORIZON,
        "num_workers": NUM_WORKERS,
        "update_every": UPDATE_EVERY,
        "enable_normalization": ENABLE_NORMALIZATION,
        "lambda_state": LAMBDA_STATE,
        "lambda_latent": LAMBDA_LATENT,
        "dagger": DAGGER,
        "dagger_start_epoch": DAGGER_START_EPOCH,
        "dagger_interval": DAGGER_INTERVAL,
        "dagger_trajectories": DAGGER_TRAJECTORIES,
        "dagger_max_buffer": DAGGER_MAX_BUFFER,
        "dagger_original_fraction": DAGGER_ORIGINAL_FRACTION,
        "dagger_horizon": DAGGER_HORIZON,
        "dagger_plan_batch_size": DAGGER_PLAN_BATCH_SIZE,
        "dagger_qp_regularization": DAGGER_QP_REGULARIZATION,
    }
    args = parse_train_args(
        defaults,
        description="Train linear-decoder Deep Koopman model for 3D Quadrotor",
    )
    train(args)
