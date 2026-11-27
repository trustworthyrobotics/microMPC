"""Train a Koopman model with a task-agnostic differentiable inverse QP.

The ordinary state and latent rollout losses are retained.  In addition, the
learned condensed dynamics are inverted to find the control sequence that best
explains each demonstrated future state sequence.  Matching that sequence to
the demonstrated controls trains the model's multi-step control response
without putting a goal-reaching task or hand-designed task Q/R in the QP.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from micrompc.trainer.training import KoopmanTrainer, LossWeights
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from micrompc.examples.quadrotor.shared.models import KoopmanDynamics
from micrompc.examples.quadrotor.shared.training import (
    QuadrotorDataset,
    checkpoint_path_for_epoch,
    collect_and_rebuild_dataloader,
    dagger_config,
    make_dataloader,
    save_checkpoint,
    validate_dagger_args,
)
from micrompc.examples.quadrotor.shared.dagger import (
    DaggerReplayBuffer,
    should_collect_dagger,
)
from micrompc.examples.quadrotor.shared.inverse_qp import DifferentiableInverseQP
from micrompc.examples.quadrotor.shared.utils import (
    get_activation_fn,
    get_activation_name,
    get_unique_path,
    parse_train_args,
    print_config,
    resolve_path,
)


DATA_PATH = "data/expert_data.pt"
MODEL_SAVE_PATH = "runs/koopman_inverse_qp/last.pt"
LOG_DIR = "runs/koop_quadrotor_lindec_diffQP"

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
NUM_WORKERS = 20
UPDATE_EVERY = 1
ENABLE_NORMALIZATION = False

LAMBDA_STATE = 1.0
LAMBDA_RECON = 1.5
LAMBDA_LATENT = 0.1
LAMBDA_CONTROLS = 1e-2
QP_REGULARIZATION = 1e-4

DAGGER = True
DAGGER_START_EPOCH = 10
DAGGER_INTERVAL = 1
DAGGER_TRAJECTORIES = 5_000
DAGGER_MAX_BUFFER = 50_000
DAGGER_ORIGINAL_FRACTION = 0.5
DAGGER_HORIZON = 75
DAGGER_PLAN_BATCH_SIZE = 4096
DAGGER_QP_REGULARIZATION = 1e-4


def train(args) -> None:
    if args.update_every < 0:
        raise ValueError(f"--update_every must be >= 0, got {args.update_every}.")
    if args.lambda_controls < 0.0:
        raise ValueError(f"--lambda_controls must be >= 0, got {args.lambda_controls}.")
    if not 0.0 <= args.lambda_recon < float("inf"):
        raise ValueError(
            f"--lambda_recon must be finite and >= 0, got {args.lambda_recon}."
        )
    validate_dagger_args(args)

    data_path = resolve_path(args.data_path)
    model_save_path = get_unique_path(resolve_path(args.model_save_path))
    log_dir = get_unique_path(resolve_path(args.log_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = QuadrotorDataset(
        data_path,
        multi_step_horizon=args.multi_step_horizon,
        enable_normalization=args.enable_normalization,
    )
    dataloader = make_dataloader(dataset, args)

    control_mean = dataset.u_k.mean(dim=0)
    control_std = dataset.u_k.std(dim=0).clamp_min(1e-6)
    inverse_qp = DifferentiableInverseQP(
        horizon=args.multi_step_horizon,
        control_mean=control_mean,
        control_std=control_std,
        regularization=args.qp_regularization,
    )
    objective = KoopmanTrainer(
        LossWeights(args.lambda_state, args.lambda_latent, args.lambda_controls)
    )
    replay_buffer = DaggerReplayBuffer(args.dagger_max_buffer) if args.dagger else None

    activation_fn = get_activation_fn(args.activation)
    model_config = {
        "state_dim": args.state_dim,
        "control_dim": args.control_dim,
        "latent_dim": args.latent_dim,
        "hidden_width": args.hidden_width,
        "depth": args.hidden_depth,
        "activation_fn": activation_fn,
    }
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
    writer = SummaryWriter(log_dir)
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)

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
        "lambda_recon": args.lambda_recon,
        "lambda_latent": args.lambda_latent,
        "lambda_controls": args.lambda_controls,
        "qp_regularization": args.qp_regularization,
        "control_mean": control_mean.tolist(),
        "control_std": control_std.tolist(),
        **dagger_config(args),
    }
    print("Starting task-agnostic differentiable inverse-QP training (3D Quadrotor)")
    print_config(model_config, title="Model Configuration")
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
            qp_controls = inverse_qp.solve_from_rollout(
                model,
                predicted_states=x_pred_seq,
                demonstrated_controls=u_seq,
                target_states=x_kp1_seq,
            )
            target_controls = inverse_qp.normalize_controls(u_seq)

            losses = objective(
                x_pred_seq,
                x_kp1_seq,
                z_pred_seq,
                z_target_seq,
                qp_controls,
                target_controls,
                additional_losses={
                    "reconstruction": (
                        args.lambda_recon,
                        nn.functional.mse_loss(model.decode(model.encoder(x_k)), x_k),
                    )
                },
            )
            loss = losses["loss"]
            loss_state, loss_latent = losses["state"], losses["latent"]
            loss_controls, loss_recon = losses["control"], losses["reconstruction"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if global_step % 10 == 0:
                writer.add_scalar("Loss/Total", loss.item(), global_step)
                writer.add_scalar("Loss/State", loss_state.item(), global_step)
                writer.add_scalar("Loss/Reconstruction", loss_recon.item(), global_step)
                writer.add_scalar("Loss/Latent", loss_latent.item(), global_step)
                writer.add_scalar("Loss/Controls", loss_controls.item(), global_step)
                writer.add_scalar(
                    "Hyperparams/Learning_Rate", scheduler.get_last_lr()[0], global_step
                )
            pbar.set_postfix(
                loss=f"{loss.item():.4f}", controls=f"{loss_controls.item():.4f}"
            )
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
                epoch_save_path,
                model,
                model_config,
                training_config,
                dataset,
                args.activation,
                epoch,
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
        model_save_path,
        model,
        model_config,
        training_config,
        dataset,
        args.activation,
        args.epochs,
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
        "lambda_recon": LAMBDA_RECON,
        "lambda_latent": LAMBDA_LATENT,
        "lambda_controls": LAMBDA_CONTROLS,
        "qp_regularization": QP_REGULARIZATION,
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
    train(
        parse_train_args(
            defaults,
            description="Train quadrotor Koopman model with a differentiable inverse QP",
        )
    )
