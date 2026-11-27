"""Shared trajectory plots, animations, and prediction-error reporting."""

from pathlib import Path
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def print_planning_error_metrics(
    step_traj_mse, step_pos_traj_mse, k_state_mse_curve, k_pos_mse_curve
):
    if not step_traj_mse:
        print("No planning-error metrics available (no MPC steps executed).")
        return
    step_traj_mse = np.asarray(step_traj_mse, dtype=np.float64)
    step_pos_traj_mse = np.asarray(step_pos_traj_mse, dtype=np.float64)
    print("\n" + "=" * 62)
    print("Planning Model-Error Metrics (Condensed Physical Map vs True Rollout)")
    print("=" * 62)
    print(
        f"state_12d_mse: mean={step_traj_mse.mean():.8f}, std={step_traj_mse.std():.8f}"
    )
    print(
        f"pos_xyz_mse:   mean={step_pos_traj_mse.mean():.8f}, std={step_pos_traj_mse.std():.8f}"
    )
    print(f"k=1:  state={k_state_mse_curve[0]:.8f}, position={k_pos_mse_curve[0]:.8f}")
    print(
        f"k={len(k_state_mse_curve)}: state={k_state_mse_curve[-1]:.8f}, position={k_pos_mse_curve[-1]:.8f}"
    )
    print("=" * 62)


def plot_3d(x_plan, x_true, start_state, goal_state, planned_trajs=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    if planned_trajs:
        for trajectory in planned_trajs:
            ax.plot(*trajectory[:, :3].T, color="gray", alpha=0.15, linewidth=1)
    if x_plan is not None:
        ax.plot(*x_plan[:, :3].T, "r", linewidth=2, label="Condensed physical plan")
    ax.plot(*x_true[:, :3].T, "b", linewidth=2, label="True dynamics")
    ax.scatter(*start_state[:3], c="g", s=60, label="Start")
    ax.scatter(*goal_state[:3], c="r", s=60, label="Goal")
    ax.set(xlabel="X", ylabel="Y", zlabel="Z", xlim=(0, 5), ylim=(0, 5), zlim=(0, 5))
    ax.set_title("Quadrotor MPC")
    ax.legend()
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    return fig


def plot_controls(u_traj):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(u_traj[:, 0], label="u1 (thrust)")
    ax1.set_title("Applied Controls")
    ax1.set_ylabel("Thrust")
    ax1.grid(True)
    ax1.legend()
    for index in range(1, 4):
        ax2.plot(u_traj[:, index], label=f"u{index + 1}")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Torques")
    ax2.grid(True)
    ax2.legend()
    plt.tight_layout()
    return fig


def show_3d_animation(
    start_state,
    goal_state,
    avg_solve_s,
    planned_trajs,
    true_traj,
    setup_axis=None,
    *,
    dt=0.025,
):
    if len(true_traj) < 2:
        return None, None
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*start_state[:3], c="g", s=60, label="Start")
    ax.scatter(*goal_state[:3], c="r", s=60, label="Goal")
    ax.set(xlabel="X", ylabel="Y", zlabel="Z", xlim=(0, 5), ylim=(0, 5), zlim=(0, 5))
    ax.set_title("Quadrotor MPC")
    if setup_axis is not None:
        setup_axis(ax)
    planned_line = None
    if planned_trajs:
        (planned_line,) = ax.plot([], [], [], "r", linewidth=2, label="Plan")
    (true_line,) = ax.plot([], [], [], "b", linewidth=2, label="True")
    ax.legend()

    def animate(frame):
        if planned_line is not None:
            plan = planned_trajs[frame]
            planned_line.set_data(plan[:, 0], plan[:, 1])
            planned_line.set_3d_properties(plan[:, 2])
        true_line.set_data(true_traj[: frame + 2, 0], true_traj[: frame + 2, 1])
        true_line.set_3d_properties(true_traj[: frame + 2, 2])
        return (true_line,) if planned_line is None else (planned_line, true_line)

    anim = animation.FuncAnimation(
        fig, animate, frames=len(true_traj) - 1, interval=dt * 1000, blit=False
    )
    return fig, anim


def save_animation(anim, video_save_path, frame_stride=1, *, dt=0.025):
    path = Path(video_save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fps = max(1, int(round(1.0 / (dt * frame_stride))))
        if path.suffix.lower() == ".mp4":
            writer = animation.FFMpegWriter(
                fps=fps,
                codec="libx264",
                extra_args=["-preset", "fast", "-pix_fmt", "yuv420p"],
            )
        else:
            writer = animation.PillowWriter(fps=fps)
        anim.save(path, writer=writer)
        print(f"Saved animation to {path}")
    except Exception as exc:
        print(f"Could not save animation: {exc}")
