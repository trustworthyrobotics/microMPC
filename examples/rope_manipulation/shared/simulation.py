#!/usr/bin/env python3
"""Collect batched GPU rope trajectories into an LE-WM-style HDF5 dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

if sys.platform == "darwin":
    os.environ.setdefault("MUJOCO_GL", "glfw")
else:
    os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import imageio.v2 as imageio
import mujoco
import numpy as np
import torch
from mjlab.sensor.camera_sensor import CameraSensor, CameraSensorCfg
from mjlab.sensor.sensor_context import SensorContext
from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
from tqdm.auto import tqdm

from micrompc.examples.rope_manipulation.shared.lab_env import (
    BaseEnvConfig,
    LEFT_ARM_SEED,
    LabEnv,
    RANDOM_CUBIC_SPLINE_MODE,
    RANDOM_WAYPOINT_MODE,
    RIGHT_ARM_SEED,
    RandomSplinePolicy,
    RandomWaypointPolicy,
    TaskBounds,
    TaskState,
    WS_EDGE_SAMPLE_MODE,
    WorkspaceEdgeSamplePolicy,
    build_lab_scene_xml,
    rope_rest_length,
)


DEFAULT_OUTDIR = "rope_manipulation/data/train"
DEFAULT_OUTPUT_NAME = "rope_random_cubic_spline.h5"
DEFAULT_CAMERA = "video_cam"
DEFAULT_MODE = RANDOM_CUBIC_SPLINE_MODE
DEFAULT_BATCH_SIZE = 512
IK_DEVICE = "cuda"
SAVE_MP4 = True
ACTION_DIM = 3
DISABLE_SHADOW = False
ROPE_NAME = "articulated_rope"
DEFAULT_ROPE_SEGMENTS = 22
DEFAULT_ROPE_RADIUS = 0.008
DEFAULT_ROPE_LINK_MASS = 0.0015
DEFAULT_ROPE_JOINT_DAMPING = 0.02
DEFAULT_ROPE_JOINT_STIFFNESS = 0.005
DEFAULT_ROPE_JOINT_ARMATURE = 0.001
DEFAULT_ROPE_ENDPOINT_TIMECONST = 0.02
DEFAULT_ROPE_ENDPOINT_DAMPING_RATIO = 0.09
ROPE_RGBA = (0.85, 0.2, 0.2, 1.0)
MJWARP_BACKGROUND_RGB = (25, 25, 51)
STARRY_SKY_SEED = 7
STARRY_SKY_DENSITY = 0.002

Policy = RandomWaypointPolicy | RandomSplinePolicy | WorkspaceEdgeSamplePolicy


@dataclass(frozen=True)
class ArticulatedRopeConfig:
    """Physical and visual parameters for the finite-link rope."""

    segments: int = DEFAULT_ROPE_SEGMENTS
    length: float | None = None
    radius: float = DEFAULT_ROPE_RADIUS
    link_mass: float = DEFAULT_ROPE_LINK_MASS
    joint_damping: float = DEFAULT_ROPE_JOINT_DAMPING
    joint_stiffness: float = DEFAULT_ROPE_JOINT_STIFFNESS
    joint_armature: float = DEFAULT_ROPE_JOINT_ARMATURE
    endpoint_timeconst: float = DEFAULT_ROPE_ENDPOINT_TIMECONST
    endpoint_damping_ratio: float = DEFAULT_ROPE_ENDPOINT_DAMPING_RATIO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--target-transitions", type=int, default=None)
    parser.add_argument("--num-trajectories", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument("--min-steps", type=int, default=3)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Parallel MJWarp worlds. The default is conservative because buffered RGB scales linearly.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--ik-device",
        choices=("cpu", "cuda"),
        default=IK_DEVICE,
        help="Run inverse kinematics with the existing serial CPU solver or batched CUDA solver.",
    )
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--quality", type=int, default=8)
    parser.add_argument("--compression", choices=("none", "lzf", "gzip"), default="lzf")
    parser.add_argument("--camera", default=DEFAULT_CAMERA)
    parser.add_argument(
        "--save-mp4",
        action=argparse.BooleanOptionalAction,
        default=SAVE_MP4,
        help="Write per-trajectory MP4 previews; use --no-save-mp4 for faster collection.",
    )
    parser.add_argument(
        "--disable-shadows", action="store_true", default=DISABLE_SHADOW
    )
    parser.add_argument(
        "--mode",
        default=DEFAULT_MODE,
        choices=(RANDOM_CUBIC_SPLINE_MODE, RANDOM_WAYPOINT_MODE, WS_EDGE_SAMPLE_MODE),
    )
    parser.add_argument("--control-decimation", type=int, default=25)
    parser.add_argument("--segment-duration", type=float, default=4.0)
    parser.add_argument("--num-waypoints", type=int, default=6)
    parser.add_argument("--midpoint-inflation-scale", type=float, default=0.08)
    parser.add_argument("--goal-tolerance", type=float, default=1e-4)
    parser.add_argument(
        "--rope-segments",
        type=int,
        default=DEFAULT_ROPE_SEGMENTS,
        help="Number of rigid capsule links. More links look smoother but add ball-joint DOFs.",
    )
    parser.add_argument(
        "--rope-radius",
        type=float,
        default=DEFAULT_ROPE_RADIUS,
        help="Capsule radius in metres; the default matches the CPU tendon width.",
    )
    parser.add_argument(
        "--rope-length",
        type=float,
        default=None,
        help="Total chain length in metres; defaults to the CPU tendon's 0.825 m upper length.",
    )
    parser.add_argument("--rope-link-mass", type=float, default=DEFAULT_ROPE_LINK_MASS)
    parser.add_argument(
        "--rope-joint-damping", type=float, default=DEFAULT_ROPE_JOINT_DAMPING
    )
    parser.add_argument(
        "--rope-joint-stiffness", type=float, default=DEFAULT_ROPE_JOINT_STIFFNESS
    )
    parser.add_argument(
        "--rope-joint-armature", type=float, default=DEFAULT_ROPE_JOINT_ARMATURE
    )
    parser.add_argument(
        "--rope-endpoint-timeconst",
        type=float,
        default=DEFAULT_ROPE_ENDPOINT_TIMECONST,
        help="MuJoCo equality time constant for the right-gripper attachment; lower is stiffer.",
    )
    parser.add_argument(
        "--rope-endpoint-damping-ratio",
        type=float,
        default=DEFAULT_ROPE_ENDPOINT_DAMPING_RATIO,
    )
    args = parser.parse_args()
    from micrompc.paths import resolve_path

    args.outdir = resolve_path(args.outdir)
    return args


def create_resizable_dataset(
    h5: h5py.File,
    name: str,
    shape_tail: tuple[int, ...],
    dtype: np.dtype | type,
    *,
    compression: str | None = None,
    chunks: tuple[int, ...] | bool | None = True,
) -> h5py.Dataset:
    return h5.create_dataset(
        name,
        shape=(0, *shape_tail),
        maxshape=(None, *shape_tail),
        dtype=dtype,
        compression=compression,
        chunks=chunks,
    )


def append_rows(dataset: h5py.Dataset, values: np.ndarray) -> tuple[int, int]:
    start = int(dataset.shape[0])
    end = start + int(values.shape[0])
    dataset.resize((end, *dataset.shape[1:]))
    dataset[start:end] = values
    return start, end


def valid_training_windows(
    ep_len: np.ndarray,
    *,
    history_size: int = 3,
    num_preds: int = 1,
    frameskip: int = 1,
) -> int:
    num_steps = history_size + num_preds
    required_offset = max((num_steps - 1) * frameskip, history_size * frameskip)
    return int(np.maximum(ep_len - required_offset, 0).sum())


def should_continue(
    args: argparse.Namespace, num_trajectories: int, total_transitions: int
) -> bool:
    if args.target_transitions is not None:
        return total_transitions < args.target_transitions
    return num_trajectories < args.num_trajectories


def make_policy(args: argparse.Namespace, bounds: TaskBounds, *, seed: int) -> Policy:
    if args.mode == RANDOM_CUBIC_SPLINE_MODE:
        return RandomSplinePolicy(
            bounds,
            seed=seed,
            segment_duration=args.segment_duration,
            midpoint_inflation_scale=args.midpoint_inflation_scale,
        )
    if args.mode == RANDOM_WAYPOINT_MODE:
        return RandomWaypointPolicy(
            bounds=bounds,
            seed=seed,
            segment_duration=args.segment_duration,
            num_waypoints=args.num_waypoints,
        )
    if args.mode == WS_EDGE_SAMPLE_MODE:
        return WorkspaceEdgeSamplePolicy(
            bounds=bounds, segment_duration=args.segment_duration
        )
    raise ValueError(f"Unsupported mode: {args.mode}")


def policy_duration(policy: Policy) -> float:
    if isinstance(policy, RandomWaypointPolicy):
        return float((len(policy._waypoints) - 1) * policy.segment_duration)
    return float(policy.segment_duration)


def policy_goal_state(policy: Policy) -> np.ndarray:
    if isinstance(policy, RandomWaypointPolicy):
        return np.asarray(policy._waypoints[-1], dtype=np.float32)
    if isinstance(policy, RandomSplinePolicy):
        return np.asarray(
            policy.bounds.clip(policy._control_points[-1]).as_array(), dtype=np.float32
        )
    if isinstance(policy, WorkspaceEdgeSamplePolicy):
        return np.asarray(
            policy.bounds.clip(policy._segments[0][1]).as_array(), dtype=np.float32
        )
    raise TypeError(f"Unsupported policy type: {type(policy)!r}")


def equal_link_rope_points(
    left: np.ndarray,
    right: np.ndarray,
    *,
    segments: int,
    target_length: float,
) -> np.ndarray:
    """Construct an exact equal-link, gravity-plane arc between two endpoints."""
    chord_vector = np.asarray(right, dtype=np.float64) - np.asarray(
        left, dtype=np.float64
    )
    chord = float(np.linalg.norm(chord_vector))
    if chord > target_length + 1e-6:
        raise ValueError(
            f"Rope endpoints are {chord:.6f} m apart, exceeding rope length {target_length:.6f} m."
        )
    if chord < 1e-9:
        raise ValueError(
            "Rope endpoints are coincident; a chain orientation is undefined."
        )
    link_length = target_length / float(segments)
    chord_axis = chord_vector / chord
    down = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    sag_axis = down - np.dot(down, chord_axis) * chord_axis
    if np.linalg.norm(sag_axis) < 1e-8:
        sag_axis = np.cross(chord_axis, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(sag_axis) < 1e-8:
            sag_axis = np.cross(chord_axis, np.array([0.0, 1.0, 0.0]))
    sag_axis /= np.linalg.norm(sag_axis)

    ratio = min(1.0, chord / target_length)
    # Positive angles point along ``sag_axis`` (down), so the chain starts
    # downward and ends upward.
    phase = np.linspace(1.0, -1.0, segments, dtype=np.float64)
    lower, upper = 0.0, np.pi - 1e-6
    for _ in range(64):
        middle = 0.5 * (lower + upper)
        if float(np.cos(middle * phase).mean()) > ratio:
            lower = middle
        else:
            upper = middle
    angles = upper * phase
    directions = (
        np.cos(angles)[:, None] * chord_axis[None, :]
        + np.sin(angles)[:, None] * sag_axis[None, :]
    )
    points = np.empty((segments + 1, 3), dtype=np.float64)
    points[0] = left
    points[1:] = left + np.cumsum(link_length * directions, axis=0)
    points[-1] = right
    return points


def capsule_xmat(directions: np.ndarray) -> np.ndarray:
    """Return rotation matrices whose local z axes follow directions."""
    z_axis = directions / np.maximum(
        np.linalg.norm(directions, axis=1, keepdims=True), 1e-12
    )
    helper = np.zeros_like(z_axis)
    helper[:, 2] = 1.0
    near_parallel = np.abs(z_axis[:, 2]) > 0.9
    helper[near_parallel] = np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(helper, z_axis)
    x_axis /= np.maximum(np.linalg.norm(x_axis, axis=1, keepdims=True), 1e-12)
    y_axis = np.cross(z_axis, x_axis)
    return np.stack((x_axis, y_axis, z_axis), axis=2)


def rope_joint_quaternions(points: np.ndarray) -> np.ndarray:
    """Return parent-relative ball-joint quaternions for a link polyline."""
    rotations = capsule_xmat(np.diff(points, axis=0))
    quaternions = np.empty((len(rotations), 4), dtype=np.float64)
    parent_rotation = np.eye(3, dtype=np.float64)
    for index, rotation in enumerate(rotations):
        mujoco.mju_mat2Quat(
            quaternions[index], (parent_rotation.T @ rotation).reshape(-1)
        )
        parent_rotation = rotation
    return quaternions


def build_articulated_rope_scene(
    ik_env: LabEnv,
    config: ArticulatedRopeConfig,
) -> mujoco.MjModel:
    """Build a collision-free rigid-link rope pinned to both grippers."""
    rest_length = (
        rope_rest_length(ik_env.task_bounds, ik_env.rope_spec)
        if config.length is None
        else config.length
    )
    link_length = rest_length / float(config.segments)
    rgba = " ".join(str(value) for value in ROPE_RGBA)
    lines = [f'    <body name="{ROPE_NAME}_left_anchor" mocap="true">']
    indent = "      "
    for index in range(config.segments):
        body_pos = "0 0 0" if index == 0 else f"0 0 {link_length:.9f}"
        link_name = f"{ROPE_NAME}_link_{index:02d}"
        # The first ball joint is the left endpoint pin, not a rope bend.
        # Leaving a spring or damper on it biases the first link toward the
        # mocap body's world-aligned frame and produces a persistent endpoint
        # kink. Internal joints retain the requested bending properties.
        joint_damping = 0.0 if index == 0 else config.joint_damping
        joint_stiffness = 0.0 if index == 0 else config.joint_stiffness
        lines.extend(
            [
                f'{indent}<body name="{link_name}" pos="{body_pos}">',
                (
                    f'{indent}  <joint name="{link_name}_joint" type="ball" '
                    f'damping="{joint_damping:.9f}" stiffness="{joint_stiffness:.9f}" '
                    f'armature="{config.joint_armature:.9f}"/>'
                ),
                (
                    f'{indent}  <geom name="{link_name}_geom" type="capsule" '
                    f'fromto="0 0 0 0 0 {link_length:.9f}" size="{config.radius:.9f}" '
                    f'mass="{config.link_mass:.9f}" rgba="{rgba}" group="2" '
                    'contype="0" conaffinity="0"/>'
                ),
            ]
        )
        if index == config.segments - 1:
            lines.append(
                f'{indent}  <site name="{ROPE_NAME}_right_site" pos="0 0 {link_length:.9f}" size="0.001" rgba="0 0 0 0"/>'
            )
        indent += "  "
    for _ in range(config.segments):
        indent = indent[:-2]
        lines.append(f"{indent}</body>")
    lines.append("    </body>")

    scene_xml = build_lab_scene_xml(BaseEnvConfig(worldbody_extra_xml="\n".join(lines)))
    scene_xml = scene_xml.replace(
        (
            '<texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" '
            'rgb2="0 0 0" width="512" height="3072"/>'
        ),
        (
            '<texture type="skybox" builtin="gradient" rgb1=".4 .6 .8" '
            'rgb2="0 0 0" width="800" height="800" '
            'mark="random" markrgb="1 1 1"/>'
        ),
        1,
    )
    # Keep the CPU tendon as an upper-length safeguard, but hide its straight
    # renderer because the articulated capsules are now the visible rope.
    scene_xml = scene_xml.replace(
        'stiffness="0" damping="0" rgba="0.85 0.2 0.2 1">',
        'stiffness="0" damping="0" rgba="0 0 0 0">',
        1,
    )
    equality_xml = f'''  <equality>
    <connect name="{ROPE_NAME}_right_attachment"
             body1="{ROPE_NAME}_link_{config.segments - 1:02d}"
             body2="arm2_ee_hook"
             anchor="0 0 0"
             solref="{config.endpoint_timeconst:.9f} {config.endpoint_damping_ratio:.9f}"/>
  </equality>
'''
    scene_xml = scene_xml.replace("</mujoco>", equality_xml + "</mujoco>")
    model = mujoco.MjModel.from_xml_string(scene_xml)

    equality_id = model.equality(f"{ROPE_NAME}_right_attachment").id
    right_site_id = model.site("arm2_attachment_site").id
    right_body_id = model.body("arm2_ee_hook").id
    if int(model.site_bodyid[right_site_id]) != right_body_id:
        raise RuntimeError(
            "The right attachment site is no longer a direct child of arm2_ee_hook."
        )
    model.eq_data[equality_id, :3] = np.array([0.0, 0.0, link_length])
    model.eq_data[equality_id, 3:6] = model.site_pos[right_site_id]
    model.geom_rgba[model.geom("table").id] = np.array([1.0, 1.0, 1.0, 1.0])
    return model


def move_scene_light_to_camera(model: mujoco.MjModel, camera_name: str) -> None:
    """Use a point light attached to the selected camera's parent body.

    MuJoCo Warp does not render MuJoCo's built-in camera headlight. Moving the
    scene light onto the camera provides the equivalent viewpoint-relative
    illumination and avoids the dark robot faces produced by the top light.
    """
    camera_id = model.camera(camera_name).id
    light_id = model.light("main_light").id
    model.light_type[light_id] = mujoco.mjtLightType.mjLIGHT_POINT
    model.light_mode[light_id] = mujoco.mjtCamLight.mjCAMLIGHT_FIXED
    model.light_bodyid[light_id] = model.cam_bodyid[camera_id]
    model.light_pos[light_id] = model.cam_pos[camera_id]


def make_starry_background(height: int, width: int) -> np.ndarray:
    """Return a deterministic star field matching the original bench sky."""
    background = np.empty((height, width, 3), dtype=np.uint8)
    background[:] = MJWARP_BACKGROUND_RGB
    rng = np.random.default_rng(STARRY_SKY_SEED)
    star_count = max(1, round(height * width * STARRY_SKY_DENSITY))
    star_pixels = rng.choice(height * width, size=star_count, replace=False)
    brightness = rng.integers(170, 256, size=star_count, dtype=np.uint8)
    background.reshape(-1, 3)[star_pixels] = brightness[:, None]
    return background


class BatchedRopeSimulation:
    """Rope worlds simulated by MJLab/MuJoCo Warp with batched GPU RGB."""

    def __init__(
        self,
        *,
        ik_env: LabEnv,
        batch_size: int,
        width: int,
        height: int,
        camera_name: str,
        disable_shadows: bool,
        device: str,
        rope_config: ArticulatedRopeConfig,
    ) -> None:
        model = build_articulated_rope_scene(ik_env, rope_config)
        move_scene_light_to_camera(model, camera_name)
        self.batch_size = batch_size
        self.device = device
        self.model = model
        self.rope_config = rope_config
        self.robot_nq = ik_env.model.nq
        self.robot_nv = ik_env.model.nv
        self.arm1_site_id = model.site("arm1_attachment_site").id
        self.arm2_site_id = model.site("arm2_attachment_site").id
        self.robot_qposadr = np.concatenate(
            [
                np.asarray(
                    [
                        model.joint(f"arm{arm}_joint{index}").qposadr[0]
                        for index in range(1, 8)
                    ],
                    dtype=np.int64,
                )
                for arm in (1, 2)
            ]
        )
        self.robot_dofadr = np.concatenate(
            [
                np.asarray(
                    [
                        model.joint(f"arm{arm}_joint{index}").dofadr[0]
                        for index in range(1, 8)
                    ],
                    dtype=np.int64,
                )
                for arm in (1, 2)
            ]
        )
        self.rope_joint_qposadr = np.asarray(
            [
                model.joint(f"{ROPE_NAME}_link_{index:02d}_joint").qposadr[0]
                for index in range(rope_config.segments)
            ],
            dtype=np.int64,
        )
        self.rope_left_mocap_id = int(
            model.body_mocapid[model.body(f"{ROPE_NAME}_left_anchor").id]
        )
        self.rope_rest_length = float(
            rope_rest_length(ik_env.task_bounds, ik_env.rope_spec)
            if rope_config.length is None
            else rope_config.length
        )

        integrator = (
            "implicitfast"
            if model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
            else "euler"
        )
        self.sim = Simulation(
            num_envs=batch_size,
            cfg=SimulationCfg(
                mujoco=MujocoCfg(
                    timestep=float(model.opt.timestep), integrator=integrator
                ),
            ),
            model=model,
            device=device,
        )
        camera = CameraSensorCfg(
            name="dataset_camera",
            camera_name=camera_name,
            width=width,
            height=height,
            data_types=("rgb",),
            use_textures=True,
            use_shadows=not disable_shadows,
            clone_data=False,
        ).build()
        assert isinstance(camera, CameraSensor)
        camera.initialize(model, self.sim.wp_model, self.sim.wp_data, device)
        self.sim.set_sensor_context(
            SensorContext(
                model,
                self.sim.wp_model,
                self.sim.wp_data,
                camera_sensors=[camera],
                raycast_sensors=[],
                device=device,
            )
        )
        self.camera = camera
        self.mjwarp_background_rgb = torch.tensor(
            MJWARP_BACKGROUND_RGB, dtype=torch.uint8, device=device
        )
        self.starry_background = torch.as_tensor(
            make_starry_background(height, width), dtype=torch.uint8, device=device
        ).unsqueeze(0)

    @property
    def timestep(self) -> float:
        return float(self.model.opt.timestep)

    def reset(
        self, qpos: np.ndarray | torch.Tensor, controls: np.ndarray | torch.Tensor
    ) -> None:
        if qpos.shape != (self.batch_size, self.robot_nq):
            raise ValueError(
                f"Expected robot qpos shape {(self.batch_size, self.robot_nq)}, got {qpos.shape}."
            )
        self.sim.reset()
        self.sim.data.qpos[:, self.robot_qposadr] = torch.as_tensor(
            qpos, dtype=torch.float32, device=self.device
        )
        self.sim.data.qvel.zero_()
        self.sim.data.ctrl[:] = torch.as_tensor(
            controls, dtype=torch.float32, device=self.device
        )
        self.sim.forward()
        self._initialize_rope()
        self.sim.forward()

    def step(self, controls: np.ndarray | torch.Tensor, decimation: int) -> None:
        self.sim.data.ctrl[:] = torch.as_tensor(
            controls, dtype=torch.float32, device=self.device
        )
        for _ in range(decimation):
            self._sync_left_anchor()
            self.sim.step()
        self.sim.forward()

    def _sync_left_anchor(self) -> None:
        left = self.sim.data.site_xpos[:, self.arm1_site_id, :]
        self.sim.data.mocap_pos[:, self.rope_left_mocap_id, :] = left
        self.sim.data.mocap_quat[:, self.rope_left_mocap_id, :].zero_()
        self.sim.data.mocap_quat[:, self.rope_left_mocap_id, 0] = 1.0

    def _initialize_rope(self) -> None:
        """Initialize the rigid links in an equality-consistent sagging arc."""
        left = self.sim.data.site_xpos[:, self.arm1_site_id, :].cpu().numpy()
        right = self.sim.data.site_xpos[:, self.arm2_site_id, :].cpu().numpy()
        joint_quats = np.empty(
            (self.batch_size, self.rope_config.segments, 4), dtype=np.float32
        )
        for env_index in range(self.batch_size):
            points = equal_link_rope_points(
                left[env_index],
                right[env_index],
                segments=self.rope_config.segments,
                target_length=self.rope_rest_length,
            )
            joint_quats[env_index] = rope_joint_quaternions(points)
        self._sync_left_anchor()
        for index, qposadr in enumerate(self.rope_joint_qposadr):
            self.sim.data.qpos[:, qposadr : qposadr + 4] = torch.as_tensor(
                joint_quats[:, index], dtype=torch.float32, device=self.device
            )
        self.sim.data.qvel.zero_()

    def render(self) -> np.ndarray:
        self.camera.update(0.0)
        self.sim.sense()
        rgb = self.camera.data.rgb
        if rgb is None:
            raise RuntimeError("MJWarp camera did not produce RGB data.")
        # MuJoCo Warp currently ignores skybox textures and fills every missed
        # ray with a constant color. Replace only those untouched pixels with
        # the original bench-style star field.
        background_mask = torch.all(
            rgb == self.mjwarp_background_rgb, dim=-1, keepdim=True
        )
        rgb = torch.where(background_mask, self.starry_background, rgb)
        return rgb.cpu().numpy()

    def robot_qpos(self) -> np.ndarray:
        return self.sim.data.qpos[:, self.robot_qposadr].cpu().numpy()

    def robot_qpos_tensor(self) -> torch.Tensor:
        return self.sim.data.qpos[:, self.robot_qposadr]

    def extract_step_info(
        self, targets: np.ndarray, elapsed_times: np.ndarray
    ) -> dict[str, np.ndarray]:
        qpos = (
            self.sim.data.qpos[:, self.robot_qposadr].cpu().numpy().astype(np.float32)
        )
        qvel = self.sim.data.qvel[:, self.robot_dofadr].cpu().numpy().astype(np.float32)
        control = self.sim.data.ctrl.cpu().numpy().astype(np.float32)
        left = (
            self.sim.data.site_xpos[:, self.arm1_site_id, :]
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        right = (
            self.sim.data.site_xpos[:, self.arm2_site_id, :]
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        rope_length = np.linalg.norm(right - left, axis=1, keepdims=True).astype(
            np.float32
        )
        time = elapsed_times[:, None].astype(np.float32)
        observation = np.concatenate(
            (targets, qpos, qvel, control, left, right, rope_length), axis=1
        ).astype(np.float32)
        return {
            "observation": observation,
            "task_target": targets.astype(np.float32),
            "qpos": qpos,
            "qvel": qvel,
            "control": control,
            "left_attachment_pos": left,
            "right_attachment_pos": right,
            "rope_length": rope_length,
            "time": time,
        }


class BatchedCudaIKSolver:
    """Batched damped-least-squares IK using MJWarp kinematics and torch CUDA."""

    def __init__(
        self, ik_env: LabEnv, *, batch_size: int, device: str = "cuda"
    ) -> None:
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "--ik-device cuda requested, but torch cannot access a CUDA device."
            )
        self.device = device
        self.batch_size = batch_size
        self.config = ik_env.task_controller.config
        integrator = (
            "implicitfast"
            if ik_env.model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
            else "euler"
        )
        self.sim = Simulation(
            num_envs=batch_size,
            cfg=SimulationCfg(
                mujoco=MujocoCfg(
                    timestep=float(ik_env.model.opt.timestep), integrator=integrator
                ),
            ),
            model=ik_env.model,
            device=device,
        )
        self.dof_indices = torch.as_tensor(
            np.stack((ik_env.arm1_dofadr, ik_env.arm2_dofadr)),
            dtype=torch.long,
            device=device,
        )
        self.site_ids = torch.as_tensor(
            (ik_env.arm1_site_id, ik_env.arm2_site_id), dtype=torch.long, device=device
        )
        site_body_ids = ik_env.model.site_bodyid[
            [ik_env.arm1_site_id, ik_env.arm2_site_id]
        ]
        self.root_body_ids = torch.as_tensor(
            ik_env.model.body_rootid[site_body_ids], dtype=torch.long, device=device
        )
        self.lower = torch.as_tensor(
            np.stack(
                (
                    ik_env.joint_lower[ik_env.arm1_qposadr],
                    ik_env.joint_lower[ik_env.arm2_qposadr],
                )
            ),
            dtype=torch.float32,
            device=device,
        )
        self.upper = torch.as_tensor(
            np.stack(
                (
                    ik_env.joint_upper[ik_env.arm1_qposadr],
                    ik_env.joint_upper[ik_env.arm2_qposadr],
                )
            ),
            dtype=torch.float32,
            device=device,
        )
        self.posture_seed = torch.as_tensor(
            np.stack((LEFT_ARM_SEED, RIGHT_ARM_SEED)),
            dtype=torch.float32,
            device=device,
        )
        self.target_rot = torch.as_tensor(
            self.config.target_rot, dtype=torch.float32, device=device
        )
        self.posture_eye = torch.eye(7, dtype=torch.float32, device=device).expand(
            2, -1, -1
        )
        self.damping_eye = torch.eye(13, dtype=torch.float32, device=device)

    def initial_seed(self) -> torch.Tensor:
        seed = torch.cat((self.posture_seed[0], self.posture_seed[1]))
        return seed.unsqueeze(0).expand(self.batch_size, -1).clone()

    def solve(
        self,
        qpos: np.ndarray | torch.Tensor,
        desired_targets: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        q = (
            torch.as_tensor(qpos, dtype=torch.float32, device=self.device)
            .reshape(self.batch_size, 2, 7)
            .clone()
        )
        task_targets = torch.as_tensor(
            desired_targets, dtype=torch.float32, device=self.device
        )
        target_pos = torch.stack(
            (
                torch.stack(
                    (task_targets[:, 0], 0.5 * task_targets[:, 2], task_targets[:, 1]),
                    dim=-1,
                ),
                torch.stack(
                    (task_targets[:, 0], -0.5 * task_targets[:, 2], task_targets[:, 1]),
                    dim=-1,
                ),
            ),
            dim=1,
        )
        active = torch.ones((self.batch_size, 2), dtype=torch.bool, device=self.device)

        for iteration in range(self.config.max_iterations):
            self.sim.data.qpos[:] = q.reshape(self.batch_size, -1)
            self.sim.data.qvel.zero_()
            self.sim.forward()

            current_pos = self.sim.data.site_xpos[:, self.site_ids]
            current_rot = self.sim.data.site_xmat[:, self.site_ids]
            pos_err = target_pos - current_pos
            target_rot = self.target_rot.expand_as(current_rot)
            rot_err = 0.5 * sum(
                torch.cross(current_rot[..., :, axis], target_rot[..., :, axis], dim=-1)
                for axis in range(3)
            )
            converged = (torch.linalg.vector_norm(pos_err, dim=-1) < 2e-4) & (
                torch.linalg.vector_norm(rot_err, dim=-1) < 2e-3
            )
            active &= ~converged
            if iteration % 5 == 4 and not bool(torch.any(active)):
                break

            cdof = self.sim.data.cdof[:, self.dof_indices]
            angular = cdof[..., :3]
            linear = cdof[..., 3:]
            offset = current_pos - self.sim.data.subtree_com[:, self.root_body_ids]
            jac_pos = (
                linear + torch.cross(angular, offset.unsqueeze(-2), dim=-1)
            ).transpose(-1, -2)
            jac_rot = angular.transpose(-1, -2)
            jacobian = torch.cat(
                (
                    self.config.position_weight * jac_pos,
                    self.config.orientation_weight * jac_rot,
                    self.config.posture_weight
                    * self.posture_eye.unsqueeze(0).expand(self.batch_size, -1, -1, -1),
                ),
                dim=-2,
            )
            error = torch.cat(
                (
                    self.config.position_weight * pos_err,
                    self.config.orientation_weight * rot_err,
                    self.config.posture_weight * (self.posture_seed.unsqueeze(0) - q),
                ),
                dim=-1,
            )
            lhs = (
                jacobian @ jacobian.transpose(-1, -2)
                + self.config.damping * self.damping_eye
            )
            delta = jacobian.transpose(-1, -2) @ torch.linalg.solve(
                lhs, error.unsqueeze(-1)
            )
            candidate = torch.clamp(
                q + self.config.step_scale * delta.squeeze(-1),
                min=self.lower,
                max=self.upper,
            )
            q = torch.where(active.unsqueeze(-1), candidate, q)

        return q.reshape(self.batch_size, -1)


def solve_initial_states(
    ik_env: LabEnv, initial_targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    qpos = np.empty((len(initial_targets), ik_env.model.nq), dtype=np.float32)
    controls = np.empty((len(initial_targets), ik_env.model.nu), dtype=np.float32)
    for env_idx, target in enumerate(initial_targets):
        ik_env.reset(TaskState.from_array(target))
        qpos[env_idx] = ik_env.data.qpos
        controls[env_idx] = ik_env.data.ctrl
    return qpos, controls


def solve_initial_states_cuda(
    ik_solver: BatchedCudaIKSolver, initial_targets: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor]:
    controls = ik_solver.solve(ik_solver.initial_seed(), initial_targets)
    return controls, controls


def solve_joint_targets(
    ik_env: LabEnv,
    qpos: np.ndarray,
    desired_targets: np.ndarray,
) -> np.ndarray:
    controls = np.empty((len(qpos), ik_env.model.nu), dtype=np.float32)
    for env_idx, (joint_positions, desired) in enumerate(
        zip(qpos, desired_targets, strict=True)
    ):
        mujoco.mj_resetData(ik_env.model, ik_env.data)
        ik_env.data.qpos[:] = joint_positions
        mujoco.mj_forward(ik_env.model, ik_env.data)
        ik_env.set_task_target(TaskState.from_array(desired))
        controls[env_idx] = ik_env.joint_controller.target
    return controls


def collect_trajectory_batch(
    *,
    args: argparse.Namespace,
    ik_env: LabEnv,
    simulation: BatchedRopeSimulation,
    cuda_ik_solver: BatchedCudaIKSolver | None,
    trajectory_seeds: np.ndarray,
    control_timestep: float,
) -> list[tuple[dict[str, np.ndarray], bool, bool]]:
    batch_size = simulation.batch_size
    policies = [
        make_policy(args, ik_env.task_bounds, seed=int(seed))
        for seed in trajectory_seeds
    ]
    durations = np.asarray(
        [policy_duration(policy) for policy in policies], dtype=np.float32
    )
    final_goals = np.stack([policy_goal_state(policy) for policy in policies])
    current_targets = np.stack(
        [policy.sample(0.0).as_array() for policy in policies]
    ).astype(np.float32)
    if cuda_ik_solver is None:
        qpos, controls = solve_initial_states(ik_env, current_targets)
    else:
        qpos, controls = solve_initial_states_cuda(cuda_ik_solver, current_targets)
    simulation.reset(qpos, controls)

    sample = simulation.extract_step_info(
        current_targets, np.zeros(batch_size, dtype=np.float32)
    )
    arrays = {
        key: np.empty(
            (batch_size, args.max_episode_steps + 1, *values.shape[1:]),
            dtype=values.dtype,
        )
        for key, values in sample.items()
    }
    actions = np.empty(
        (batch_size, args.max_episode_steps, ACTION_DIM), dtype=np.float32
    )
    frames = np.empty(
        (batch_size, args.max_episode_steps + 1, args.height, args.width, 3),
        dtype=np.uint8,
    )
    for key, values in sample.items():
        arrays[key][:, 0] = values
    frames[:, 0] = simulation.render()

    lengths = np.full(batch_size, args.max_episode_steps, dtype=np.int32)
    terminated = np.zeros(batch_size, dtype=np.bool_)
    active = np.ones(batch_size, dtype=np.bool_)
    for step_idx in range(args.max_episode_steps):
        sample_times = np.minimum((step_idx + 1) * control_timestep, durations)
        desired_targets = np.stack(
            [
                policy.sample(float(sample_time)).as_array()
                for policy, sample_time in zip(policies, sample_times, strict=True)
            ]
        ).astype(np.float32)
        actions[:, step_idx] = desired_targets - current_targets
        if cuda_ik_solver is None:
            controls = solve_joint_targets(
                ik_env, simulation.robot_qpos(), desired_targets
            )
        else:
            controls = cuda_ik_solver.solve(
                simulation.robot_qpos_tensor(), desired_targets
            )
        simulation.step(controls, args.control_decimation)
        current_targets = desired_targets
        sample = simulation.extract_step_info(current_targets, sample_times)
        for key, values in sample.items():
            arrays[key][:, step_idx + 1] = values
        frames[:, step_idx + 1] = simulation.render()

        at_goal = (
            np.linalg.norm(current_targets - final_goals, axis=1) <= args.goal_tolerance
        )
        done = active & at_goal
        lengths[done] = step_idx + 1
        terminated[done] = True
        active[done] = False
        if not np.any(active):
            break

    trajectories = []
    for env_idx, length in enumerate(lengths):
        data = {
            key: values[env_idx, : length + 1].copy() for key, values in arrays.items()
        }
        data["pixels"] = frames[env_idx, : length + 1]
        data["action"] = actions[env_idx, :length].copy()
        trajectories.append(
            (data, bool(terminated[env_idx]), not bool(terminated[env_idx]))
        )
    return trajectories


def validate_args(args: argparse.Namespace) -> None:
    if args.target_transitions is not None and args.target_transitions < 1:
        raise ValueError("--target-transitions must be positive when provided.")
    for name in (
        "num_trajectories",
        "max_episode_steps",
        "min_steps",
        "control_decimation",
        "batch_size",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.segment_duration <= 0.0:
        raise ValueError("--segment-duration must be positive.")
    if args.num_waypoints < 2:
        raise ValueError("--num-waypoints must be at least 2.")
    if args.midpoint_inflation_scale < 0.0:
        raise ValueError("--midpoint-inflation-scale cannot be negative.")
    if args.goal_tolerance < 0.0:
        raise ValueError("--goal-tolerance cannot be negative.")
    if args.rope_segments < 2:
        raise ValueError("--rope-segments must be at least 2.")
    if args.rope_length is not None and args.rope_length <= 0.0:
        raise ValueError("--rope-length must be positive when provided.")
    for name in (
        "rope_radius",
        "rope_link_mass",
        "rope_joint_armature",
        "rope_endpoint_timeconst",
        "rope_endpoint_damping_ratio",
    ):
        if getattr(args, name) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    for name in ("rope_joint_damping", "rope_joint_stiffness"):
        if getattr(args, name) < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    outdir = args.outdir.expanduser().resolve()
    video_dir = outdir / "videos"
    output_path = outdir / args.output_name
    outdir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output file already exists: {output_path}. Pass --overwrite to replace it."
        )
    if output_path.exists():
        output_path.unlink()

    ik_env = LabEnv()
    rope_config = ArticulatedRopeConfig(
        segments=args.rope_segments,
        length=args.rope_length,
        radius=args.rope_radius,
        link_mass=args.rope_link_mass,
        joint_damping=args.rope_joint_damping,
        joint_stiffness=args.rope_joint_stiffness,
        joint_armature=args.rope_joint_armature,
        endpoint_timeconst=args.rope_endpoint_timeconst,
        endpoint_damping_ratio=args.rope_endpoint_damping_ratio,
    )
    simulation = BatchedRopeSimulation(
        ik_env=ik_env,
        batch_size=args.batch_size,
        width=args.width,
        height=args.height,
        camera_name=args.camera,
        disable_shadows=args.disable_shadows,
        device=args.device,
        rope_config=rope_config,
    )
    cuda_ik_solver = None
    if args.ik_device == "cuda":
        ik_cuda_device = args.device if str(args.device).startswith("cuda") else "cuda"
        cuda_ik_solver = BatchedCudaIKSolver(
            ik_env, batch_size=args.batch_size, device=ik_cuda_device
        )
    control_timestep = simulation.timestep * args.control_decimation
    control_freq_hz = 1.0 / control_timestep
    obs_dim = (
        3 + simulation.robot_nq + simulation.robot_nv + simulation.model.nu + 3 + 3 + 1
    )
    compression = None if args.compression == "none" else args.compression
    step_counts: list[int] = []
    terminated_flags: list[bool] = []
    truncated_flags: list[bool] = []
    skipped_short = 0
    seed_offset = 0

    with h5py.File(output_path, "w") as h5:
        h5.attrs.update(
            {
                "format": "stable_worldmodel_hdf5",
                "source": "rope_manipulation/data/generate_data.py",
                "seed": args.seed,
                "video_dir": str(video_dir),
                "video_resolution": json.dumps([args.height, args.width]),
                "save_mp4": args.save_mp4,
                "camera": args.camera,
                "disable_shadows": args.disable_shadows,
                "mode": args.mode,
                "video_fps": control_freq_hz,
                "physics_timestep": simulation.timestep,
                "control_timestep": control_timestep,
                "control_decimation": args.control_decimation,
                "max_episode_steps": args.max_episode_steps,
                "segment_duration": args.segment_duration,
                "num_waypoints": args.num_waypoints,
                "goal_tolerance": args.goal_tolerance,
                "observation_dim": obs_dim,
                "action_dim": ACTION_DIM,
                "qpos_dim": simulation.robot_nq,
                "qvel_dim": simulation.robot_nv,
                "control_dim": simulation.model.nu,
                "task_bounds": json.dumps(
                    {
                        "reach": list(ik_env.task_bounds.reach),
                        "height": list(ik_env.task_bounds.height),
                        "width": list(ik_env.task_bounds.width),
                    }
                ),
                "nominal_task_state": json.dumps(
                    ik_env.nominal_state.as_array().tolist()
                ),
                "simulation_backend": "mjlab_mujoco_warp",
                "batch_size": args.batch_size,
                "device": args.device,
                "ik_device": args.ik_device,
                "rope_rendering": "non-colliding articulated capsule chain",
                "rope_segments": rope_config.segments,
                "rope_rest_length": simulation.rope_rest_length,
                "rope_radius": rope_config.radius,
                "rope_link_mass": rope_config.link_mass,
                "rope_joint_damping": rope_config.joint_damping,
                "rope_joint_stiffness": rope_config.joint_stiffness,
                "rope_joint_armature": rope_config.joint_armature,
                "rope_endpoint_timeconst": rope_config.endpoint_timeconst,
                "rope_endpoint_damping_ratio": rope_config.endpoint_damping_ratio,
            }
        )

        ep_len_ds = create_resizable_dataset(h5, "ep_len", (), np.int64)
        ep_offset_ds = create_resizable_dataset(h5, "ep_offset", (), np.int64)
        reward_ds = create_resizable_dataset(h5, "reward", (), np.float32)
        seed_ds = create_resizable_dataset(h5, "episode_seed", (), np.int64)
        terminated_ds = create_resizable_dataset(h5, "terminated", (), np.bool_)
        truncated_ds = create_resizable_dataset(h5, "truncated", (), np.bool_)
        pixels_ds = create_resizable_dataset(
            h5,
            "pixels",
            (args.height, args.width, 3),
            np.uint8,
            compression=compression,
            chunks=(1, args.height, args.width, 3),
        )
        action_ds = create_resizable_dataset(h5, "action", (ACTION_DIM,), np.float32)
        obs_ds = create_resizable_dataset(h5, "observation", (obs_dim,), np.float32)
        task_target_ds = create_resizable_dataset(h5, "task_target", (3,), np.float32)
        qpos_ds = create_resizable_dataset(
            h5, "qpos", (simulation.robot_nq,), np.float32
        )
        qvel_ds = create_resizable_dataset(
            h5, "qvel", (simulation.robot_nv,), np.float32
        )
        control_ds = create_resizable_dataset(
            h5, "control", (simulation.model.nu,), np.float32
        )
        left_ds = create_resizable_dataset(h5, "left_attachment_pos", (3,), np.float32)
        right_ds = create_resizable_dataset(
            h5, "right_attachment_pos", (3,), np.float32
        )
        rope_length_ds = create_resizable_dataset(h5, "rope_length", (1,), np.float32)
        time_ds = create_resizable_dataset(h5, "time", (1,), np.float32)
        episode_idx_ds = create_resizable_dataset(h5, "episode_idx", (), np.int64)
        step_idx_ds = create_resizable_dataset(h5, "step_idx", (), np.int64)

        progress_total = (
            args.target_transitions
            if args.target_transitions is not None
            else args.num_trajectories
        )
        progress_desc = (
            "Collecting transitions"
            if args.target_transitions is not None
            else "Collecting trajectories"
        )
        progress_unit = "step" if args.target_transitions is not None else "traj"
        with tqdm(
            total=progress_total, desc=progress_desc, unit=progress_unit
        ) as progress:
            while should_continue(
                args, len(step_counts), int(np.sum(step_counts, dtype=np.int64))
            ):
                seeds = np.arange(
                    args.seed + seed_offset,
                    args.seed + seed_offset + args.batch_size,
                    dtype=np.int64,
                )
                seed_offset += args.batch_size
                trajectories = collect_trajectory_batch(
                    args=args,
                    ik_env=ik_env,
                    simulation=simulation,
                    cuda_ik_solver=cuda_ik_solver,
                    trajectory_seeds=seeds,
                    control_timestep=control_timestep,
                )
                for trajectory_seed, (trajectory, terminated, truncated) in zip(
                    seeds, trajectories, strict=True
                ):
                    if not should_continue(
                        args, len(step_counts), int(np.sum(step_counts, dtype=np.int64))
                    ):
                        break
                    num_actions = int(trajectory["action"].shape[0])
                    if num_actions < args.min_steps:
                        skipped_short += 1
                        continue
                    episode_idx = len(step_counts)
                    if args.save_mp4:
                        imageio.mimwrite(
                            video_dir / f"trajectory_{episode_idx:07d}.mp4",
                            trajectory["pixels"],
                            fps=control_freq_hz,
                            quality=args.quality,
                            macro_block_size=1,
                        )

                    padded_actions = np.empty(
                        (num_actions + 1, ACTION_DIM), dtype=np.float32
                    )
                    padded_actions[:-1] = trajectory["action"]
                    padded_actions[-1] = np.nan
                    offset, _ = append_rows(pixels_ds, trajectory["pixels"])
                    append_rows(action_ds, padded_actions)
                    append_rows(obs_ds, trajectory["observation"])
                    append_rows(task_target_ds, trajectory["task_target"])
                    append_rows(qpos_ds, trajectory["qpos"])
                    append_rows(qvel_ds, trajectory["qvel"])
                    append_rows(control_ds, trajectory["control"])
                    append_rows(left_ds, trajectory["left_attachment_pos"])
                    append_rows(right_ds, trajectory["right_attachment_pos"])
                    append_rows(rope_length_ds, trajectory["rope_length"])
                    append_rows(time_ds, trajectory["time"])
                    append_rows(
                        episode_idx_ds,
                        np.full(num_actions + 1, episode_idx, dtype=np.int64),
                    )
                    append_rows(step_idx_ds, np.arange(num_actions + 1, dtype=np.int64))
                    append_rows(
                        ep_len_ds, np.asarray([num_actions + 1], dtype=np.int64)
                    )
                    append_rows(ep_offset_ds, np.asarray([offset], dtype=np.int64))
                    append_rows(reward_ds, np.asarray([0.0], dtype=np.float32))
                    append_rows(seed_ds, np.asarray([trajectory_seed], dtype=np.int64))
                    append_rows(terminated_ds, np.asarray([terminated], dtype=np.bool_))
                    append_rows(truncated_ds, np.asarray([truncated], dtype=np.bool_))
                    step_counts.append(num_actions)
                    terminated_flags.append(terminated)
                    truncated_flags.append(truncated)
                    progress.update(
                        num_actions if args.target_transitions is not None else 1
                    )
                    progress.set_postfix(
                        episodes=len(step_counts), transitions=int(np.sum(step_counts))
                    )

        ep_len = np.asarray(ep_len_ds[:], dtype=np.int64)
        h5.attrs["num_episodes"] = len(step_counts)
        h5.attrs["total_frames"] = int(pixels_ds.shape[0])
        h5.attrs["total_transitions"] = int(np.sum(step_counts, dtype=np.int64))
        h5.attrs["skipped_short_episodes"] = skipped_short
        h5.attrs["mean_reward"] = 0.0
        h5.attrs["mean_episode_steps"] = (
            float(np.mean(step_counts)) if step_counts else 0.0
        )
        h5.attrs["terminated_fraction"] = (
            float(np.mean(terminated_flags)) if terminated_flags else 0.0
        )
        h5.attrs["truncated_fraction"] = (
            float(np.mean(truncated_flags)) if truncated_flags else 0.0
        )
        h5.attrs["usable_train_windows_default"] = valid_training_windows(ep_len)

    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "video_dir": str(video_dir),
                "simulation_backend": "mjlab_mujoco_warp",
                "batch_size": args.batch_size,
                "ik_device": args.ik_device,
                "num_episodes": len(step_counts),
                "total_transitions": int(np.sum(step_counts, dtype=np.int64)),
                "mean_episode_steps": float(np.mean(step_counts))
                if step_counts
                else 0.0,
                "control_timestep": control_timestep,
                "control_fps": control_freq_hz,
                "mode": args.mode,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
