#!/usr/bin/env python3
"""Closed-loop GR00T VLA control of the G1 in MuJoCo.

This script connects to a GR00T policy server (REAL_G1 embodiment) via ZMQ,
streams head-camera frames + proprioception from MuJoCo, and applies the
predicted action chunk to the robot's upper body (waist + arms + hands).

The walker ONNX policy is kept running to balance the legs while VLA
controls the upper body.

Usage:
  # (1) Open a tunnel/server so 127.0.0.1:5555 reaches the GR00T server.
  #     See Isaac-GR00T/run_client.sh — it uses cloudflared.
  # (2) Run:
  uv run python vla_run.py \
      --prompt "pick the red cube and put into yellow container" \
      --host 127.0.0.1 --port 5555
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path
from typing import Any

import cv2
from PIL import Image
import msgpack_numpy as mnp
import mujoco
import numpy as np
import onnxruntime as ort
import zmq

SCRIPT_DIR = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# REAL_G1 embodiment mapping
# --------------------------------------------------------------------------- #
LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]
RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
WAIST_JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
# Order matches dataset (info.json feature names):
# left_hand:  kLeftHandThumb0/1/2, kLeftHandMiddle0/1, kLeftHandIndex0/1
# right_hand: kRightHandThumb0/1/2, kRightHandIndex0/1, kRightHandMiddle0/1
LEFT_HAND_JOINTS = [
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint",
]
RIGHT_HAND_JOINTS = [
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
]

# Dataset initial joint positions (from real-robot trajectory, same as teleoperate.py)
DATASET_INIT_QPOS = {
    "left_shoulder_pitch_joint":  -0.186601,
    "left_shoulder_roll_joint":    0.316981,
    "left_shoulder_yaw_joint":     0.287689,
    "left_elbow_joint":           -0.635120,
    "left_wrist_roll_joint":      -0.243937,
    "left_wrist_pitch_joint":      0.690306,
    "left_wrist_yaw_joint":       -0.015776,
    "right_shoulder_pitch_joint": -0.364927,
    "right_shoulder_roll_joint":  -0.258941,
    "right_shoulder_yaw_joint":   -0.201870,
    "right_elbow_joint":          -0.514582,
    "right_wrist_roll_joint":      0.261457,
    "right_wrist_pitch_joint":     0.718800,
    "right_wrist_yaw_joint":       0.073139,
    "left_hand_thumb_0_joint":    -0.427740,
    "left_hand_thumb_1_joint":     1.028506,
    "left_hand_thumb_2_joint":     0.184237,
    "left_hand_middle_0_joint":    0.175383,
    "left_hand_middle_1_joint":   -0.047019,
    "left_hand_index_0_joint":     0.184752,
    "left_hand_index_1_joint":    -0.015129,
    "right_hand_thumb_0_joint":   -0.404762,
    "right_hand_thumb_1_joint":   -1.038415,
    "right_hand_thumb_2_joint":   -0.326415,
    "right_hand_index_0_joint":   -0.189341,
    "right_hand_index_1_joint":    0.012102,
    "right_hand_middle_0_joint":  -0.184233,
    "right_hand_middle_1_joint":   0.011182,
}

LANGUAGE_KEY = "annotation.human.task_description"
VIDEO_KEYS = ["cam_left_high", "cam_right_high", "cam_left_wrist", "cam_right_wrist"]  # matches finetuning dataset
ACTION_HORIZON = 40
EXEC_HORIZON = 16  # how many predicted actions to execute before re-querying
IMG_H = 480
IMG_W = 640


# --------------------------------------------------------------------------- #
# ZMQ client (minimal — mirrors gr00t/policy/server_client.py protocol)
# --------------------------------------------------------------------------- #
class PolicyClient:
    def __init__(self, host: str, port: int, timeout_ms: int = 60000):
        self.ctx = zmq.Context()
        self.host, self.port, self.timeout = host, port, timeout_ms
        self._connect()

    def _connect(self):
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout)
        self.sock.setsockopt(zmq.SNDTIMEO, self.timeout)
        self.sock.connect(f"tcp://{self.host}:{self.port}")

    def _call(self, endpoint: str, data: dict | None = None, requires_input: bool = True):
        req = {"endpoint": endpoint}
        if requires_input:
            req["data"] = data or {}
        try:
            self.sock.send(mnp.packb(req))
            msg = self.sock.recv()
        except zmq.error.Again:
            self.sock.close()
            self._connect()
            raise
        resp = mnp.unpackb(msg, raw=False)
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"Server error: {resp['error']}")
        return resp

    def ping(self) -> bool:
        try:
            self._call("ping", requires_input=False)
            return True
        except Exception:
            return False

    def get_action(self, observation: dict[str, Any]) -> dict[str, np.ndarray]:
        resp = self._call("get_action", {"observation": observation, "options": None})
        action, _info = resp[0], resp[1] if len(resp) > 1 else {}
        return action


# --------------------------------------------------------------------------- #
# ONNX walker (CPU)
# --------------------------------------------------------------------------- #
class ONNXPolicy:
    def __init__(self, path: str):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        self.iname = self.sess.get_inputs()[0].name
        self.oname = self.sess.get_outputs()[0].name

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        if obs.ndim == 1:
            obs = obs[None]
        return self.sess.run([self.oname], {self.iname: obs.astype(np.float32)})[0][0]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def quat_apply_inv(quat, vec):
    w, xyz = quat[0], quat[1:4]
    t = np.cross(xyz, vec) * 2
    return vec - w * t + np.cross(xyz, t)


def quat_to_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


def quat_mul(q1, q2):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, q1, q2)
    return out


def quat_inv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def eef_9d(site_xpos_world, site_xmat_world, base_pos, base_quat):
    """Return [pos(3) in base frame, rot6d(6) = first 2 rows of R_base_site].

    GR00T convention (pose.py::_matrix_to_rot6d): first two ROWS of the
    relative rotation matrix, flattened in C-order (row-major).
    """
    pos_rel = quat_apply_inv(base_quat, site_xpos_world - base_pos)
    # site->world rotation matrix (MuJoCo row-major flat -> NumPy C-order)
    R_site_world = site_xmat_world.reshape(3, 3)
    # base->world rotation matrix from quat
    R_base = quat_to_mat(base_quat)
    # relative rotation base->site = R_base^{-1} @ R_site_world
    R_rel = R_base.T @ R_site_world
    rot6d = R_rel[:2, :].flatten()  # first 2 rows, C-order
    return np.concatenate([pos_rel, rot6d]).astype(np.float32)


def mat_to_quat(mat: np.ndarray) -> np.ndarray:
    """Convert a 3×3 rotation matrix to a MuJoCo quaternion (w, x, y, z)."""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, mat.flatten())
    return q


def set_armature(model, joint_names):
    A_5020, A_7520_14, A_7520_22, A_4010, A_2x = 0.00360972, 0.01017752, 0.02510192, 0.00425, 0.00721945
    for i, name in enumerate(joint_names):
        dof = 6 + i
        if "elbow" in name or "shoulder" in name or "wrist_roll" in name:
            model.dof_armature[dof] = A_5020
        elif "hip_pitch" in name or "hip_yaw" in name or name == "waist_yaw_joint":
            model.dof_armature[dof] = A_7520_14
        elif "hip_roll" in name or "knee" in name:
            model.dof_armature[dof] = A_7520_22
        elif "wrist_pitch" in name or "wrist_yaw" in name:
            model.dof_armature[dof] = A_4010
        elif "ankle" in name or name in ("waist_pitch_joint", "waist_roll_joint"):
            model.dof_armature[dof] = A_2x
        else:
            model.dof_armature[dof] = A_5020


# --------------------------------------------------------------------------- #
# Grip assist
# --------------------------------------------------------------------------- #
class GripAssist:
    """Kinematically attach a grippable object to the palm when the hand is close.

    When the palm site is within GRIP_DIST of a cube, the cube's freejoint is
    overridden every step so the cube rigidly follows the hand.  Once latched,
    the grip releases only when the VLA returns the fingers toward the open
    (dataset-init) pose — detected by a low mean deviation of finger-joint qpos
    from the init values.
    """

    GRIP_DIST   = 0.08  # metres — latch when palm is within this radius of cube
    # Raw closure thresholds based on actual joint ranges from g1.xml:
    #   right index_0: range [0, 1.5708]  →  0 = open, positive = curling
    #   left  index_0: range [-1.5708, 0] →  0 = open, negative = curling
    # closure = signed_value * closing_direction, so it is always ≥ 0 when curling.
    GRIP_THRESH = 0.30  # closure above this → hand is gripping
    OPEN_THRESH = 0.08  # closure below this → hand is open  (must be < GRIP_THRESH)
    MIN_GRIP_STEPS = 20 # debounce: keep latched for at least this many steps

    # Closing direction: +1 means positive qpos = curled (right), -1 means negative = curled (left).
    _CLOSE_DIR = {"right": +1.0, "left": -1.0}
    # Primary index_0 joint per hand — the joint that moves most during a grip.
    _INDEX0_JOINT = {"right": "right_hand_index_0_joint", "left": "left_hand_index_0_joint"}

    def __init__(self, model, data):
        self.model = model
        self.data  = data

        # Grippable objects: body id + freejoint qpos/dof start addresses.
        _obj_joints = {"red_cube": "red_cube_joint", "green_cube": "green_cube_joint"}
        self._obj_body: dict[str, int] = {}
        self._obj_qpos: dict[str, int] = {}
        self._obj_dof:  dict[str, int] = {}
        for bname, jname in _obj_joints.items():
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,  bname)
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if bid >= 0 and jid >= 0:
                self._obj_body[bname] = bid
                self._obj_qpos[bname] = int(model.jnt_qposadr[jid])
                self._obj_dof [bname] = int(model.jnt_dofadr [jid])

        # Palm site ids.
        self._palm = {
            "left":  mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_palm"),
            "right": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_palm"),
        }

        # index_0 qpos address per hand — used as the primary grip/release sensor.
        # We read the raw joint value and multiply by _CLOSE_DIR to get a
        # non-negative "closure" score (0 = fully open, ~1.57 = fully closed).
        self._index0_addr: dict[str, int] = {}
        for hand, jname in self._INDEX0_JOINT.items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid >= 0:
                self._index0_addr[hand] = int(model.jnt_qposadr[jid])

        # Active grip state: hand → (obj_name, rel_pos_in_palm_frame, rel_quat_palm2obj)
        self._grip:       dict[str, tuple | None] = {"left": None, "right": None}
        self._grip_steps: dict[str, int]          = {"left": 0,    "right": 0}
        self._log_counter = 0  # for periodic debug prints

    # ------------------------------------------------------------------ #
    def _palm_pose(self, hand: str) -> tuple[np.ndarray, np.ndarray]:
        sid = self._palm[hand]
        pos = self.data.site_xpos[sid].copy()
        mat = self.data.site_xmat[sid].reshape(3, 3).copy()
        return pos, mat

    def _index_closure(self, hand: str) -> float:
        """Raw closure of index_0: joint_value * closing_direction (≥0 when curling)."""
        addr = self._index0_addr.get(hand)
        if addr is None:
            return 0.0
        return float(self.data.qpos[addr]) * self._CLOSE_DIR[hand]

    def update(self):
        """Call once per simulation step, after mj_step."""
        self._log_counter += 1
        latched: set[str] = {
            info[0] for info in self._grip.values() if info is not None
        }

        for hand in ("left", "right"):
            palm_pos, palm_mat = self._palm_pose(hand)
            palm_quat = mat_to_quat(palm_mat)
            idx_closure = self._index_closure(hand)

            # Periodic debug: raw joint value + closure so thresholds can be tuned.
            if self._log_counter % 200 == 0:
                addr = self._index0_addr.get(hand)
                raw = float(self.data.qpos[addr]) if addr is not None else float("nan")
                state = "GRIP" if self._grip[hand] else "open"
                print(f"[grip] {hand} index_0 raw={raw:+.3f}  closure={idx_closure:.3f}"
                      f"  (latch>{self.GRIP_THRESH}, release<{self.OPEN_THRESH})"
                      f"  [{state}]")

            if self._grip[hand] is not None:
                obj_name, rel_pos, rel_quat = self._grip[hand]
                self._grip_steps[hand] += 1

                # Teleport cube so it rigidly follows the palm.
                new_pos  = palm_pos + palm_mat @ rel_pos
                new_quat = quat_mul(palm_quat, rel_quat)
                addr = self._obj_qpos[obj_name]
                self.data.qpos[addr:addr + 3] = new_pos
                self.data.qpos[addr + 3:addr + 7] = new_quat
                self.data.qvel[self._obj_dof[obj_name]:self._obj_dof[obj_name] + 6] = 0.0

                # Release when index finger returns toward open pose.
                if (self._grip_steps[hand] > self.MIN_GRIP_STEPS
                        and idx_closure < self.OPEN_THRESH):
                    # Drop the object below the palm so it clears the thumb geometry,
                    # then give it a downward velocity impulse to escape any remaining
                    # finger contact before handing control back to physics.
                    obj_addr = self._obj_qpos[obj_name]
                    obj_dof  = self._obj_dof[obj_name]
                    self.data.qpos[obj_addr + 2] -= 0.05          # 5 cm below current pos
                    self.data.qvel[obj_dof:obj_dof + 3]     = [0.0, 0.0, -0.5]  # drop velocity
                    self.data.qvel[obj_dof + 3:obj_dof + 6] = 0.0
                    print(f"[grip] {hand} released {obj_name} "
                          f"(index closure={idx_closure:.3f} < OPEN_THRESH={self.OPEN_THRESH})")
                    self._grip[hand] = None
                    self._grip_steps[hand] = 0
                    latched.discard(obj_name)

            else:
                # Latch only when index finger is closing AND palm is close to cube.
                if idx_closure < self.GRIP_THRESH:
                    continue  # index finger is open — don't attach anything
                for obj_name, body_id in self._obj_body.items():
                    if obj_name in latched:
                        continue
                    obj_pos  = self.data.xpos[body_id].copy()
                    obj_quat = self.data.xquat[body_id].copy()
                    dist = np.linalg.norm(palm_pos - obj_pos)
                    if dist < self.GRIP_DIST:
                        rel_pos  = palm_mat.T @ (obj_pos - palm_pos)
                        rel_quat = quat_mul(quat_inv(palm_quat), obj_quat)
                        self._grip[hand]       = (obj_name, rel_pos, rel_quat)
                        self._grip_steps[hand] = 0
                        latched.add(obj_name)
                        print(f"[grip] {hand} latched {obj_name} "
                              f"(dist={dist:.3f} m, index closure={idx_closure:.3f})")
                        break


# --------------------------------------------------------------------------- #
# Main controller
# --------------------------------------------------------------------------- #
class VLAController:
    def __init__(self, model, data, walker, config, client: PolicyClient, prompt: str,
                 freeze_non_arm: bool = True, apply_hands: bool = True,
                 save_obs_dir: str | None = None):
        self.model, self.data = model, data
        self.walker, self.client, self.prompt = walker, client, prompt
        self.config = config
        # When True, only left_arm + right_arm (+ optionally hands) are driven by VLA;
        # all other body joints are pinned to default and the floating base is frozen.
        self.freeze_non_arm = freeze_non_arm
        self.apply_hands = apply_hands

        # Initial base pose snapshot for freezing.
        self._base_qpos0 = None  # filled in main after pose init
        self.joint_names = config["joint_names"]
        self.num_joints = len(self.joint_names)

        # qvel indices (actual MuJoCo dof addresses, not a naive 6+index guess)
        self.qvel_idx: dict[str, int] = {}
        for n in self.joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
            if jid >= 0:
                self.qvel_idx[n] = int(model.jnt_dofadr[jid])
        self.default = np.zeros(self.num_joints, dtype=np.float32)
        for n, v in config["default_joint_pos"].items():
            if n in self.joint_names:
                self.default[self.joint_names.index(n)] = v
        # Override arm defaults to match dataset initial pose (teleoperate.py convention).
        _arm_defaults = {n: v for n, v in DATASET_INIT_QPOS.items()
                         if n in (LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS)}
        for n, v in _arm_defaults.items():
            if n in self.joint_names:
                self.default[self.joint_names.index(n)] = v
        self.action_scales = np.array(
            [config["action_scales"][n] for n in self.joint_names], dtype=np.float32
        )

        # Cache: site ids, actuator ids
        self.left_palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_palm")
        self.right_palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_palm")

        self.body_act = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
                         for n in self.joint_names}
        self.hand_act = {}
        for n in LEFT_HAND_JOINTS + RIGHT_HAND_JOINTS:
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            if aid >= 0:
                self.hand_act[n] = aid

        # Look up qpos address for *every* joint we care about (body + hands).
        # Hand joints aren't in model_config.joint_names so use the MJCF model.
        self._qpos_addr: dict[str, int] = {}
        for n in (self.joint_names + LEFT_HAND_JOINTS + RIGHT_HAND_JOINTS):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
            if jid >= 0:
                self._qpos_addr[n] = int(model.jnt_qposadr[jid])

        # Walker state
        self.last_action = np.zeros(self.num_joints, dtype=np.float32)
        self.cmd = np.zeros(3, dtype=np.float32)  # zero vel — stationary

        # Action chunk + per-chunk reference state (the "current state" at inference time)
        self.chunk: dict[str, np.ndarray] | None = None
        self.chunk_step = 0
        self.chunk_ref: dict[str, np.ndarray] | None = None

        # Async inference — background thread so the sim loop never blocks.
        self._infer_lock    = threading.Lock()
        self._pending_chunk: tuple | None = None   # (chunk_dict, elapsed_s)
        self._infer_thread: threading.Thread | None = None

        # Renderer for camera observations (separate from preview)
        self.obs_renderer = mujoco.Renderer(model, IMG_H, IMG_W)

        # Optional: save every server observation to disk for debugging
        self.save_obs_dir = Path(save_obs_dir) if save_obs_dir else None
        self._obs_counter = 0
        if self.save_obs_dir:
            self.save_obs_dir.mkdir(parents=True, exist_ok=True)
            print(f"[VLA] Saving observations to {self.save_obs_dir}")

    # ------- proprio readers -------
    def _qpos(self, name: str) -> float:
        return float(self.data.qpos[self._qpos_addr[name]])

    def _joint_array(self, names: list[str]) -> np.ndarray:
        return np.array([self._qpos(n) for n in names], dtype=np.float32)

    def _base_pose(self):
        return self.data.qpos[:3].copy(), self.data.qpos[3:7].copy()

    # ------- walker (legs balance) -------
    def _walker_obs(self):
        base_pos, base_quat = self._base_pose()
        lin_world = self.data.qvel[:3].copy()
        ang_body = self.data.qvel[3:6].copy()
        lin_body = quat_apply_inv(base_quat, lin_world)
        proj_g = quat_apply_inv(base_quat, np.array([0.0, 0.0, -1.0]))
        jpos = np.array([self._qpos(n) - self.default[i]
                         for i, n in enumerate(self.joint_names)], dtype=np.float32)
        jvel = np.array([self.data.qvel[self.qvel_idx[n]] for n in self.joint_names],
                        dtype=np.float32)
        return np.concatenate([lin_body, ang_body, proj_g, jpos, jvel,
                               self.last_action, self.cmd]).astype(np.float32)

    # ------- VLA observation -------
    def _render_cameras(self) -> dict[str, np.ndarray]:
        frames = {}
        for cam in VIDEO_KEYS:
            self.obs_renderer.update_scene(self.data, camera=cam)
            frames[cam] = self.obs_renderer.render().copy()  # (H, W, 3) uint8
        return frames

    def _build_observation(self) -> dict[str, Any]:
        # State keys must match finetuning modality (G1_Dex3_ObjectPlacement_Dataset):
        # left_arm, right_arm, left_hand, right_hand only.
        state = {
            "left_arm": self._joint_array(LEFT_ARM_JOINTS),
            "right_arm": self._joint_array(RIGHT_ARM_JOINTS),
            "left_hand": self._joint_array(LEFT_HAND_JOINTS),
            "right_hand": self._joint_array(RIGHT_HAND_JOINTS),
        }
        # Wrap each (D,) -> (1 batch, 1 timestep, D)
        state_batched = {k: v[None, None, :] for k, v in state.items()}

        # Video: 1 frame per camera -> (1, 1, H, W, 3) — dataset horizon=1
        frames = self._render_cameras()
        video = {k: v[None, None] for k, v in frames.items()}

        return {
            "video": video,
            "state": state_batched,
            "language": {LANGUAGE_KEY: [[self.prompt]]},
        }, state  # also return current state for delta integration

    # ------- targets -------
    def _apply_chunk_targets(self, target_body: np.ndarray) -> dict[int, float]:
        """Compute joint targets from the current chunk_step of self.chunk.

        The server applies relative→absolute conversion before returning, so
        chunk values are already absolute joint targets (radians).  Write them
        directly as ctrl targets — DO NOT add current state again.

        Returns a dict {actuator_id: ctrl_value} for hand actuators.
        target_body is modified in-place for body actuators.
        """
        c = self.chunk
        t = min(self.chunk_step, c["right_arm"].shape[0] - 1)

        # Per-joint clamp bounds (task-relevant subset of training q01/q99).
        # Prevents the model from driving arms to extreme out-of-task poses.
        _JOINT_CLAMP = {
            # left arm: [lo, hi]  (training q01/q99 tightened for pick-place task)
            "left_shoulder_pitch_joint": (-0.8, 0.5),
            "left_shoulder_roll_joint":  (0.1, 0.8),
            "left_shoulder_yaw_joint":   (-0.9, 0.6),
            "left_elbow_joint":          (-0.1, 1.5),   # don't hyper-extend
            "left_wrist_roll_joint":     (-1.3, 1.3),
            "left_wrist_pitch_joint":    (-0.9, 0.9),
            "left_wrist_yaw_joint":      (-0.9, 0.9),
            # right arm
            "right_shoulder_pitch_joint":(-0.8, 0.5),
            "right_shoulder_roll_joint": (-0.8, -0.1),
            "right_shoulder_yaw_joint":  (-0.7, 1.2),
            "right_elbow_joint":         (-0.1, 1.5),   # don't hyper-extend
            "right_wrist_roll_joint":    (-1.4, 1.5),
            "right_wrist_pitch_joint":   (-0.9, 0.9),
            "right_wrist_yaw_joint":     (-1.0, 0.9),
            # waist
            "waist_yaw_joint":           (-1.0, 1.0),
            "waist_roll_joint":          (-0.4, 0.4),
            "waist_pitch_joint":         (-0.4, 0.4),
        }

        def apply_joint_group(names: list[str], key: str):
            vals = c[key][t]  # (D,) absolute joint targets
            for i, n in enumerate(names):
                if n in self.joint_names:
                    idx = self.joint_names.index(n)
                    v = float(vals[i])
                    if n in _JOINT_CLAMP:
                        lo, hi = _JOINT_CLAMP[n]
                        v = max(lo, min(hi, v))
                    target_body[idx] = v

        apply_joint_group(LEFT_ARM_JOINTS, "left_arm")
        apply_joint_group(RIGHT_ARM_JOINTS, "right_arm")

        hand_ctrl: dict[int, float] = {}
        if self.apply_hands:
            hand_groups = [(LEFT_HAND_JOINTS, "left_hand"), (RIGHT_HAND_JOINTS, "right_hand")]
            for names, key in hand_groups:
                vals = c[key][t]
                for i, n in enumerate(names):
                    if n in self.hand_act:
                        hand_ctrl[self.hand_act[n]] = float(vals[i])
        return hand_ctrl

    # ------- async VLA inference (runs in background thread) -------
    def _run_inference(self, obs_dict: dict, cur_state: dict):
        try:
            t0 = time.time()
            action_chunk = self.client.get_action(obs_dict)
            dt = time.time() - t0
            chunk = {k: np.asarray(v)[0] for k, v in action_chunk.items()}
            with self._infer_lock:
                self._pending_chunk = (chunk, cur_state, dt)
        except Exception as e:
            print(f"[VLA] async inference failed: {e!r}")

    # ------- main step (called at 50 Hz control rate) -------
    def step(self) -> tuple[np.ndarray, dict[int, float]]:
        if self.freeze_non_arm:
            target_body = self.default.copy()
        else:
            obs = self._walker_obs()
            action = self.walker(obs)
            target_body = self.default + action * self.action_scales
            self.last_action = action.copy()

        # 2) Swap in completed async chunk if ready.
        with self._infer_lock:
            if self._pending_chunk is not None:
                chunk, cur_state, dt = self._pending_chunk
                self._pending_chunk = None
                self.chunk      = chunk
                self.chunk_ref  = cur_state
                self.chunk_step = 0
                T   = next(iter(chunk.values())).shape[0]
                la0 = chunk["left_arm"][0]
                ra0 = chunk["right_arm"][0]
                print(f"[VLA] new chunk ({dt:.2f}s)  horizon={T}  keys={list(chunk.keys())}")
                print(f"[VLA] state  left_arm ={np.round(cur_state['left_arm'],  3).tolist()}")
                print(f"[VLA] action left_arm[0]={np.round(la0, 3).tolist()}")
                print(f"[VLA] state  right_arm={np.round(cur_state['right_arm'], 3).tolist()}")
                print(f"[VLA] action right_arm[0]={np.round(ra0, 3).tolist()}")

        # 3) Kick off next inference when the current chunk is nearly exhausted,
        #    but only if no request is already in flight.
        infer_running = self._infer_thread is not None and self._infer_thread.is_alive()
        if not infer_running and (self.chunk is None or self.chunk_step >= EXEC_HORIZON):
            obs_dict, cur_state = self._build_observation()
            if self.save_obs_dir:
                self._save_observation(obs_dict, cur_state)
            self._infer_thread = threading.Thread(
                target=self._run_inference, args=(obs_dict, cur_state), daemon=True
            )
            self._infer_thread.start()

        # 4) Apply current chunk to override upper body (sim continues unblocked).
        hand_ctrl: dict[int, float] = {}
        if self.chunk is not None:
            hand_ctrl = self._apply_chunk_targets(target_body)
            self.chunk_step += 1

        return target_body, hand_ctrl

    def _save_observation(self, obs_dict: dict[str, Any], cur_state: dict[str, np.ndarray]):
        """Save observation to disk as .npz + .json + .png frames for debugging."""
        idx = self._obs_counter
        self._obs_counter += 1
        prefix = self.save_obs_dir / f"obs_{idx:04d}"

        # --- images ---
        for cam_name, video in obs_dict["video"].items():
            cur_frame = video[0, 0]
            Image.fromarray(cur_frame).save(f"{prefix}_frame_{cam_name}.png")

        # --- JSON (human-readable) ---
        json_data = {
            "index": idx,
            "prompt": self.prompt,
            "state": {k: v.tolist() for k, v in cur_state.items()},
        }
        with open(f"{prefix}.json", "w") as f:
            json.dump(json_data, f, indent=2)

        # --- NPZ (exact arrays, load with np.load) ---
        npz_data: dict[str, Any] = {"prompt": self.prompt}
        for k, v in cur_state.items():
            npz_data[f"state_{k}"] = v
        for k, v in obs_dict["video"].items():
            npz_data[f"video_{k}"] = v
        np.savez(f"{prefix}.npz", **npz_data)

    def write_ctrl(self, target_body: np.ndarray, hand_ctrl: dict[int, float]):
        for i, n in enumerate(self.joint_names):
            aid = self.body_act[n]
            if aid >= 0:
                self.data.ctrl[aid] = target_body[i]
        for aid, v in hand_ctrl.items():
            self.data.ctrl[aid] = v


# --------------------------------------------------------------------------- #
# Object randomization
# --------------------------------------------------------------------------- #
_OBJ_SPAWN = {
    # joint_name        half-height  upright quat [w,x,y,z]
    "red_cube_joint":   (0.035, [1.0, 0.0, 0.0, 0.0]),   # cylinder, z-axis up
    "green_cube_joint": (0.040, [1.0, 0.0, 0.0, 0.0]),   # box, z-axis up
}
_TABLE_TOP_Z    = 0.84          # table surface (world frame)
_YELLOW_CENTER  = np.array([-0.20, 0.00])
_SPAWN_X        = (-0.35, -0.05)   # reachable x band
_SPAWN_Y        = (-0.35,  0.35)   # reachable y band
_MIN_YELLOW     = 0.16          # keep this far from yellow-box centre
_MIN_OBJECTS    = 0.12          # keep objects this far from each other


def randomize_objects(model, data, seed=None):
    """Place grippable objects at random reachable positions on the table."""
    rng = np.random.default_rng(seed)
    placed: list[np.ndarray] = []

    for jname, (half_z, quat_init) in _OBJ_SPAWN.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            print(f"[spawn] joint {jname!r} not found — skipping")
            continue
        addr = int(model.jnt_qposadr[jid])
        dof  = int(model.jnt_dofadr[jid])

        for attempt in range(200):
            x = rng.uniform(*_SPAWN_X)
            y = rng.uniform(*_SPAWN_Y)
            xy = np.array([x, y])

            if np.linalg.norm(xy - _YELLOW_CENTER) < _MIN_YELLOW:
                continue
            if any(np.linalg.norm(xy - p) < _MIN_OBJECTS for p in placed):
                continue

            z = _TABLE_TOP_Z + half_z + 0.003   # tiny clearance above table
            yaw = rng.uniform(0, 2 * np.pi)
            qw, qz = np.cos(yaw / 2), np.sin(yaw / 2)

            data.qpos[addr:addr + 3]     = [x, y, z]
            data.qpos[addr + 3:addr + 7] = [qw, 0.0, 0.0, qz]
            data.qvel[dof:dof + 6]       = 0.0
            placed.append(xy)
            print(f"[spawn] {jname}: x={x:.3f}  y={y:.3f}  yaw={np.degrees(yaw):.1f}°"
                  f"  (attempt {attempt+1})")
            break
        else:
            print(f"[spawn] WARNING: could not place {jname} after 200 attempts")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="pick the red cube and put into yellow container")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--no-preview", action="store_true",
                    help="Disable OpenCV head-cam preview window")
    ap.add_argument("--use-walker", action="store_true",
                    help="Run walker policy + apply waist/hand actions (full body). "
                         "Default: freeze base and non-arm joints, arms only.")
    ap.add_argument("--no-hands", action="store_true",
                    help="Don't apply VLA hand (finger) deltas.")
    ap.add_argument("--save-obs-dir", default=None,
                    help="Directory to save every observation sent to the server "
                         "as .npz + .json + .png for debugging.")
    ap.add_argument("--no-grip-assist", action="store_true",
                    help="Disable grip assist (kinematic weld when palm is near a cube).")
    args = ap.parse_args()

    # Load config + scene
    with open(SCRIPT_DIR / "model_config.json") as f:
        config = json.load(f)
    joint_names = config["joint_names"]

    print(f"[init] Loading scene {SCRIPT_DIR / 'scene.xml'}")
    model = mujoco.MjModel.from_xml_path(str(SCRIPT_DIR / "scene.xml"))
    model.opt.timestep = 0.005
    set_armature(model, joint_names)
    data = mujoco.MjData(model)

    # Inject dataset defaults so freeze_non_arm pins joints to the same pose as teleoperate.py
    for n, v in DATASET_INIT_QPOS.items():
        if n in config["default_joint_pos"]:
            config["default_joint_pos"][n] = v

    # Initial pose — x=-0.50 matches teleoperate.py (robot closer to table)
    data.qpos[0] = -0.50
    data.qpos[2] = 0.76
    data.qpos[3:7] = [1, 0, 0, 0]
    for n, v in config["default_joint_pos"].items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        if jid >= 0:
            addr = int(model.jnt_qposadr[jid])
            data.qpos[addr] = v
    # Override arms + hands with dataset initial pose (matches teleoperate.py)
    for n, v in DATASET_INIT_QPOS.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        if jid >= 0:
            data.qpos[int(model.jnt_qposadr[jid])] = v
    mujoco.mj_forward(model, data)
    randomize_objects(model, data)
    mujoco.mj_forward(model, data)   # update xpos/xmat after repositioning

    # Walker
    print("[init] Loading walker.onnx")
    walker = ONNXPolicy(str(SCRIPT_DIR / "walker.onnx"))
    walker(np.zeros(99, dtype=np.float32))  # warm

    # Policy server
    print(f"[init] Connecting to GR00T server tcp://{args.host}:{args.port}")
    client = PolicyClient(args.host, args.port)
    if not client.ping():
        print(f"[ERROR] Cannot reach server at {args.host}:{args.port}. "
              f"Is run_client.sh tunnel up?")
        return
    print("[init] Server reachable.")

    ctrl = VLAController(model, data, walker, config, client, args.prompt,
                         freeze_non_arm=not args.use_walker,
                         apply_hands=not args.no_hands,
                         save_obs_dir=args.save_obs_dir)
    print(f"[mode] freeze_non_arm={ctrl.freeze_non_arm} apply_hands={ctrl.apply_hands}")

    grip_assist = None if args.no_grip_assist else GripAssist(model, data)
    print(f"[mode] grip_assist={'disabled' if grip_assist is None else 'enabled'}")

    # Preview renderer
    preview = None
    if not args.no_preview:
        preview = mujoco.Renderer(model, 480, 640)
        print("[init] Camera preview enabled (640×480).")

    print(f"[ready] Prompt: {args.prompt!r}")
    print("[ready] Launching viewer — Esc to quit.")

    from mujoco import viewer
    decimation = 4
    step_count = 0
    target_body = ctrl.default.copy()
    hand_ctrl: dict[int, float] = {}
    sim_time = 0.0
    last_preview = 0.0

    # Snapshot initial floating-base pose so we can pin it every step.
    ctrl._base_qpos0 = data.qpos[:7].copy()

    # --- Interactive wrist-camera 6-DOF control ---
    # Camera select : 1 = left wrist   2 = right wrist
    # Position      : Q/A = x   W/S = y   E/D = z          (5 mm/step)
    # Rotation      : ←/→ = yaw   ↑/↓ = pitch   ,/. = roll  (5°/step)
    # All rotations are in the wrist body frame (extrinsic):
    #   yaw → body z-axis, pitch → body y-axis, roll → body x-axis
    _KEY_UP, _KEY_DOWN, _KEY_LEFT, _KEY_RIGHT = 265, 264, 263, 262
    _KEY_COMMA, _KEY_PERIOD = 44, 46
    _KEY_Q, _KEY_A = 81, 65
    _KEY_W, _KEY_S = 87, 83
    _KEY_E, _KEY_D = 69, 68
    _KEY_1, _KEY_2 = 49, 50

    _WRIST_CAMS = ["cam_left_wrist", "cam_right_wrist"]
    _POS_STEP  = 0.005   # metres
    _ROT_STEP  = 5.0     # degrees

    _cam_ids = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, n)
                for n in _WRIST_CAMS}
    # baseline values read from the XML-compiled model
    _cam_base_pos  = {n: model.cam_pos [_cam_ids[n]].copy() for n in _WRIST_CAMS}
    _cam_base_quat = {n: model.cam_quat[_cam_ids[n]].copy() for n in _WRIST_CAMS}
    # accumulated deltas
    _cam_dpos = {n: np.zeros(3) for n in _WRIST_CAMS}          # [dx, dy, dz]
    _cam_drot = {n: np.zeros(3) for n in _WRIST_CAMS}          # [yaw°, pitch°, roll°]
    _active_cam = [_WRIST_CAMS[0]]

    def _quat_to_xyaxes(q: np.ndarray) -> str:
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, q)
        R = mat.reshape(3, 3)
        cx, cy = R[:, 0], R[:, 1]
        return (f"{cx[0]:.4f} {cx[1]:.4f} {cx[2]:.4f}  "
                f"{cy[0]:.4f} {cy[1]:.4f} {cy[2]:.4f}")

    def _apply_cam(name: str) -> None:
        cid = _cam_ids[name]

        # position
        new_pos = _cam_base_pos[name] + _cam_dpos[name]
        model.cam_pos[cid] = new_pos

        # rotation: extrinsic yaw(z) → pitch(y) → roll(x), pre-multiplied onto base
        yr, pr, rr = [np.deg2rad(a) for a in _cam_drot[name]]
        qz = np.array([np.cos(yr/2), 0.0,          0.0,          np.sin(yr/2)])
        qy = np.array([np.cos(pr/2), 0.0,          np.sin(pr/2), 0.0         ])
        qx = np.array([np.cos(rr/2), np.sin(rr/2), 0.0,          0.0         ])
        tmp = np.zeros(4); q_delta = np.zeros(4); new_quat = np.zeros(4)
        mujoco.mju_mulQuat(tmp,     qz, qy)
        mujoco.mju_mulQuat(q_delta, tmp, qx)
        mujoco.mju_mulQuat(new_quat, q_delta, _cam_base_quat[name])
        model.cam_quat[cid] = new_quat

        p  = new_pos
        dr = _cam_drot[name]
        xy = _quat_to_xyaxes(new_quat)
        print(f"[cam] {name}")
        print(f"      pos   : {p[0]:+.4f}  {p[1]:+.4f}  {p[2]:+.4f}"
              f"  (Δ {_cam_dpos[name][0]:+.4f} {_cam_dpos[name][1]:+.4f} {_cam_dpos[name][2]:+.4f})")
        print(f"      rot   : yaw={dr[0]:+.1f}°  pitch={dr[1]:+.1f}°  roll={dr[2]:+.1f}°")
        print(f'      XML   : pos="{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}" xyaxes="{xy}"')

    def _on_key(keycode: int) -> None:
        n = _active_cam[0]
        if   keycode == _KEY_1:
            _active_cam[0] = _WRIST_CAMS[0]
            print(f"[cam] Active → {_active_cam[0]}")
        elif keycode == _KEY_2:
            _active_cam[0] = _WRIST_CAMS[1]
            print(f"[cam] Active → {_active_cam[0]}")
        # position
        elif keycode == _KEY_Q: _cam_dpos[n][0] += _POS_STEP; _apply_cam(n)
        elif keycode == _KEY_A: _cam_dpos[n][0] -= _POS_STEP; _apply_cam(n)
        elif keycode == _KEY_W: _cam_dpos[n][1] += _POS_STEP; _apply_cam(n)
        elif keycode == _KEY_S: _cam_dpos[n][1] -= _POS_STEP; _apply_cam(n)
        elif keycode == _KEY_E: _cam_dpos[n][2] += _POS_STEP; _apply_cam(n)
        elif keycode == _KEY_D: _cam_dpos[n][2] -= _POS_STEP; _apply_cam(n)
        # rotation
        elif keycode == _KEY_LEFT:   _cam_drot[n][0] += _ROT_STEP; _apply_cam(n)
        elif keycode == _KEY_RIGHT:  _cam_drot[n][0] -= _ROT_STEP; _apply_cam(n)
        elif keycode == _KEY_UP:     _cam_drot[n][1] += _ROT_STEP; _apply_cam(n)
        elif keycode == _KEY_DOWN:   _cam_drot[n][1] -= _ROT_STEP; _apply_cam(n)
        elif keycode == _KEY_COMMA:  _cam_drot[n][2] += _ROT_STEP; _apply_cam(n)
        elif keycode == _KEY_PERIOD: _cam_drot[n][2] -= _ROT_STEP; _apply_cam(n)

    print("[cam] Wrist-cam 6-DOF control:")
    print("[cam]   1/2      → select left/right wrist cam")
    print("[cam]   Q/A      → pos x +/-     W/S → pos y +/-     E/D → pos z +/-   (5mm/step)")
    print("[cam]   ←/→      → yaw  +/-      ↑/↓ → pitch +/-     ,/. → roll  +/-   (5°/step)")

    with viewer.launch_passive(model, data, key_callback=_on_key) as v:
        t0 = time.time()
        _hz_last_wall = t0
        _hz_step_count = 0
        while v.is_running():
            wall = time.time() - t0
            if wall - sim_time > 0.05:
                sim_time = wall - 0.05
            while sim_time < wall:
                if step_count % decimation == 0:
                    target_body, hand_ctrl = ctrl.step()
                ctrl.write_ctrl(target_body, hand_ctrl)
                mujoco.mj_step(model, data)
                _hz_step_count += 1
                if _hz_step_count >= 200:  # print every 200 sim steps (~1 s at 200 Hz)
                    now = time.time()
                    elapsed = now - _hz_last_wall
                    sim_hz  = _hz_step_count / elapsed          # actual sim step rate
                    ctrl_hz = (_hz_step_count / decimation) / elapsed  # control policy rate
                    print(f"[hz] sim={sim_hz:.1f} Hz  ctrl={ctrl_hz:.1f} Hz  "
                          f"(target sim=200 Hz, ctrl=50 Hz)")
                    _hz_last_wall = now
                    _hz_step_count = 0
                if ctrl.freeze_non_arm:
                    # Pin floating base in place (kinematic freeze).
                    data.qpos[:7] = ctrl._base_qpos0
                    data.qvel[:6] = 0.0
                    # Pin non-arm body joints to default.
                    for i, n in enumerate(ctrl.joint_names):
                        if n in LEFT_ARM_JOINTS or n in RIGHT_ARM_JOINTS:
                            continue
                        addr = ctrl._qpos_addr[n]
                        data.qpos[addr] = ctrl.default[i]
                        data.qvel[ctrl.qvel_idx[n]] = 0.0
                if grip_assist is not None:
                    grip_assist.update()
                step_count += 1
                sim_time += model.opt.timestep
            v.sync()

            # Camera preview at ~10 Hz
            if preview is not None and time.time() - last_preview > 0.1:
                last_preview = time.time()
                for cam_name in VIDEO_KEYS:
                    preview.update_scene(data, camera=cam_name)
                    img = preview.render()
                    cv2.imshow(cam_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    if preview is not None:
        cv2.destroyAllWindows()
    print("[done]")


if __name__ == "__main__":
    main()
