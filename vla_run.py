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
LEFT_HAND_JOINTS = [
    "left_hand_index_0_joint", "left_hand_index_1_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
]
RIGHT_HAND_JOINTS = [
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
]

LANGUAGE_KEY = "annotation.human.task_description"
VIDEO_KEY = "ego_view"
VIDEO_DELTA = -20  # second video frame is current frame; first is 20 ctrl steps ago
ACTION_HORIZON = 40
EXEC_HORIZON = 8  # how many predicted actions to execute before re-querying
IMG_SIZE = 224


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
        # Override arm defaults to match REAL_G1 training distribution mean.
        _arm_defaults = {
            "left_shoulder_pitch_joint": -0.24, "left_shoulder_roll_joint": 0.26,
            "left_shoulder_yaw_joint": -0.22, "left_elbow_joint": 0.07,
            "left_wrist_roll_joint": -0.12, "left_wrist_pitch_joint": 0.01,
            "left_wrist_yaw_joint": 0.12,
            "right_shoulder_pitch_joint": -0.25, "right_shoulder_roll_joint": -0.29,
            "right_shoulder_yaw_joint": 0.13, "right_elbow_joint": 0.08,
            "right_wrist_roll_joint": 0.02, "right_wrist_pitch_joint": 0.03,
            "right_wrist_yaw_joint": -0.10,
        }
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

        # Frame buffer: head_cam history (most-recent first), used to assemble video.ego_view
        self.frame_hist: list[np.ndarray] = []

        # Action chunk + per-chunk reference state (the "current state" at inference time)
        self.chunk: dict[str, np.ndarray] | None = None
        self.chunk_step = 0
        self.chunk_ref: dict[str, np.ndarray] | None = None

        # Renderer for head_cam observation (separate from preview)
        self.obs_renderer = mujoco.Renderer(model, IMG_SIZE, IMG_SIZE)

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
    def _render_head_cam(self) -> np.ndarray:
        self.obs_renderer.update_scene(self.data, camera="head_cam")
        return self.obs_renderer.render().copy()  # (H, W, 3) uint8

    def _build_observation(self) -> dict[str, Any]:
        base_pos, base_quat = self._base_pose()
        l_palm_xpos = self.data.site_xpos[self.left_palm]
        l_palm_xmat = self.data.site_xmat[self.left_palm]
        r_palm_xpos = self.data.site_xpos[self.right_palm]
        r_palm_xmat = self.data.site_xmat[self.right_palm]

        state = {
            "left_wrist_eef_9d": eef_9d(l_palm_xpos, l_palm_xmat, base_pos, base_quat),
            "right_wrist_eef_9d": eef_9d(r_palm_xpos, r_palm_xmat, base_pos, base_quat),
            "left_hand": self._joint_array(LEFT_HAND_JOINTS),
            "right_hand": self._joint_array(RIGHT_HAND_JOINTS),
            "left_arm": self._joint_array(LEFT_ARM_JOINTS),
            "right_arm": self._joint_array(RIGHT_ARM_JOINTS),
            "waist": self._joint_array(WAIST_JOINTS),
        }
        # Wrap each (D,) -> (1 batch, 1 timestep, D)
        state_batched = {k: v[None, None, :] for k, v in state.items()}

        # Video: 2 frames -> (1, 2, H, W, 3)
        cur = self._render_head_cam()
        if len(self.frame_hist) >= abs(VIDEO_DELTA):
            past = self.frame_hist[-abs(VIDEO_DELTA)]
        elif self.frame_hist:
            past = self.frame_hist[0]
        else:
            past = cur
        video = np.stack([past, cur], axis=0)[None]  # (1, 2, H, W, 3)

        return {
            "video": {VIDEO_KEY: video},
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
        apply_joint_group(WAIST_JOINTS, "waist")

        hand_ctrl: dict[int, float] = {}
        if self.apply_hands:
            hand_groups = [(LEFT_HAND_JOINTS, "left_hand"), (RIGHT_HAND_JOINTS, "right_hand")]
            for names, key in hand_groups:
                vals = c[key][t]
                for i, n in enumerate(names):
                    if n in self.hand_act:
                        hand_ctrl[self.hand_act[n]] = float(vals[i])
        return hand_ctrl

    # ------- main step (called at 50 Hz control rate) -------
    def step(self) -> tuple[np.ndarray, dict[int, float]]:
        if self.freeze_non_arm:
            # No walker: pin everything to default. Only arms (and hands) get VLA targets.
            target_body = self.default.copy()
        else:
            # Walker: full 29D body targets (legs + everything)
            obs = self._walker_obs()
            action = self.walker(obs)
            target_body = self.default + action * self.action_scales
            self.last_action = action.copy()

        # 2) Frame history (cheap — just append latest)
        self.frame_hist.append(self._render_head_cam())
        if len(self.frame_hist) > abs(VIDEO_DELTA) + 5:
            self.frame_hist.pop(0)

        # 3) VLA chunk: query if we don't have one or we've executed enough
        if self.chunk is None or self.chunk_step >= EXEC_HORIZON:
            try:
                obs_dict, cur_state = self._build_observation()
                if self.save_obs_dir:
                    self._save_observation(obs_dict, cur_state)
                t0 = time.time()
                action_chunk = self.client.get_action(obs_dict)
                dt = time.time() - t0
                # Strip batch dim -> (T, D)
                self.chunk = {k: np.asarray(v)[0] for k, v in action_chunk.items()}
                self.chunk_ref = cur_state
                self.chunk_step = 0
                la0 = self.chunk["left_arm"][0]
                ra0 = self.chunk["right_arm"][0]
                T = next(iter(self.chunk.values())).shape[0]
                print(f"[VLA] new chunk ({dt:.2f}s). keys={list(self.chunk.keys())} horizon={T}")
                print(f"[VLA] state  left_arm ={np.round(cur_state['left_arm'], 3).tolist()}")
                print(f"[VLA] action left_arm[0]={np.round(la0, 3).tolist()}")
                print(f"[VLA] state  right_arm={np.round(cur_state['right_arm'], 3).tolist()}")
                print(f"[VLA] action right_arm[0]={np.round(ra0, 3).tolist()}")
                # EEF z trajectory: diagnose whether model plans to go up or down
                leef_z = self.chunk["left_wrist_eef_9d"][:, 2]
                reef_z = self.chunk["right_wrist_eef_9d"][:, 2]
                print(f"[VLA] left_eef  z: state={cur_state['left_wrist_eef_9d'][2]:.3f} "
                      f"chunk=[{leef_z[0]:.3f},{leef_z[4]:.3f},{leef_z[9]:.3f},{leef_z[19]:.3f},{leef_z[T-1]:.3f}] (t=0,4,9,19,39)")
                print(f"[VLA] right_eef z: state={cur_state['right_wrist_eef_9d'][2]:.3f} "
                      f"chunk=[{reef_z[0]:.3f},{reef_z[4]:.3f},{reef_z[9]:.3f},{reef_z[19]:.3f},{reef_z[T-1]:.3f}] (t=0,4,9,19,39)")
                nav = self.chunk.get("navigate_command")
                bh = self.chunk.get("base_height_command")
                if nav is not None:
                    print(f"[VLA] navigate_command[0]={np.round(nav[0], 3).tolist()}")
                if bh is not None:
                    print(f"[VLA] base_height[0]={np.round(bh[0], 3).tolist()}")
            except Exception as e:
                print(f"[VLA] inference failed: {e!r}")
                # keep walker targets only

        # 4) Apply chunk to override upper body
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
        video = obs_dict["video"][VIDEO_KEY]  # (1, 2, H, W, 3)
        past_frame = video[0, 0]
        cur_frame = video[0, 1]
        Image.fromarray(past_frame).save(f"{prefix}_frame_past.png")
        Image.fromarray(cur_frame).save(f"{prefix}_frame_current.png")

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

    # Initial pose (moved back to x=-0.7 so arms clear table at x=-0.40)
    data.qpos[0] = -0.7
    data.qpos[2] = 0.76
    data.qpos[3:7] = [1, 0, 0, 0]
    for n, v in config["default_joint_pos"].items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        if jid >= 0:
            addr = int(model.jnt_qposadr[jid])
            data.qpos[addr] = v
    # Override arm defaults to be closer to REAL_G1 training distribution mean.
    # model_config defaults (elbow=0.6, shoulder_pitch=0.2) are significantly
    # OOD vs training mean (elbow≈0.07, shoulder_pitch≈-0.24).
    arm_init = {
        "left_shoulder_pitch_joint": -0.24, "left_shoulder_roll_joint": 0.26,
        "left_shoulder_yaw_joint": -0.22, "left_elbow_joint": 0.07,
        "left_wrist_roll_joint": -0.12, "left_wrist_pitch_joint": 0.01,
        "left_wrist_yaw_joint": 0.12,
        "right_shoulder_pitch_joint": -0.25, "right_shoulder_roll_joint": -0.29,
        "right_shoulder_yaw_joint": 0.13, "right_elbow_joint": 0.08,
        "right_wrist_roll_joint": 0.02, "right_wrist_pitch_joint": 0.03,
        "right_wrist_yaw_joint": -0.10,
    }
    for n, v in arm_init.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        if jid >= 0:
            data.qpos[int(model.jnt_qposadr[jid])] = v
    mujoco.mj_forward(model, data)

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

    # Preview renderer
    preview = None
    if not args.no_preview:
        preview = mujoco.Renderer(model, 320, 320)

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

    with viewer.launch_passive(model, data) as v:
        t0 = time.time()
        while v.is_running():
            wall = time.time() - t0
            if wall - sim_time > 0.05:
                sim_time = wall - 0.05
            while sim_time < wall:
                if step_count % decimation == 0:
                    target_body, hand_ctrl = ctrl.step()
                ctrl.write_ctrl(target_body, hand_ctrl)
                mujoco.mj_step(model, data)
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
                step_count += 1
                sim_time += model.opt.timestep
            v.sync()

            # Head-cam preview at ~10 Hz
            if preview is not None and time.time() - last_preview > 0.1:
                last_preview = time.time()
                preview.update_scene(data, camera="head_cam")
                img = preview.render()
                cv2.imshow("VLA head_cam", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    if preview is not None:
        cv2.destroyAllWindows()
    print("[done]")


if __name__ == "__main__":
    main()
