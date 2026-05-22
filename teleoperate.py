#!/usr/bin/env python3
"""
teleoperate.py  —  Full-range direct joint control + episode recording for the G1.

Bypasses the ONNX reacher. Every arm/waist joint is driven directly to its full
hardware range. Intended for collecting pick-and-place demonstrations for VLA
fine-tuning.

Keyboard bindings (press in the MuJoCo viewer window):
  TAB        : Cycle active limb  LEFT_ARM → RIGHT_ARM → WAIST
  1–7        : Select joint within active limb  (WAIST: 1–3)
  UP / DOWN  : Move selected joint  +step / −step
  [ / ]      : Halve / double step size  (default 0.05 rad)
  ,          : Toggle RIGHT hand grip
  M          : Toggle LEFT hand grip
  Space      : Reset robot (discards current unsaved episode)
  R          : Toggle recording (start / stop)
  S          : Save episode to disk  (also stops recording)
  P          : Print current arm joint angles
  Esc / Q    : Quit

Xbox controller (connect before launching):
  Left  stick Y   : ShoulderPitch  (push up = arm forward)
  Left  stick X   : ShoulderRoll   (push right = roll right)
  Right stick Y   : Elbow          (push up = bend)
  Right stick X   : WristRoll      (push right = roll right)
  D-pad ▲ ▼       : Fine ShoulderPitch ±step
  Back  (⧉)       : Switch active arm  left ↔ right
  LB / RB         : Halve / double step size
  Left trigger    : Left hand grip  (hold = closed)
  Right trigger   : Right hand grip (hold = closed)
  A               : Toggle recording
  Start           : Save episode
  Y               : Reset robot
  X               : Print joints

Usage:
  uv run python teleoperate.py
  uv run python teleoperate.py --no-cameras
  uv run python teleoperate.py --no-gamepad          # keyboard only
  uv run python teleoperate.py --save-dir ./my_demos --use-walker
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

try:
    import cv2 as _cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    _cv2 = None

try:
    from PIL import Image as _PIL_Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    _PIL_Image = None

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False
    pygame = None  # type: ignore

SCRIPT_DIR = Path(__file__).resolve().parent

# ── Dataset initial joint positions (from real-robot trajectory) ──
DATASET_INIT_QPOS = {
  "left_shoulder_pitch_joint": -0.186601,
  "left_shoulder_roll_joint":   0.316981,
  "left_shoulder_yaw_joint":    0.287689,
  "left_elbow_joint":          -0.635120,
  "left_wrist_roll_joint":     -0.243937,
  "left_wrist_pitch_joint":     0.690306,
  "left_wrist_yaw_joint":      -0.015776,
  "right_shoulder_pitch_joint": -0.364927,
  "right_shoulder_roll_joint": -0.258941,
  "right_shoulder_yaw_joint":  -0.201870,
  "right_elbow_joint":         -0.514582,
  "right_wrist_roll_joint":     0.261457,
  "right_wrist_pitch_joint":    0.718800,
  "right_wrist_yaw_joint":      0.073139,
  "left_hand_thumb_0_joint":   -0.427740,
  "left_hand_thumb_1_joint":    1.028506,
  "left_hand_thumb_2_joint":    0.184237,
  "left_hand_middle_0_joint":   0.175383,
  "left_hand_middle_1_joint":  -0.047019,
  "left_hand_index_0_joint":    0.184752,
  "left_hand_index_1_joint":   -0.015129,
  "right_hand_thumb_0_joint":  -0.404762,
  "right_hand_thumb_1_joint":  -1.038415,
  "right_hand_thumb_2_joint":  -0.326415,
  "right_hand_index_0_joint":  -0.189341,
  "right_hand_index_1_joint":   0.012102,
  "right_hand_middle_0_joint": -0.184233,
  "right_hand_middle_1_joint":  0.011182,
}

# ────────────────────────────────────────────────────────────────────────────
# Joint groups  (order matches REAL_G1 GR00T training statistics)
# ────────────────────────────────────────────────────────────────────────────
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

# Hand joints in GR00T order: [index_0, index_1, middle_0, middle_1, thumb_0, thumb_1, thumb_2]
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

# Power-grasp closed positions (at mechanical limits)
RIGHT_GRIP_CLOSED = {
    "right_hand_index_0_joint": 1.4, "right_hand_index_1_joint": 1.5,
    "right_hand_middle_0_joint": 1.4, "right_hand_middle_1_joint": 1.5,
    "right_hand_thumb_0_joint": 0.8, "right_hand_thumb_1_joint": -0.9,
    "right_hand_thumb_2_joint": -1.5,
}
LEFT_GRIP_CLOSED = {
    "left_hand_index_0_joint": -1.4, "left_hand_index_1_joint": -1.5,
    "left_hand_middle_0_joint": -1.4, "left_hand_middle_1_joint": -1.5,
    "left_hand_thumb_0_joint": -0.8, "left_hand_thumb_1_joint": 0.9,
    "left_hand_thumb_2_joint": 1.5,
}

# Hardware limits from g1.xml (used for clamping keyboard input)
JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "left_shoulder_pitch_joint": (-3.089, 2.670),
    "left_shoulder_roll_joint":  (-1.588, 2.252),
    "left_shoulder_yaw_joint":   (-2.618, 2.618),
    "left_elbow_joint":          (-1.047, 2.094),
    "left_wrist_roll_joint":     (-1.972, 1.972),
    "left_wrist_pitch_joint":    (-1.614, 1.614),
    "left_wrist_yaw_joint":      (-1.614, 1.614),
    "right_shoulder_pitch_joint":(-3.089, 2.670),
    "right_shoulder_roll_joint": (-2.252, 1.588),
    "right_shoulder_yaw_joint":  (-2.618, 2.618),
    "right_elbow_joint":         (-1.047, 2.094),
    "right_wrist_roll_joint":    (-1.972, 1.972),
    "right_wrist_pitch_joint":   (-1.614, 1.614),
    "right_wrist_yaw_joint":     (-1.614, 1.614),
    "waist_yaw_joint":           (-2.618, 2.618),
    "waist_roll_joint":          (-0.520, 0.520),
    "waist_pitch_joint":         (-0.520, 0.520),
}

LIMB_CYCLE = ["left_arm", "right_arm", "waist"]
LIMB_JOINTS: dict[str, list[str]] = {
    "left_arm":  LEFT_ARM_JOINTS,
    "right_arm": RIGHT_ARM_JOINTS,
    "waist":     WAIST_JOINTS,
}
LIMB_JOINT_LABELS: dict[str, list[str]] = {
    "left_arm":  ["L-ShPitch", "L-ShRoll", "L-ShYaw", "L-Elbow", "L-WrRoll", "L-WrPitch", "L-WrYaw"],
    "right_arm": ["R-ShPitch", "R-ShRoll", "R-ShYaw", "R-Elbow", "R-WrRoll", "R-WrPitch", "R-WrYaw"],
    "waist":     ["WaistYaw", "WaistRoll", "WaistPitch"],
}

# ────────────────────────────────────────────────────────────────────────────
# Small helpers
# ────────────────────────────────────────────────────────────────────────────
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


def quat_apply_inv(q, v):
    w, xyz = q[0], q[1:4]
    t = np.cross(xyz, v) * 2
    return v - w * t + np.cross(xyz, t)


def quat_to_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


def eef_9d(site_xpos, site_xmat, base_pos, base_quat) -> np.ndarray:
    pos_rel = quat_apply_inv(base_quat, site_xpos - base_pos)
    R_rel = quat_to_mat(base_quat).T @ site_xmat.reshape(3, 3)
    return np.concatenate([pos_rel, R_rel[:2].flatten()]).astype(np.float32)


# ────────────────────────────────────────────────────────────────────────────
# Xbox gamepad helper
# ────────────────────────────────────────────────────────────────────────────
# Standard Linux axis/button indices for Xbox 360 / One controller via xpad
_GP_AXIS_LX        = 0
_GP_AXIS_LY        = 1
_GP_AXIS_LTRIGGER  = 2   # resting at -1.0, fully pressed at +1.0
_GP_AXIS_RX        = 3
_GP_AXIS_RY        = 4
_GP_AXIS_RTRIGGER  = 5   # resting at -1.0, fully pressed at +1.0
_GP_BTN_A          = 0
_GP_BTN_B          = 1
_GP_BTN_X          = 2
_GP_BTN_Y          = 3
_GP_BTN_LB         = 4
_GP_BTN_RB         = 5
_GP_BTN_BACK       = 6   # View / Back / Select
_GP_BTN_START      = 7   # Menu / Start
_TRIGGER_THRESHOLD = 0.5  # trigger axis value above which grip is "closed"
_STICK_DEADZONE    = 0.12
_STICK_MAX_DRAD    = 0.022  # max joint delta per 50 Hz control step (≈1.1 rad/s)


class GamepadController:
    """Thin wrapper around a single pygame joystick."""

    def __init__(self):
        if not HAS_PYGAME:
            raise RuntimeError("pygame is not installed; run: uv add pygame")
        pygame.init()
        pygame.joystick.init()
        n = pygame.joystick.get_count()
        if n == 0:
            raise RuntimeError("No gamepad detected. Connect Xbox controller and retry.")
        self.joy = pygame.joystick.Joystick(0)
        self.joy.init()
        print(f"[gamepad] Connected: {self.joy.get_name()}  "
              f"({self.joy.get_numaxes()} axes, {self.joy.get_numbuttons()} buttons)")
        # track previous hat state to detect edge transitions
        self._prev_hat  = (0, 0)
        self._prev_btns = {}

    def _btn(self, idx: int) -> bool:
        try:
            return bool(self.joy.get_button(idx))
        except Exception:
            return False

    def _axis(self, idx: int) -> float:
        try:
            return float(self.joy.get_axis(idx))
        except Exception:
            return 0.0

    def poll(self, ctrl: "TeleoperateController", state: dict):
        """Process all pending pygame events and apply gamepad state to ctrl."""
        pygame.event.pump()

        # ── Helpers ──────────────────────────────────────────────────────
        def sdelta(raw_axis: float, invert: bool = True) -> float:
            """Convert a raw stick axis value to a joint delta (rad/step)."""
            v = -raw_axis if invert else raw_axis
            if abs(v) <= _STICK_DEADZONE:
                return 0.0
            norm = (abs(v) - _STICK_DEADZONE) / (1.0 - _STICK_DEADZONE)
            return float(np.sign(v)) * norm * _STICK_MAX_DRAD

        def move(name: str, delta: float):
            if not name or delta == 0.0:
                return
            lo, hi = JOINT_LIMITS.get(name, (-3.14, 3.14))
            ctrl._targets[name] = float(np.clip(ctrl._targets.get(name, 0.0) + delta, lo, hi))

        # ── Sticks → ShPitch / ShRoll / Elbow / WrRoll ──────────────────
        # Left  stick: Y → ShoulderPitch,  X → ShoulderRoll
        # Right stick: Y → Elbow,          X → WristRoll
        limb = LIMB_CYCLE[ctrl.limb_idx]
        p = "left" if limb == "left_arm" else "right" if limb == "right_arm" else None
        if p:
            move(f"{p}_shoulder_pitch_joint", sdelta(self._axis(_GP_AXIS_LY)))
            move(f"{p}_shoulder_roll_joint",  sdelta(self._axis(_GP_AXIS_LX), invert=False))
            move(f"{p}_elbow_joint",          sdelta(self._axis(_GP_AXIS_RY)))
            move(f"{p}_wrist_roll_joint",     sdelta(self._axis(_GP_AXIS_RX), invert=False))

        # ── D-pad → discrete ShoulderPitch nudge (fine tuning) ──────────
        hat = self.joy.get_hat(0) if self.joy.get_numhats() > 0 else (0, 0)
        if hat[1] != self._prev_hat[1]:
            if hat[1] == 1 and p:
                move(f"{p}_shoulder_pitch_joint", +ctrl.step_size)
            elif hat[1] == -1 and p:
                move(f"{p}_shoulder_pitch_joint", -ctrl.step_size)
        self._prev_hat = hat

        # ── Buttons (edge-triggered) ─────────────────────────────────────
        def _edge(idx: int) -> bool:
            cur = self._btn(idx)
            prev_v = self._prev_btns.get(idx, False)
            self._prev_btns[idx] = cur
            return cur and not prev_v

        if _edge(_GP_BTN_BACK):          # Back → toggle left_arm / right_arm
            ctrl.limb_idx = 1 if ctrl.limb_idx == 0 else 0
            ctrl.joint_idx = 0
            print(f"[gamepad] active arm: {LIMB_CYCLE[ctrl.limb_idx]}")

        if _edge(_GP_BTN_LB):
            ctrl.step_size = max(0.005, round(ctrl.step_size / 2, 4))
            print(f"[step] {ctrl.step_size:.4f} rad")

        if _edge(_GP_BTN_RB):
            ctrl.step_size = min(0.5, round(ctrl.step_size * 2, 4))
            print(f"[step] {ctrl.step_size:.4f} rad")

        if _edge(_GP_BTN_A):             # A → toggle recording
            ctrl.key_callback(ctrl.KEY_R)

        if _edge(_GP_BTN_START):         # Start → save episode
            ctrl._save_episode()

        if _edge(_GP_BTN_Y):             # Y → reset robot
            state["reset"] = True

        if _edge(_GP_BTN_X):             # X → print joints
            print("[joints]")
            ctrl._print_joints()

        # ── Triggers → grip hold (hold = closed) ─────────────────────────
        lt = (self._axis(_GP_AXIS_LTRIGGER) + 1.0) / 2.0   # remap [−1,+1] → [0,1]
        rt = (self._axis(_GP_AXIS_RTRIGGER) + 1.0) / 2.0
        new_lg = lt > _TRIGGER_THRESHOLD
        new_rg = rt > _TRIGGER_THRESHOLD
        if new_lg != ctrl.left_grip:
            ctrl.left_grip = new_lg
            print(f"[grip L] {'CLOSED' if ctrl.left_grip else 'open'}")
        if new_rg != ctrl.right_grip:
            ctrl.right_grip = new_rg
            print(f"[grip R] {'CLOSED' if ctrl.right_grip else 'open'}")


# ────────────────────────────────────────────────────────────────────────────
# Main controller
# ────────────────────────────────────────────────────────────────────────────
class TeleoperateController:
    # GLFW key codes
    KEY_TAB   = 258
    KEY_NUMS  = list(range(49, 56))   # 1–7
    KEY_UP    = 265
    KEY_DOWN  = 264
    KEY_LB    = 91    # [  → halve step
    KEY_RB    = 93    # ]  → double step
    KEY_COMMA = 44    # ,  → right grip
    KEY_M     = 77    # M  → left grip
    KEY_P     = 80    # P  → print angles
    KEY_R     = 82    # R  → record toggle
    KEY_S     = 83    # S  → save episode

    def __init__(self, model, data, walker, config,
                 freeze_base: bool = True,
                 save_dir: Path | None = None):
        self.model = model
        self.data = data
        self.walker = walker
        self.config = config
        self.freeze_base = freeze_base

        # Joint config from model_config.json
        self.joint_names: list[str] = config["joint_names"]
        self.num_joints = len(self.joint_names)

        self.default = np.zeros(self.num_joints, dtype=np.float32)
        for n, v in config["default_joint_pos"].items():
            if n in self.joint_names:
                self.default[self.joint_names.index(n)] = v

        self.action_scales = np.array(
            [config["action_scales"][n] for n in self.joint_names], dtype=np.float32
        )

        # Pre-cache MuJoCo addresses for every joint we use
        all_joints = (self.joint_names + LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS
                      + WAIST_JOINTS + LEFT_HAND_JOINTS + RIGHT_HAND_JOINTS)
        self._qpos_addr: dict[str, int] = {}
        self._qvel_addr: dict[str, int] = {}
        self._act_id: dict[str, int] = {}
        for n in dict.fromkeys(all_joints):   # deduplicated
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
            if jid >= 0:
                self._qpos_addr[n] = int(model.jnt_qposadr[jid])
                self._qvel_addr[n] = int(model.jnt_dofadr[jid])
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            if aid >= 0:
                self._act_id[n] = aid

        self.left_palm  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_palm")
        self.right_palm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_palm")


        # Teleoperation state
        self.limb_idx  = 1     # start on right_arm
        self.joint_idx = 0     # first joint in current limb
        self.step_size = 0.05  # rad per keypress
        self.right_grip = False
        self.left_grip  = False
        self.last_walker_action = np.zeros(self.num_joints, dtype=np.float32)

        # Manual joint targets – initialised from current qpos
        self._targets: dict[str, float] = {}
        mujoco.mj_forward(model, data)
        for n in LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + WAIST_JOINTS:
            addr = self._qpos_addr.get(n)
            self._targets[n] = float(data.qpos[addr]) if addr is not None else 0.0

        # Base pose snapshot for kinematic freeze
        self._base_qpos0 = data.qpos[:7].copy()

        # Episode recording
        self.recording = False
        self._episode_joints: dict[str, list] = {
            k: [] for k in ("left_arm", "right_arm", "waist",
                            "left_hand", "right_hand",
                            "left_wrist_eef_9d", "right_wrist_eef_9d",
                            "left_grip", "right_grip", "timestamp")
        }
        self._episode_frames: list[np.ndarray] = []
        self._frame_interval = 5   # save a camera frame every N control steps
        self._ctrl_step = 0

        self.save_dir = save_dir or (SCRIPT_DIR / "episodes")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._ep_idx = len(list(self.save_dir.glob("episode_*")))

        # Off-screen renderer for observation images (224×224, matches VLA input)
        self._obs_renderer = mujoco.Renderer(model, 224, 224)

        self._print_status()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _qpos(self, name: str) -> float:
        return float(self.data.qpos[self._qpos_addr[name]])

    def _joint_array(self, names: list[str]) -> np.ndarray:
        return np.array([self._qpos(n) for n in names], dtype=np.float32)

    def _render_head(self) -> np.ndarray:
        self._obs_renderer.update_scene(self.data, camera="cam_left_high")
        return self._obs_renderer.render().copy()

    def _print_status(self):
        limb   = LIMB_CYCLE[self.limb_idx]
        joints = LIMB_JOINTS[limb]
        labels = LIMB_JOINT_LABELS[limb]
        jlabel = labels[self.joint_idx] if self.joint_idx < len(labels) else "?"
        jname  = joints[self.joint_idx] if self.joint_idx < len(joints) else "?"
        cur    = self._targets.get(jname, 0.0)
        lo, hi = JOINT_LIMITS.get(jname, (-3.14, 3.14))
        print(f"[ctrl] limb={limb}  joint[{self.joint_idx+1}]={jlabel}  "
              f"cur={cur:.4f}  range=[{lo:.3f},{hi:.3f}]  step={self.step_size:.3f}rad")

    def _print_joints(self):
        for limb, joints in LIMB_JOINTS.items():
            vals = {n: round(self._targets.get(n, 0.0), 4) for n in joints}
            print(f"  {limb}: {vals}")

    # ── keyboard callback ─────────────────────────────────────────────────────

    def key_callback(self, key: int):
        if key == self.KEY_TAB:
            self.limb_idx  = (self.limb_idx + 1) % len(LIMB_CYCLE)
            self.joint_idx = 0
            self._print_status()

        elif key in self.KEY_NUMS:
            j = key - self.KEY_NUMS[0]
            if j < len(LIMB_JOINTS[LIMB_CYCLE[self.limb_idx]]):
                self.joint_idx = j
                self._print_status()

        elif key == self.KEY_UP:
            self._nudge(+self.step_size)

        elif key == self.KEY_DOWN:
            self._nudge(-self.step_size)

        elif key == self.KEY_LB:
            self.step_size = max(0.005, round(self.step_size / 2, 4))
            print(f"[step] {self.step_size:.4f} rad")

        elif key == self.KEY_RB:
            self.step_size = min(0.5, round(self.step_size * 2, 4))
            print(f"[step] {self.step_size:.4f} rad")

        elif key == self.KEY_COMMA:
            self.right_grip = not self.right_grip
            print(f"[grip R] {'CLOSED' if self.right_grip else 'open'}")

        elif key == self.KEY_M:
            self.left_grip = not self.left_grip
            print(f"[grip L] {'CLOSED' if self.left_grip else 'open'}")

        elif key == self.KEY_P:
            print("[joints]")
            self._print_joints()

        elif key == self.KEY_R:
            if self.recording:
                self.recording = False
                n = len(self._episode_joints["timestamp"])
                print(f"[rec] STOPPED  ({n} steps, {len(self._episode_frames)} frames)")
            else:
                self._clear_episode()
                self.recording = True
                print("[rec] ● RECORDING …  (press R to stop, S to save)")

        elif key == self.KEY_S:
            self._save_episode()

    def _nudge(self, delta: float):
        limb   = LIMB_CYCLE[self.limb_idx]
        joints = LIMB_JOINTS[limb]
        if self.joint_idx >= len(joints):
            return
        name = joints[self.joint_idx]
        lo, hi = JOINT_LIMITS.get(name, (-3.14, 3.14))
        new_val = float(np.clip(self._targets.get(name, 0.0) + delta, lo, hi))
        self._targets[name] = new_val
        label = LIMB_JOINT_LABELS[limb][self.joint_idx]
        bar_pos = int(20 * (new_val - lo) / max(hi - lo, 1e-6))
        bar = "[" + "=" * bar_pos + " " * (20 - bar_pos) + "]"
        print(f"  {label:12s} {new_val:+.4f}  {bar}  [{lo:.3f}, {hi:.3f}]")

    # ── simulation step ───────────────────────────────────────────────────────

    def step(self) -> np.ndarray:
        """Compute one control cycle and return 29D body joint targets."""

        if self.freeze_base:
            target = self.default.copy()
        else:
            # Walker policy keeps legs balanced
            base_quat  = self.data.qpos[3:7]
            lin_world  = self.data.qvel[:3]
            lin_body   = quat_apply_inv(base_quat, lin_world)
            ang_body   = self.data.qvel[3:6]
            proj_g     = quat_apply_inv(base_quat, np.array([0., 0., -1.]))
            jpos = np.array([self.data.qpos[7 + i] - self.default[i]
                             for i in range(self.num_joints)], dtype=np.float32)
            jvel = np.array([self.data.qvel[6 + i]
                             for i in range(self.num_joints)], dtype=np.float32)
            cmd = np.zeros(3, dtype=np.float32)
            obs = np.concatenate([lin_body, ang_body, proj_g, jpos, jvel,
                                  self.last_walker_action, cmd]).astype(np.float32)
            wa = self.walker(obs)
            target = self.default + wa * self.action_scales
            self.last_walker_action = wa.copy()

        # Override arms + waist with keyboard targets
        for name, val in self._targets.items():
            if name in self.joint_names:
                target[self.joint_names.index(name)] = val

        # ── record ──────────────────────────────────────────────────────────
        if self.recording:
            base_pos  = self.data.qpos[:3].copy()
            base_quat = self.data.qpos[3:7].copy()
            leef = eef_9d(self.data.site_xpos[self.left_palm],
                          self.data.site_xmat[self.left_palm], base_pos, base_quat)
            reef = eef_9d(self.data.site_xpos[self.right_palm],
                          self.data.site_xmat[self.right_palm], base_pos, base_quat)
            self._episode_joints["left_arm"].append(self._joint_array(LEFT_ARM_JOINTS))
            self._episode_joints["right_arm"].append(self._joint_array(RIGHT_ARM_JOINTS))
            self._episode_joints["waist"].append(self._joint_array(WAIST_JOINTS))
            self._episode_joints["left_hand"].append(self._joint_array(LEFT_HAND_JOINTS))
            self._episode_joints["right_hand"].append(self._joint_array(RIGHT_HAND_JOINTS))
            self._episode_joints["left_wrist_eef_9d"].append(leef)
            self._episode_joints["right_wrist_eef_9d"].append(reef)
            self._episode_joints["left_grip"].append(bool(self.left_grip))
            self._episode_joints["right_grip"].append(bool(self.right_grip))
            self._episode_joints["timestamp"].append(time.time())

            if self._ctrl_step % self._frame_interval == 0:
                self._episode_frames.append(self._render_head())

        self._ctrl_step += 1
        return target

    def write_ctrl(self, target: np.ndarray):
        """Apply joint targets + grip state to MuJoCo actuators."""
        for n in self.joint_names:
            aid = self._act_id.get(n, -1)
            if aid >= 0:
                target_val = target[self.joint_names.index(n)]
                self.data.ctrl[aid] = target_val

        # Right hand grip
        for n, closed_v in RIGHT_GRIP_CLOSED.items():
            aid = self._act_id.get(n, -1)
            if aid >= 0:
                open_v = DATASET_INIT_QPOS.get(n, 0.0)
                self.data.ctrl[aid] = closed_v if self.right_grip else open_v

        # Left hand grip
        for n, closed_v in LEFT_GRIP_CLOSED.items():
            aid = self._act_id.get(n, -1)
            if aid >= 0:
                open_v = DATASET_INIT_QPOS.get(n, 0.0)
                self.data.ctrl[aid] = closed_v if self.left_grip else open_v

    # ── episode I/O ──────────────────────────────────────────────────────────

    def _clear_episode(self):
        for lst in self._episode_joints.values():
            lst.clear()
        self._episode_frames.clear()

    def _save_episode(self):
        if self.recording:
            self.recording = False
        n_steps = len(self._episode_joints["timestamp"])
        if n_steps == 0:
            print("[save] Nothing to save (episode is empty).")
            return

        ep_dir = self.save_dir / f"episode_{self._ep_idx:04d}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        # Save state arrays
        npz_data = {k: np.array(v) for k, v in self._episode_joints.items()}
        np.savez_compressed(ep_dir / "states.npz", **npz_data)

        # Save camera frames
        n_frames = len(self._episode_frames)
        if n_frames:
            frames_dir = ep_dir / "frames"
            frames_dir.mkdir(exist_ok=True)
            if HAS_PIL:
                for i, f in enumerate(self._episode_frames):
                    _PIL_Image.fromarray(f).save(frames_dir / f"{i:06d}.png")
            elif HAS_CV2:
                for i, f in enumerate(self._episode_frames):
                    _cv2.imwrite(str(frames_dir / f"{i:06d}.png"),
                                 _cv2.cvtColor(f, _cv2.COLOR_RGB2BGR))

        # Save metadata
        with open(ep_dir / "meta.json", "w") as f:
            json.dump({
                "n_steps": n_steps,
                "n_frames": n_frames,
                "frame_interval_ctrl_steps": self._frame_interval,
                "control_hz": 50,
                "image_size": 224,
                "embodiment_tag": "real_g1_relative_eef_relative_joints",
                "joints": {
                    "left_arm": LEFT_ARM_JOINTS,
                    "right_arm": RIGHT_ARM_JOINTS,
                    "waist": WAIST_JOINTS,
                    "left_hand": LEFT_HAND_JOINTS,
                    "right_hand": RIGHT_HAND_JOINTS,
                },
            }, f, indent=2)

        print(f"[save] episode_{self._ep_idx:04d}  →  {ep_dir}")
        print(f"       {n_steps} steps  |  {n_frames} frames  |  states.npz + meta.json")
        self._ep_idx += 1
        self._clear_episode()


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-dir", default=None,
                    help="Root directory for saved episodes (default: ./episodes/)")
    ap.add_argument("--no-cameras", action="store_true",
                    help="Disable OpenCV camera preview windows")
    ap.add_argument("--use-walker", action="store_true",
                    help="Run walker policy to balance legs (default: freeze base + legs)")
    ap.add_argument("--cam-fps", type=int, default=10,
                    help="Preview camera FPS (default 10)")
    ap.add_argument("--no-gamepad", action="store_true",
                    help="Disable Xbox controller even if pygame is available")
    args = ap.parse_args()

    with open(SCRIPT_DIR / "model_config.json") as f:
        config = json.load(f)
    joint_names = config["joint_names"]

    print(f"[init] Loading scene …")
    model = mujoco.MjModel.from_xml_path(str(SCRIPT_DIR / "scene.xml"))
    model.opt.timestep = 0.005
    set_armature(model, joint_names)
    data = mujoco.MjData(model)

    # Inject dataset defaults into config so freeze_base uses them
    for n, v in DATASET_INIT_QPOS.items():
        if n in config["default_joint_pos"]:
            config["default_joint_pos"][n] = v

    # Initial robot pose
    data.qpos[0] = -0.50   # closer to table: cubes now ~0.30 m forward in pelvis frame
    data.qpos[2] = 0.76
    data.qpos[3:7] = [1, 0, 0, 0]
    for n, v in config["default_joint_pos"].items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        if jid >= 0:
            data.qpos[int(model.jnt_qposadr[jid])] = v
    # Override arms + hands with dataset initial pose
    for n, v in DATASET_INIT_QPOS.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
        if jid >= 0:
            data.qpos[int(model.jnt_qposadr[jid])] = v
    mujoco.mj_forward(model, data)

    walker = None
    if args.use_walker:
        print("[init] Loading walker.onnx …")
        walker = ONNXPolicy(str(SCRIPT_DIR / "walker.onnx"))
        walker(np.zeros(99, dtype=np.float32))  # warm-up
    else:
        print("[init] Base frozen (no walker).")

    save_dir = Path(args.save_dir) if args.save_dir else None
    ctrl = TeleoperateController(
        model, data, walker, config,
        freeze_base=not args.use_walker,
        save_dir=save_dir,
    )
    base_qpos0 = data.qpos[:7].copy()

    # Gamepad
    gamepad = None
    if not args.no_gamepad and HAS_PYGAME:
        try:
            gamepad = GamepadController()
        except RuntimeError as e:
            print(f"[gamepad] {e} — keyboard only.")
    elif not HAS_PYGAME and not args.no_gamepad:
        print("[gamepad] pygame not installed — keyboard only.")

    # Preview renderer (separate from obs renderer to allow different size)
    preview_renderer = None
    if not args.no_cameras and HAS_CV2:
        preview_renderer = mujoco.Renderer(model, 480, 640)
        print("[init] Camera preview enabled (640×480).")

    print()
    print("=" * 60)
    print("  TELEOPERATE  —  G1 Direct Joint Control")
    print("=" * 60)
    print("  KEYBOARD")
    print("  TAB          Cycle limb  (left_arm / right_arm / waist)")
    print("  1–7          Select joint within current limb")
    print("  UP / DOWN    Move joint  +step / −step")
    print("  [ / ]        Halve / double step size")
    print("  ,            Toggle RIGHT hand grip")
    print("  M            Toggle LEFT hand grip")
    print("  R            Start / stop recording")
    print("  S            Save episode to disk")
    print("  P            Print current joint angles")
    print("  Space        Reset robot")
    if gamepad:
        print()
        print("  XBOX CONTROLLER")
        print("  Left  stick Y  ShoulderPitch  (up = forward)")
        print("  Left  stick X  ShoulderRoll   (right = right)")
        print("  Right stick Y  Elbow          (up = bend)")
        print("  Right stick X  WristRoll      (right = roll)")
        print("  D-pad ▲ ▼      Fine ShoulderPitch ±step")
        print("  Back           Switch arm  left ↔ right")
        print("  LB / RB        Halve / double step size")
        print("  L-trigger      Left hand grip  (hold = closed)")
        print("  R-trigger      Right hand grip (hold = closed)")
        print("  A              Toggle recording")
        print("  Start          Save episode")
        print("  Y              Reset robot")
        print("  X              Print joints")
    print(f"  Episodes → {ctrl.save_dir}")
    print("=" * 60)
    print()

    from mujoco import viewer
    decimation  = 4
    step_count  = 0
    target      = ctrl.default.copy()
    sim_time    = 0.0
    last_prev   = 0.0
    prev_intv   = 1.0 / max(args.cam_fps, 1)
    state       = {"reset": False}

    def on_key(key: int):
        if key == 32:   # Space
            state["reset"] = True
        else:
            ctrl.key_callback(key)

    with viewer.launch_passive(model, data, key_callback=on_key) as v:
        t0 = time.time()
        while v.is_running():
            # ── Reset ──────────────────────────────────────────────────────
            if state["reset"]:
                mujoco.mj_resetData(model, data)
                data.qpos[0] = -0.50
                data.qpos[2] = 0.76
                data.qpos[3:7] = [1, 0, 0, 0]
                for n, val in config["default_joint_pos"].items():
                    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                    if jid >= 0:
                        data.qpos[int(model.jnt_qposadr[jid])] = val
                for n, val in DATASET_INIT_QPOS.items():
                    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                    if jid >= 0:
                        data.qpos[int(model.jnt_qposadr[jid])] = val
                mujoco.mj_forward(model, data)
                base_qpos0 = data.qpos[:7].copy()
                ctrl.last_walker_action[:] = 0
                ctrl.right_grip = False
                ctrl.left_grip  = False
                if ctrl.recording:
                    ctrl.recording = False
                    ctrl._clear_episode()
                # Reset manual targets to current qpos
                for n in LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + WAIST_JOINTS:
                    addr = ctrl._qpos_addr.get(n)
                    ctrl._targets[n] = float(data.qpos[addr]) if addr is not None else 0.0
                target = ctrl.default.copy()
                state["reset"] = False
                print("[reset] Robot reset.")

            # ── Gamepad ────────────────────────────────────────────────────
            if gamepad is not None:
                try:
                    gamepad.poll(ctrl, state)
                except Exception as e:
                    print(f"[gamepad] error: {e}")
                    gamepad = None

            # ── Physics ────────────────────────────────────────────────────
            wall = time.time() - t0
            if wall - sim_time > 0.05:    # cap lag
                sim_time = wall - 0.05
            while sim_time < wall:
                if step_count % decimation == 0:
                    target = ctrl.step()
                ctrl.write_ctrl(target)
                try:
                    mujoco.mj_step(model, data)
                except mujoco.FatalError as e:
                    print(f"[physics] solver error — opening grip to recover: {e}")
                    ctrl.right_grip = False
                    ctrl.left_grip  = False
                if ctrl.freeze_base:
                    data.qpos[:7] = base_qpos0
                    data.qvel[:6] = 0.0
                    for i, n in enumerate(joint_names):
                        if n not in (LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + WAIST_JOINTS):
                            addr = ctrl._qpos_addr.get(n)
                            vaddr = ctrl._qvel_addr.get(n)
                            if addr is not None:
                                data.qpos[addr] = ctrl.default[i]
                            if vaddr is not None:
                                data.qvel[vaddr] = 0.0
                step_count += 1
                sim_time += model.opt.timestep
            v.sync()

            # ── Camera preview ─────────────────────────────────────────────
            if preview_renderer is not None and HAS_CV2:
                now = time.time()
                if now - last_prev >= prev_intv:
                    last_prev = now
                    preview_renderer.update_scene(data, camera="cam_left_high")
                    img = preview_renderer.render()
                    _cv2.imshow("cam_left_high", _cv2.cvtColor(img, _cv2.COLOR_RGB2BGR))
                    preview_renderer.update_scene(data, camera="cam_right_wrist")
                    wimg = preview_renderer.render()
                    _cv2.imshow("cam_right_wrist", _cv2.cvtColor(wimg, _cv2.COLOR_RGB2BGR))
                    if _cv2.waitKey(1) & 0xFF == 27:
                        break

    if HAS_CV2:
        try:
            _cv2.destroyAllWindows()
        except Exception:
            pass
    if HAS_PYGAME:
        pygame.quit()
    print("[done]")


if __name__ == "__main__":
    main()
