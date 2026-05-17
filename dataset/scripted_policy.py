#!/usr/bin/env python3
"""Sprint 1.4 — Scripted pick-and-place using the right_reacher ONNX policy.

State machine (right arm only, robot held stationary by walker at zero vel):
  WARMUP    — let physics settle (robot stands still)
  APPROACH  — move palm above target cube (pre-grasp position)
  DESCEND   — lower palm to cube height for grasping
  GRASP     — close grip and stabilize
  LIFT      — raise arm clear of table surface
  TRANSPORT — move horizontally to above yellow box
  LOWER     — descend into yellow box
  PLACE     — open grip, wait for cube to drop
  DONE / FAILED

Saves:
  output/scripted_<cube>_cube.mp4  — head_cam video at 25 fps with phase overlay

Usage:
  uv run python dataset/scripted_policy.py                  # red cube (default)
  uv run python dataset/scripted_policy.py --cube green
  uv run python dataset/scripted_policy.py --cube all       # both cubes
  uv run python dataset/scripted_policy.py --max-steps 3000
"""

import argparse
import json
from enum import Enum, auto
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

REPO_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

RIGHT_ARM_JOINTS = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

ALL_ARM_PATTERNS = ("shoulder", "elbow", "wrist")

FINGER_CLOSED = {
    "right_hand_thumb_0_joint":  0.8,
    "right_hand_thumb_1_joint": -0.9,
    "right_hand_thumb_2_joint": -1.5,
    "right_hand_index_0_joint":  1.4,
    "right_hand_index_1_joint":  1.5,
    "right_hand_middle_0_joint": 1.4,
    "right_hand_middle_1_joint": 1.5,
}

DECIMATION   = 4       # physics sub-steps per control step (50 Hz control)
ARM_MAX_DELTA = 0.012  # max joint change per step (rate limiter from run.py)

BOX_WORLD = np.array([-0.05, 0.13, 0.733], np.float32)  # yellow box world pos

H_ABOVE   = 0.18   # pelvis-frame height: above cube during approach
H_GRASP   = 0.02   # pelvis-frame height: at cube level for grasping
H_LIFT    = 0.22   # pelvis-frame height: lift after grasp
H_PLACE   = 0.04   # pelvis-frame height: inside box for placing

RENDER_W, RENDER_H = 640, 480
RENDER_FPS = 25


class Phase(Enum):
    WARMUP    = auto()
    APPROACH  = auto()
    DESCEND   = auto()
    GRASP     = auto()
    LIFT      = auto()
    TRANSPORT = auto()
    LOWER     = auto()
    PLACE     = auto()
    DONE      = auto()
    FAILED    = auto()


# ── ONNX Policy wrapper ─────────────────────────────────────────────────────
class ONNXPolicy:
    def __init__(self, path: Path):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(
            str(path), opts, providers=["CPUExecutionProvider"]
        )
        self.iname = self.sess.get_inputs()[0].name
        self.oname = self.sess.get_outputs()[0].name

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        if obs.ndim == 1:
            obs = obs[None]
        return self.sess.run([self.oname], {self.iname: obs.astype(np.float32)})[0][0]


# ── Math helpers ────────────────────────────────────────────────────────────
def quat_apply_inv(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, xyz = q[0], q[1:4]
    t = np.cross(xyz, v) * 2.0
    return v - w * t + np.cross(xyz, t)


def set_armature(model, joint_names):
    A5, A7_14, A7_22, A4, A2x5 = (
        0.00360972, 0.01017752, 0.02510192, 0.00425000, 0.00721945,
    )
    for i, n in enumerate(joint_names):
        dof = 6 + i
        if any(p in n for p in ("elbow", "shoulder", "wrist_roll")):
            model.dof_armature[dof] = A5
        elif any(p in n for p in ("hip_pitch", "hip_yaw")) or n == "waist_yaw_joint":
            model.dof_armature[dof] = A7_14
        elif any(p in n for p in ("hip_roll", "knee")):
            model.dof_armature[dof] = A7_22
        elif any(p in n for p in ("wrist_pitch", "wrist_yaw")):
            model.dof_armature[dof] = A4
        elif "ankle" in n or n in ("waist_pitch_joint", "waist_roll_joint"):
            model.dof_armature[dof] = A2x5
        else:
            model.dof_armature[dof] = A5


# ── Episode runner ───────────────────────────────────────────────────────────
def run_episode(cube_name: str, max_ctrl_steps: int = 2000) -> bool:
    """Run one scripted pick-and-place episode. Returns True on success."""

    # ── Load model & config ────────────────────────────────────────────────
    model = mujoco.MjModel.from_xml_path(str(REPO_DIR / "scene.xml"))
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)

    with open(REPO_DIR / "model_config.json") as f:
        cfg = json.load(f)

    joint_names = cfg["joint_names"]
    n_j = len(joint_names)

    default_pos = np.zeros(n_j, np.float32)
    for n, v in cfg["default_joint_pos"].items():
        if n in joint_names:
            default_pos[joint_names.index(n)] = v

    action_scales = np.array([cfg["action_scales"][n] for n in joint_names], np.float32)
    arm_scales    = np.array([cfg["action_scales"][n] for n in RIGHT_ARM_JOINTS], np.float32)
    arm_defaults  = np.array([cfg["default_joint_pos"].get(n, 0.0) for n in RIGHT_ARM_JOINTS], np.float32)

    set_armature(model, joint_names)

    # ── Init pose ──────────────────────────────────────────────────────────
    data.qpos[0] = -0.6
    data.qpos[2] = 0.76
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for n, v in cfg["default_joint_pos"].items():
        if n in joint_names:
            data.qpos[7 + joint_names.index(n)] = v
    mujoco.mj_forward(model, data)

    # ── IDs ────────────────────────────────────────────────────────────────
    j_qposadr  = {n: 7 + i for i, n in enumerate(joint_names)}
    j_qveladr  = {n: 6 + i for i, n in enumerate(joint_names)}
    arm_indices = [joint_names.index(n) for n in RIGHT_ARM_JOINTS]
    all_arm_idx = [
        i for i, n in enumerate(joint_names)
        if any(p in n for p in ALL_ARM_PATTERNS)
    ]
    palm_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_palm")

    cube_jid      = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{cube_name}_cube_joint")
    cube_qposadr  = model.jnt_qposadr[cube_jid]

    actuator_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        for n in joint_names
    ]
    finger_acts = [
        (mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n), v)
        for n, v in FINGER_CLOSED.items()
    ]
    finger_acts = [(aid, v) for aid, v in finger_acts if aid >= 0]

    # ── Policies ───────────────────────────────────────────────────────────
    walker  = ONNXPolicy(REPO_DIR / "walker.onnx")
    reacher = ONNXPolicy(REPO_DIR / "right_reacher.onnx")
    walker(np.zeros(99,  np.float32))
    reacher(np.zeros(36, np.float32))

    # ── Renderer ───────────────────────────────────────────────────────────
    renderer = mujoco.Renderer(model, RENDER_H, RENDER_W)
    frames: list[np.ndarray] = []

    # ── Controller state ───────────────────────────────────────────────────
    last_action     = np.zeros(n_j, np.float32)
    last_arm_action = np.zeros(7,   np.float32)
    last_arm_target = None
    reach_target    = np.array([0.3, -0.2, 0.2], np.float32)
    reach_orient    = np.zeros(3, np.float32)
    grip_closed     = False
    phase           = Phase.WARMUP
    phase_step      = 0
    success         = False

    # ── Inline helpers ─────────────────────────────────────────────────────
    def get_pelvis():
        return data.qpos[:3].copy(), data.qpos[3:7].copy()

    def world_to_pf(wpos):
        p, q = get_pelvis()
        return quat_apply_inv(q, wpos - p)

    def get_proj_gravity():
        _, q = get_pelvis()
        return quat_apply_inv(q, np.array([0.0, 0.0, -1.0]))

    def get_jpos():
        return np.array(
            [data.qpos[j_qposadr[n]] - default_pos[i] for i, n in enumerate(joint_names)],
            np.float32,
        )

    def get_jvel():
        return np.array([data.qvel[j_qveladr[n]] for n in joint_names], np.float32)

    def get_arm_pos():
        return np.array(
            [data.qpos[j_qposadr[n]] - arm_defaults[i] for i, n in enumerate(RIGHT_ARM_JOINTS)],
            np.float32,
        )

    def get_arm_vel():
        return np.array([data.qvel[j_qveladr[n]] for n in RIGHT_ARM_JOINTS], np.float32)

    def get_palm_pos_pf():
        p, q = get_pelvis()
        return quat_apply_inv(q, data.site_xpos[palm_sid] - p)

    def get_palm_orient_pf():
        mat = data.site_xmat[palm_sid].reshape(3, 3)
        pq = np.zeros(4)
        mujoco.mju_mat2Quat(pq, mat.flatten())
        _, pelvis_q = get_pelvis()
        pi = np.array([pelvis_q[0], -pelvis_q[1], -pelvis_q[2], -pelvis_q[3]])
        w1, x1, y1, z1 = pi
        w2, x2, y2, z2 = pq
        r = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])
        w, x, y, z = r
        return np.array([
            np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)),
            np.arcsin(np.clip(2*(w*y - z*x), -1.0, 1.0)),
            np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)),
        ], np.float32)

    def apply_ctrl(target):
        for i, aid in enumerate(actuator_ids):
            if aid >= 0:
                data.ctrl[aid] = target[i]
        for aid, cval in finger_acts:
            data.ctrl[aid] = cval if grip_closed else 0.0

    # ── Main loop ──────────────────────────────────────────────────────────
    print(f"  Phase: WARMUP (stabilising robot...)")
    ctrl_step = 0

    while ctrl_step < max_ctrl_steps and phase not in (Phase.DONE, Phase.FAILED):

        for _ in range(DECIMATION):
            mujoco.mj_step(model, data)

        proj_g = get_proj_gravity()
        jpos   = get_jpos()
        jvel   = get_jvel()
        p, q   = get_pelvis()

        # Walker obs — zero velocity command keeps robot stationary
        lin_vel = quat_apply_inv(q, data.qvel[:3])
        walker_obs = np.concatenate([
            lin_vel, data.qvel[3:6], proj_g,
            jpos, jvel, last_action,
            np.zeros(3, np.float32),
        ])
        walker_act = walker(walker_obs)
        target     = default_pos + walker_act * action_scales

        # All arm joints reset (reacher will override right arm below)
        for idx in all_arm_idx:
            target[idx] = default_pos[idx]

        # ── State machine ────────────────────────────────────────────────
        cube_world = data.qpos[cube_qposadr:cube_qposadr + 3].astype(np.float32)
        cube_pf    = world_to_pf(cube_world)
        box_pf     = world_to_pf(BOX_WORLD)
        palm_pf    = get_palm_pos_pf()

        d_palm_cube    = float(np.linalg.norm(palm_pf - cube_pf))
        d_palm_box_xy  = float(np.linalg.norm(palm_pf[:2] - box_pf[:2]))

        if phase == Phase.WARMUP:
            reach_target[:] = [0.3, -0.2, 0.2]
            if phase_step >= 80:
                phase = Phase.APPROACH
                phase_step = 0
                print(f"  Phase: APPROACH  cube_pf={cube_pf.round(3)}")

        elif phase == Phase.APPROACH:
            reach_target[:] = [cube_pf[0], cube_pf[1], cube_pf[2] + H_ABOVE]
            if d_palm_cube < 0.13 or phase_step > 200:
                phase = Phase.DESCEND
                phase_step = 0
                print(f"  Phase: DESCEND   dist={d_palm_cube:.3f}m")

        elif phase == Phase.DESCEND:
            reach_target[:] = [cube_pf[0], cube_pf[1], cube_pf[2] + H_GRASP]
            if d_palm_cube < 0.055 or phase_step > 200:
                phase = Phase.GRASP
                phase_step = 0
                grip_closed = True
                print(f"  Phase: GRASP     dist={d_palm_cube:.3f}m")

        elif phase == Phase.GRASP:
            reach_target[:] = [cube_pf[0], cube_pf[1], cube_pf[2] + H_GRASP]
            if phase_step >= 50:
                phase = Phase.LIFT
                phase_step = 0
                print("  Phase: LIFT")

        elif phase == Phase.LIFT:
            reach_target[:] = [cube_pf[0], cube_pf[1], H_LIFT]
            if palm_pf[2] > H_LIFT - 0.05 or phase_step > 160:
                phase = Phase.TRANSPORT
                phase_step = 0
                print(f"  Phase: TRANSPORT palm_z={palm_pf[2]:.3f}")

        elif phase == Phase.TRANSPORT:
            reach_target[:] = [box_pf[0], box_pf[1], H_LIFT]
            if d_palm_box_xy < 0.08 or phase_step > 280:
                phase = Phase.LOWER
                phase_step = 0
                print(f"  Phase: LOWER     dist_xy={d_palm_box_xy:.3f}m")

        elif phase == Phase.LOWER:
            reach_target[:] = [box_pf[0], box_pf[1], box_pf[2] + H_PLACE]
            if palm_pf[2] < box_pf[2] + H_PLACE + 0.06 or phase_step > 160:
                phase = Phase.PLACE
                phase_step = 0
                grip_closed = False
                print(f"  Phase: PLACE     palm_z={palm_pf[2]:.3f}")

        elif phase == Phase.PLACE:
            reach_target[:] = [box_pf[0], box_pf[1], H_LIFT]
            if phase_step >= 100:
                cube_in_box = (
                    abs(cube_world[0] - BOX_WORLD[0]) < 0.12
                    and abs(cube_world[1] - BOX_WORLD[1]) < 0.12
                    and cube_world[2] < BOX_WORLD[2] + 0.12
                )
                success = bool(cube_in_box)
                phase   = Phase.DONE
                status  = "SUCCESS ✓" if success else "FAILED ✗"
                print(f"  Phase: DONE      {status}")
                print(f"           cube={cube_world.round(3)}  box={BOX_WORLD.round(3)}")

        # ── Reacher overlay (right arm) ──────────────────────────────────
        arm_pos    = get_arm_pos()
        arm_vel    = get_arm_vel()
        palm_orient = get_palm_orient_pf()

        reacher_obs = np.concatenate([
            reach_target, reach_orient,
            palm_pf, palm_orient,
            arm_pos, arm_vel,
            last_arm_action, proj_g,
        ])
        arm_act    = reacher(reacher_obs)
        arm_target = arm_defaults + arm_act * arm_scales

        if last_arm_target is not None:
            delta      = np.clip(arm_target - last_arm_target, -ARM_MAX_DELTA, ARM_MAX_DELTA)
            arm_target = last_arm_target + delta
        last_arm_target = arm_target.copy()

        for i, idx in enumerate(arm_indices):
            target[idx] = arm_target[i]

        last_arm_action = arm_act.copy()
        last_action     = walker_act.copy()

        apply_ctrl(target)

        # ── Render every 2nd ctrl step → ~25 fps ────────────────────────
        if ctrl_step % 2 == 0:
            renderer.update_scene(data, camera="head_cam")
            frame = renderer.render().copy()
            # Phase overlay (requires opencv — graceful fallback if missing)
            try:
                import cv2
                label = f"Phase: {phase.name}  step:{ctrl_step:04d}"
                cv2.putText(frame, label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 80), 2)
                grip_lbl = "GRIP: CLOSED" if grip_closed else "GRIP: open"
                cv2.putText(frame, grip_lbl, (10, 58),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 255, 80) if grip_closed else (200, 200, 200), 2)
            except ImportError:
                pass
            frames.append(frame)

        ctrl_step  += 1
        phase_step += 1

    # ── Save video ─────────────────────────────────────────────────────────
    out_path = OUTPUT_DIR / f"scripted_{cube_name}_cube.mp4"
    if frames:
        try:
            import cv2
            h, w = frames[0].shape[:2]
            writer = cv2.VideoWriter(
                str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), RENDER_FPS, (w, h)
            )
            for f in frames:
                writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            writer.release()
            duration = len(frames) / RENDER_FPS
            print(f"  Video → {out_path}  ({len(frames)} frames, {duration:.1f}s)")
        except ImportError:
            print("  [WARN] opencv not found — skipping video. Run: uv add opencv-python")
        except Exception as e:
            print(f"  [WARN] Video save failed: {e}")
    else:
        print("  [WARN] No frames rendered.")

    return success


# ── Entry point ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Sprint 1 — Scripted pick-and-place")
    parser.add_argument(
        "--cube", choices=["red", "green", "all"], default="red",
        help="Which cube to pick (default: red)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=2000,
        help="Max control steps per episode (50 Hz, default=2000 ≈ 40s)",
    )
    args = parser.parse_args()

    cubes   = ["red", "green"] if args.cube == "all" else [args.cube]
    results = {}

    for cube in cubes:
        print(f"\n{'='*52}")
        print(f"  Cube: {cube.upper()}  →  Yellow Box")
        print(f"{'='*52}")
        results[cube] = run_episode(cube, args.max_steps)

    print(f"\n{'='*52}")
    print("  Results:")
    for cube, ok in results.items():
        print(f"    {cube:5s}:  {'SUCCESS ✓' if ok else 'FAILED  ✗'}")
    print(f"{'='*52}\n")


if __name__ == "__main__":
    main()
