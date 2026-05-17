#!/usr/bin/env python3
"""Minimal MuJoCo viewer for scene.xml.

Loads scene.xml, initialises the robot at its spawn pose, and opens the
MuJoCo passive viewer with real-time physics.  No ONNX policies required.
Joint PD controllers hold the robot at its default pose so it stands still.

Usage:
  uv run python view_scene.py
  uv run python view_scene.py --scene tabletop_scene_fixed.xml
  uv run python view_scene.py --scene sim/tabletop_scene.xml

Controls (MuJoCo viewer built-in):
  Left-drag          Rotate camera
  Right-drag / Scroll  Pan / zoom
  Double-click body  Select & inspect
  Space              Pause / unpause simulation
  Backspace          Reset simulation
  Ctrl+R             Toggle recording
"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from mujoco import viewer

REPO_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="View scene.xml in MuJoCo viewer")
    parser.add_argument(
        "--scene", default="scene.xml",
        help="Scene XML file relative to repo root (default: scene.xml)",
    )
    args = parser.parse_args()

    scene_path = REPO_DIR / args.scene
    print(f"Loading: {scene_path}")

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    model.opt.timestep = 0.005
    data  = mujoco.MjData(model)

    # Load default joint positions from model_config.json
    cfg_path = REPO_DIR / "model_config.json"
    joint_names: list[str] = []
    default_pos = np.zeros(model.nv, np.float32)

    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        joint_names = cfg.get("joint_names", [])
        for name, val in cfg.get("default_joint_pos", {}).items():
            if name in joint_names:
                idx = joint_names.index(name)
                default_pos[idx] = val

    # Set robot spawn pose (pelvis at -0.6, 0, 0.76 — upright)
    data.qpos[0] = -0.6
    data.qpos[2] =  0.76
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for name, val in cfg.get("default_joint_pos", {}).items():
        if name in joint_names:
            data.qpos[7 + joint_names.index(name)] = val
    mujoco.mj_forward(model, data)

    # Build actuator id → default position map for PD hold
    actuator_default: list[tuple[int, float]] = []
    for i, name in enumerate(joint_names):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if aid >= 0:
            actuator_default.append((aid, float(default_pos[i])))

    decimation = 4          # physics steps per control step (50 Hz)
    ctrl_step  = 0
    sim_time   = 0.0

    print("Opening MuJoCo viewer  (close window to quit)")
    print("  Robot is held at default pose by PD controllers.")
    print("  Double-click any body to inspect it.\n")

    with viewer.launch_passive(model, data) as v:
        t0 = time.time()
        while v.is_running():
            wall = time.time() - t0

            # Cap physics catchup to avoid spiral-of-death after pause
            if wall - sim_time > 0.05:
                sim_time = wall - 0.05

            while sim_time < wall:
                if ctrl_step % decimation == 0:
                    # Hold all robot joints at default (no policy needed)
                    for aid, pos in actuator_default:
                        data.ctrl[aid] = pos

                mujoco.mj_step(model, data)
                ctrl_step += 1
                sim_time  += model.opt.timestep

            v.sync()


if __name__ == "__main__":
    main()
