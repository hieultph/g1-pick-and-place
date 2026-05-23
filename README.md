# g1-manipulation-challenge

MuJoCo simulation environment for closed-loop VLA inference on the **Unitree G1 humanoid robot**. The robot receives camera frames + proprioception, sends them to a remote GR00T policy server over ZMQ, and applies the predicted joint actions to pick and place objects.

Part of the [G1 pick-and-place VLA project](../README.md). The companion inference server lives in [`../Isaac-GR00T`](../Isaac-GR00T/README_PERSONAL.md).

---

## Quick Start

```bash
# Install dependencies
uv sync

# 1. Start the GR00T policy server (on a GPU machine — see ../Isaac-GR00T)
# 2. Open a tunnel if the server is remote
cloudflared access tcp --hostname <your-tunnel> --url 127.0.0.1:5555 &

# 3. Run closed-loop VLA inference
uv run python vla_run.py \
    --prompt "pick the red cube and put into yellow container" \
    --host 127.0.0.1 --port 5555

# Run without grip assist (physics-only grasping)
uv run python vla_run.py --prompt "..." --no-grip-assist

# Save every server observation as .npz + .json + .png for debugging
uv run python vla_run.py --prompt "..." --save-obs-dir ./saved_obs

# Interactive viewer (no VLA server needed)
uv run python main.py
```

---

## File Structure

```
g1-manipulation-challenge/
├── vla_run.py          # Main closed-loop inference script
├── teleoperate.py      # Keyboard + gamepad joint control + episode recording
├── main.py             # Interactive MuJoCo viewer for manual exploration
├── view_scene.py       # Passive scene viewer (no control)
├── scene.xml           # MuJoCo scene: table, red cube, green cube, yellow box
├── g1.xml              # G1 robot MJCF (29 body DOF + 14 finger DOF + cameras)
├── model_config.json   # Joint names, default poses, action scales, obs stats
└── assets/             # Robot mesh files (.obj / .STL)
```

---

## Scene

The robot spawns at x = −0.60 facing +X. All coordinates are in world frame.

| Object | World pos (x, y, z) | Notes |
|--------|-------------------|-------|
| red_cube | (−0.13, −0.17, 0.850) | 4×4×8 cm, dynamic freejoint |
| green_cube | (−0.13, +0.20, 0.850) | 4×4×8 cm, dynamic freejoint |
| yellow_box | (−0.20, 0.00, 0.84) | 20×20 cm open-top container, static |

---

## vla_run.py — Architecture

```
MuJoCo sim (200 Hz)
    │
    ├── VLAController (50 Hz)
    │     ├── render cam_left_high + cam_right_high  → (1,1,480,640,3)
    │     ├── read joint state (left_arm, right_arm, left_hand, right_hand)
    │     └── ZMQ → GR00T server → action chunk (40 steps × 28 DOF)
    │
    ├── GripAssist (200 Hz)      ← kinematic weld when index fingertip near cube
    │
    └── data.ctrl write          ← position control targets applied to MuJoCo
```

**Control modes:**

| Flag | Behaviour |
|------|-----------|
| *(default)* | `freeze_non_arm=True` — base + legs + waist pinned; only arms + hands driven by VLA |
| `--no-hands` | VLA arms only, hand joints frozen |
| `--no-grip-assist` | No kinematic grip weld, physics-only contact |

---

## Grip Assist

`GripAssist` kinematically attaches a cube to the robot's hand when a grip is detected, bypassing MuJoCo contact physics for a reliable grasp.

**How it works:**

1. Each step, computes the world-space position of the **index-1 fingertip** (`data.xpos[index1_body] + data.xmat[index1_body] @ [0.028, 0, 0]`)
2. If `index_closure ≥ GRIP_THRESH` **and** `dist(index_tip, cube_center) < GRIP_DIST` → **latch**
3. At latch: cube snaps to the index fingertip; its position is stored as a palm-frame offset for tracking
4. Every step while latched: cube teleported to `palm_pos + palm_mat @ rel_pos`
5. Release when `index_closure < OPEN_THRESH` for `MIN_GRIP_STEPS` steps

**Thresholds (tunable as class attributes):**

```python
GRIP_DIST   = 0.05   # metres — index tip must be this close to cube center
GRIP_THRESH = 0.30   # index closure score to trigger latch
OPEN_THRESH = 0.08   # closure score to release
MIN_GRIP_STEPS = 20  # debounce steps before release is allowed
```

The closure score is `joint_value × ±1` (sign depends on hand) so it is always ≥ 0 when curling.

---

## Wrist Camera 6-DOF Control

While the viewer is open, you can nudge the wrist cameras to align with your fine-tuned model's training distribution:

| Key | Action |
|-----|--------|
| `1` / `2` | Select left / right wrist cam |
| `Q`/`A` | pos +x / −x (5 mm/step) |
| `W`/`S` | pos +y / −y |
| `E`/`D` | pos +z / −z |
| `←`/`→` | yaw +/− (5°/step) |
| `↑`/`↓` | pitch +/− |
| `,`/`.` | roll +/− |

After adjusting, the console prints the XML `pos=` and `xyaxes=` values to copy back into `g1.xml`.

---

## VLA Observation Format

Each inference request sent to the GR00T server:

```python
{
    "video": {
        "cam_left_high":  np.ndarray(1, 1, 480, 640, 3),   # (batch, time, H, W, C)
        "cam_right_high": np.ndarray(1, 1, 480, 640, 3),
    },
    "state": {
        "left_arm":   np.ndarray(1, 1, 7),   # shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw
        "right_arm":  np.ndarray(1, 1, 7),
        "left_hand":  np.ndarray(1, 1, 7),   # thumb 0/1/2, middle 0/1, index 0/1
        "right_hand": np.ndarray(1, 1, 7),   # thumb 0/1/2, index 0/1, middle 0/1
    },
    "language": {
        "annotation.human.task_description": [["pick the red cube..."]]
    },
}
```

Response: `{ "left_arm": (40, 7), "right_arm": (40, 7), "left_hand": (40, 7), "right_hand": (40, 7) }` — absolute joint targets, applied in `EXEC_HORIZON = 40` step chunks.

---

## Robot Joint Layout

```
Body joints (29D, driven by walker or frozen):
  Legs (12): hip pitch/roll/yaw × 2, knee × 2, ankle pitch/roll × 2
  Waist (3): yaw, roll, pitch
  Arms (14): shoulder pitch/roll/yaw × 2, elbow × 2, wrist roll/pitch/yaw × 2

Finger joints (14D, driven by VLA):
  Left hand:  thumb 0/1/2, middle 0/1, index 0/1   (7D)
  Right hand: thumb 0/1/2, index 0/1, middle 0/1   (7D)
```

Joint order in `model_config.json` matches the walker policy's expected observation exactly.

---

## Dataset Initial Pose (`DATASET_INIT_QPOS`)

The robot is initialised to the first-frame joint configuration of the real-robot training dataset. This ensures the camera field-of-view and arm position at inference matches what the VLA saw during training.

Left arm values are the bilateral mirror of right arm values:
- **Y-axis joints** (shoulder pitch, elbow, wrist pitch): same sign
- **X/Z-axis joints** (shoulder roll, yaw; wrist roll, yaw): negated

---

## HzMonitor

Prints throughput and latency every 5 s:

```
[Hz] ctrl=50.0Hz | infer=0.9Hz | sim=200.0Hz | infer_latency=1124ms(avg) 1201ms(max)
```

- `ctrl` — VLA controller step rate (target 50 Hz)
- `sim` — MuJoCo physics step rate (target 200 Hz)
- `infer` — New chunk requests per second
- `infer_latency` — Round-trip ZMQ + model inference time

---

## Dependencies

Managed via `uv` (see `pyproject.toml`):

- `mujoco ≥ 3.8.1`
- `onnxruntime` — walker policy inference
- `pyzmq` + `msgpack-numpy` — policy server communication
- `opencv-python` — camera preview windows
- `numpy`, `Pillow`
