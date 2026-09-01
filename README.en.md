# R1-o6 Dual-Arm Box Grasping

This project validates a dual-arm grasp-and-lift workflow for the R1-o6 robot in Isaac Lab. The robot approaches an object from both sides, closes its hands using tactile feedback, and lifts the object from the table. The current profiles support a medium corrugated shipping box and an EPS foam box; this side-grasp method has been validated for both types.

[中文说明](README.md)

Code license: [Apache-2.0](LICENSE)

## Results

The images below show the final frames from successful runs for the two supported object types.

| Medium shipping box | EPS foam box |
|:---:|:---:|
| ![Medium shipping box lifted](docs/media/courier_box_lifted.png) | ![EPS foam box lifted](docs/media/foam_box_lifted.png) |

| Object | Baseline | Mass | Size XYZ | Initial X | Target force per hand | Lift command | Result |
|---|---|---:|---:|---:|---:|---:|---|
| `courier_box_m` | `rootback3cm-force10p5n-v41` | 1.0 kg | 0.30 x 0.22 x 0.20 m | 0.270 m | 10.5 N | 0.16 m | `done(status=lifted)` |
| `foam_box` | `flat-palm-v19` | 0.5 kg | 0.35 x 0.24 x 0.25 m | 0.295 m | 11.0 N | 0.20 m | `done(status=lifted)`; measured rise 0.1989 m |

Both profiles use `robot_root_x_offset=-0.03 m` and `clamp_x_offset=-0.03 m`. A run is counted as successful only when the server returns `done(status=lifted)` and the object rises by at least 0.15 m. The foam box can rotate by about 30–35 degrees near the end of the lift.

## What the project does

After receiving a `clamp` command, the system handles the full sequence:

- retracts the fingers and raises the arms laterally to clear the table;
- computes pre-grasp and clamp poses on both sides of the object;
- adjusts the closing motion from tactile channels that only count object contact;
- waits for stable contact on both hands before lifting;
- reports a successful lift or a protective failure from contact state and object motion.

Main entry points:

- `auto_clamp.py`: pure-Python geometry, force control, object profiles, and bimanual lift coordination;
- `pose_grasp_server.py`: Isaac Lab execution service with FK search, tactile sensing, and joint command output;
- `pose_grasp_cli.py`: JSONL/TCP client for `state`, `home`, `clamp`, and `stop`.

The validated workflow runs in Isaac Lab simulation. Real-robot experiments are outside this repository.

## Implementation overview

The state machine is:

```text
PREPARE_HANDS -> SIDE_RAISE -> OPEN_HANDS -> VERIFY_CLEARANCE
-> SOLVE_PREGRASP -> MOVE_PREGRASP -> VERIFY_PREGRASP
-> SOLVE_CLAMP -> FORCE_CLAMP -> MOVE_LIFT -> VERIFY_LIFT
```

Pre-grasp poses are computed from the object center and width, with a hand-base-to-palm compensation. The fingers stay curled during the approach and open only after the clearance check. During clamping, the controller accumulates inward compression along the object's Y axis; tangential collisions and reverse forces are not treated as useful grip force. Lift progress is limited by bilateral contact stability, arm progress difference, and command lead over measured motion.

Detailed geometry, sensor definitions, and state transitions are documented in [docs/auto_clamp.md](docs/auto_clamp.md).

## Quick start

Requirements: Linux, an NVIDIA GPU, Isaac Lab 2.x / Isaac Sim, a compatible cuRobo installation, and Python 3.10+. Install Isaac Lab and cuRobo using their own distribution instructions; this project does not provide a generic `requirements.txt`.

### Prepare configurations and assets

```bash
PROJECT_DIR="/absolute/path/to/r1_o6_dual_box_grasp"
ISAACLAB_DIR="/absolute/path/to/IsaacLab"

python3 "$PROJECT_DIR/scripts/prepare_curobo_configs.py"

cd "$ISAACLAB_DIR"
./isaaclab.sh -p "$PROJECT_DIR/scripts/convert_r1_o6_urdf.py" --force
```

### Start the service

The following command uses the validated medium-box baseline:

```bash
cd "$ISAACLAB_DIR"
LIVESTREAM=2 ./isaaclab.sh -p "$PROJECT_DIR/pose_grasp_server.py" \
  --headless --livestream 2 --enable_cameras \
  --object courier_box_m \
  --robot-config "$PROJECT_DIR/.runtime/configs/r1_o6.yml" \
  --left-config "$PROJECT_DIR/.runtime/configs/r1_o6_left.yml" \
  --right-config "$PROJECT_DIR/.runtime/configs/r1_o6_right.yml" \
  --robot-root-x-offset-m -0.03 \
  --clamp-x-offset-m -0.03 \
  --clamp-force-target-n 10.5
```

For the foam box, use `--object foam_box` and `--clamp-force-target-n 11.0`. The validated initial X positions are 0.270 m and 0.295 m. Full commands, recording options, and CLI details are in [docs/OPERATIONS.md](docs/OPERATIONS.md).

The server listens only on `127.0.0.1` by default. Cross-machine access must be enabled explicitly with `--host <trusted-interface-address>`. The JSONL/TCP control interface has no authentication or TLS and must not be exposed to the public Internet or an untrusted shared network.

### Connect and run

In another terminal:

```bash
python3 "$PROJECT_DIR/pose_grasp_cli.py" --host 127.0.0.1 --port 5560
```

Use `state` to inspect the current phase, `home` to return to the home pose, and `clamp` to start the task. A successful run ends with:

```text
Auto-clamp COMPLETE (lifted)
done status=lifted
```

## Repository layout

```text
auto_clamp.py                 Geometry, force control, and bimanual lift coordination
pose_grasp_server.py          Isaac Lab service and state machine
pose_grasp_cli.py             JSONL/TCP operation client
delivery_objects_cfg.py       Box dimensions and physical parameters
r1_o6_scene_cfg.py            Robot, table, camera, and tactile scene
configs/                      cuRobo templates and validated baselines
scripts/                      Configuration preparation and URDF -> USD conversion
tests/                        Offline regression and simulator smoke tests
docs/                         Operations, algorithm, asset, and protocol notes
R1-o6.urdf, meshes/           Robot description and STL meshes
```

## Local checks

```bash
python3 -m unittest discover -s tests -p 'test_*.py' -v
python3 -m compileall -q \
  auto_clamp.py pose_grasp_server.py pose_grasp_cli.py \
  r1_o6_scene_cfg.py delivery_objects_cfg.py scripts tests
```

These checks cover configuration, geometry, and control rules without loading Isaac Lab. They do not replace simulator contact-dynamics testing.

## Notes

The cuRobo world model does not include the table; table safety currently relies on the state machine's fingertip-clearance check. `stop` freezes the arms but does not open the fingers. Recording mode relaxes some timeout and bimanual progress-difference limits, so recorded and non-recorded runs should be compared separately.

Validated server parameters and evidence locations are recorded in [`configs/validated_runs.json`](configs/validated_runs.json). `auto_clamp.py` matches the server version by SHA-256, and the three cuRobo YAML files match the server configuration after path normalization. The `server.log` for the successful foam-box run was removed during server cleanup; the remaining evidence is `sensor_data.jsonl`, the video, and adjacent run logs.
