# 状态广播协议

`pose_grasp_server.py` 通过 JSONL/TCP 以 10 Hz 广播 `state`。执行 `get_state` 时，
返回消息的 `id` 会回填请求 id。以下以当前 V50 结构为准。

## 1. 顶层字段

| 字段 | 含义 |
|---|---|
| `type` | 固定为 `state` |
| `id` | 主动广播为 `null`，`get_state` 响应为请求 id |
| `busy` | 任意单臂/双臂轨迹或自动任务运行中 |
| `active_command` | `auto_clamp`、`auto_grasp` 或当前轨迹命令 |
| `auto_grasp_phase` | 单臂自动抓取阶段，无任务为 `null` |
| `auto_clamp_phase` | 双臂夹箱阶段，无任务为 `null` |
| `auto_clamp` | 双臂夹箱上下文摘要，无任务为 `null` |
| `arm_joints` | 10 个臂关节实际位置，单位 rad |
| `hand_joints` | 12 个 active 手关节实际位置，单位 rad |
| `ee` | 左右 hand-base 的 base/world 位姿 |
| `object_pose` | 物体世界位姿 |
| `contact_forces` | broad、filtered 和夹箱聚合力 |

四元数统一为 `wxyz`。

## 2. 自动夹箱字段

任务运行时 `auto_clamp` 包含：

```jsonc
{
  "targets": {"left": {}, "right": {}},
  "forces": {"left": {}, "right": {}},
  "clamp_alpha": {"left": 0.0, "right": 0.0},
  "stable_frames": {"left": 0, "right": 0},
  "contact_seen": {"left": false, "right": false},
  "contact_alpha": {"left": null, "right": null},
  "lift_force_drop_frames": {"left": 0, "right": 0},
  "lift_verify_frames": 0,
  "broad_force_limit_frames": 0,
  "fingertip_min_world_z": {"left": 0.0, "right": 0.0},
  "metrics": {"left": {}, "right": {}, "lift": {}}
}
```

注意：实时 `state.auto_clamp` 当前不直接暴露 `BimanualLiftState` 的
`progress_m/squeeze_offset_m/center_offset_m`。这些字段在录制目录的
`sensor_data.jsonl.bimanual_lift` 中输出。CLI 的常规 `state` 也只打印力的摘要。
`lift_force_drop_frames` 是旧双臂抬升路径留下的字段；V50 统一抬升实际使用
`BimanualLiftState.recovery_frames`，后者同样只在录制样本中可见。

## 3. 末端和物体位姿

```jsonc
{
  "ee": {
    "left": {
      "base": {"xyz": [0, 0, 0], "quat_wxyz": [1, 0, 0, 0]},
      "world": {"xyz": [0, 0, 0], "quat_wxyz": [1, 0, 0, 0]}
    },
    "right": {}
  },
  "object_pose": {
    "xyz": [0.30, 0.0, 0.82],
    "quat_wxyz": [1.0, 0.0, 0.0, 0.0]
  }
}
```

`ee.base` 由 `world_to_base()` 转换，`object_pose` 保持世界坐标。

## 4. 触觉结构

```jsonc
{
  "contact_forces": {
    "contact_hands": {
      "contact_body_count": 0,
      "max_force_n": 0.0,
      "by_body_w": {
        "lh_hand_base_link": {"fx": 0, "fy": 0, "fz": 0, "norm": 0}
      }
    },
    "fingertips_object_n": {
      "lh_thumb_distal": 0.0
    },
    "fingertips_object_raw_w": {
      "lh_thumb_distal": {
        "fx": 0, "fy": 0, "fz": 0, "norm": 0, "matrix_contacts": 1
      }
    },
    "palms_object_raw_w": {
      "left": {"fx": 0, "fy": 0, "fz": 0, "norm": 0},
      "right": {"fx": 0, "fy": 0, "fz": 0, "norm": 0}
    },
    "palms_object_n": {"left": 0.0, "right": 0.0},
    "auto_clamp_hands": {
      "left": {
        "object_axis_y_n": 0.0,
        "palm_axis_y_n": 0.0,
        "thumb_axis_y_n": 0.0,
        "non_thumb_axis_y_n": 0.0,
        "object_contact_body_count": 0,
        "object_by_body_axis_y_n": {},
        "object_by_body_norm_n": {}
      },
      "right": {}
    }
  }
}
```

`fingertips_object_*` 是单臂自动抓取和兼容展示使用的 10 指尖视图。
`auto_clamp_hands` 才是双臂夹箱使用的每手 12-body 聚合视图。`object_axis_y_n` 为
沿夹持轴的单向 Object-filtered 压缩力，不等同于 `norm`。

## 5. 数据来源

| 字段 | 数据来源 |
|---|---|
| `arm_joints` / `hand_joints` | `robot.data.joint_pos[0]` |
| `ee.{arm}.world` | `robot.data.body_pos_w`、`body_quat_w` |
| `ee.{arm}.base` | `world_to_base()` |
| `object_pose` | `object_asset.data.root_pos_w/root_quat_w` |
| `contact_hands` | 24-body `net_forces_w[0]` |
| `fingertips_object_*` | 10 个指尖 filtered `force_matrix_w` |
| `palms_object_*` | 2 个掌心 filtered `force_matrix_w` |
| `auto_clamp_hands` | 全部 24 个 filtered sensor，按左右各 12 聚合 |

## 6. CLI 显示

CLI `state` 当前打印：

```text
Status: BUSY (auto_clamp) [MOVE_LIFT]
LEFT/RIGHT hand base + world pose
Object world pose
Contact: <body count>, max=<N>
Fingertip->Object L/R
Hand->Object clamp-axis total L/R and palm L/R
Hand L/R active joint values
```

CLI 不会打印 JSON 中所有逐 body 字段。需要完整诊断时使用 `--verbose` 查看消息，或
直接读取 `sensor_data.jsonl`。

## 7. 录制样本扩展

启用 `--record-dir` 后，每个 `sensor_data.jsonl` 样本额外包含：

- `frame`、`phase`、`clamp_alpha`；
- `hand_base_pose`、`object_pose`；
- 完整 `contact_forces`；
- `bimanual_lift.filtered_left_n/right_n`；
- `progress_m`、`squeeze_offset_m`、`center_offset_m`；
- `recovery_frames`、`stable_frames` 和最近一次 `command`。

因此复盘统一抬升必须使用录制样本，而不能只依赖 10 Hz 的 CLI 摘要。
