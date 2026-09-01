# O6 触觉传感器技术文档

> 本文描述当前 `r1_o6_scene_cfg.py` 和 `pose_grasp_server.py` 的仿真实现。真实 O6
> 使用分布式电容阵列；Isaac Lab 以逐刚体 ContactReporter 近似，不等价于真实阵列的
> 空间分辨率。

## 1. 当前布局

| 区域 | 双手 body 数 | 示例 |
|---|---:|---|
| 指尖 distal | 10 | `lh_index_distal` |
| proximal/掌指结构 | 10 | `lh_index_proximal`、`lh_thumb_metacarpals` |
| 掌心 | 2 | `lh_hand_base_link` |
| 拇指 CMC base | 2 | `lh_thumb_metacarpals_base2` |
| 合计 | 24 | 每手 12 |

模块级常量：

```python
from r1_o6_scene_cfg import (
    O6_FINGERTIP_BODIES,
    O6_PROXIMAL_BODIES,
    O6_PALM_BODIES,
    O6_THUMB_BASE_BODIES,
    O6_ALL_TACTILE_BODIES,
    O6_OBJECT_CONTACT_SENSOR_NAMES,
)
```

## 2. 两层传感器

### 2.1 Broad sensor

`scene["contact_hands"]` 是一个覆盖全部 24 body 的未过滤传感器：

```python
contact_hands = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/" + _O6_TACTILE_BODY_REGEX,
    update_period=0.0,
    history_length=6,
    track_air_time=True,
    force_threshold=0.05,
    debug_vis=False,
)
```

- `net_forces_w` shape 为 `(num_envs, 24, 3)`；
- 包含物体、桌面和其它外部碰撞源的合力，不能区分来源；
- 用于全局碰撞诊断、单 body 最大力和连续过载保护；
- `current_air_time`/`last_air_time` 可用于接触时序。

### 2.2 Object-filtered sensors

`O6_OBJECT_CONTACT_SENSOR_NAMES` 为全部 24 个 body 一一映射场景 sensor 名。每个
sensor 只覆盖一个 body，并只过滤 `{ENV_REGEX_NS}/Object`：

```python
def _object_contact_sensor_cfg(body: str) -> ContactSensorCfg:
    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{body}",
        update_period=0.0,
        history_length=3,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        force_threshold=0.05,
    )
```

每个 `force_matrix_w` 通常为 `(num_envs, 1, 1, 3)`。当前总数为 24 个 filtered
sensor，不是旧版的 10 个指尖或 10 指尖加 2 掌心。

命名示例：

| Body | 场景键名 |
|---|---|
| `lh_hand_base_link` | `lh_palm_object_contact` |
| `lh_thumb_metacarpals_base2` | `lh_thumb_base_object_contact` |
| `lh_thumb_metacarpals` | `lh_thumb_metacarpals_object_contact` |
| `lh_thumb_distal` | `lh_thumb_object_contact` |
| `lh_index_proximal` | `lh_index_proximal_object_contact` |
| `lh_index_distal` | `lh_index_object_contact` |

右手以 `rh_` 对称，middle/ring/pinky 同理。

## 3. 双臂夹箱的力定义

`pose_grasp_server.py::_auto_clamp_forces()` 读取每侧 12 个 Object-filtered sensor。
控制量不是 3D 模长之和，而是夹持 Y 轴上的单向压缩分量：

- 左手保留正 `Fy`，负值截为 0；
- 右手保留负 `Fy` 的绝对压缩量，正值截为 0；
- 12 个 body 的有效分量求和得到 `object_axis_y_n`；
- broad sensor 只用于诊断和安全，不作为 10 N 目标的主输入。

每侧同时输出：

| 字段 | 含义 |
|---|---|
| `object_axis_y_n` | 全部 12 body 的 Object-filtered 单向法向力 |
| `palm_axis_y_n` | 掌心单向法向力 |
| `thumb_axis_y_n` | 含 thumb 名称的 body 法向力总和 |
| `non_thumb_axis_y_n` | 排除 thumb body 后的法向力 |
| `object_contact_body_count` | 单 body 法向力 `>=0.05 N` 的数量 |
| `object_by_body_axis_y_n` | 逐 body 的单向法向力 |
| `object_by_body_norm_n` | 逐 body 的 3D 力模长 |

抬升阶段要求每侧 `non_thumb_axis_y_n >= 3 N` 且
`object_contact_body_count >= 2`，避免只靠拇指挂住箱体。

## 4. 数据访问

```python
contact_hands = scene["contact_hands"]
contact_hands.body_names
contact_hands.data.net_forces_w
contact_hands.data.current_air_time

object_sensors = {
    body: scene[sensor_name]
    for body, sensor_name in O6_OBJECT_CONTACT_SENSOR_NAMES.items()
}
lh_palm = object_sensors["lh_hand_base_link"]
lh_palm.data.force_matrix_w
```

Isaac Lab filtered sensor 只支持单 body，因此不能把 24 body regex 和
`filter_prim_paths_expr` 合并成一个 filtered sensor。R1-o6 的 USD body 是 Robot prim
的直接子节点，当前单层 `prim_path` 才能匹配。

## 5. State 与录制输出

`state.contact_forces` 包含：

- `contact_hands.by_body_w`：24 body broad 世界系向量和模长；
- `fingertips_object_n`、`fingertips_object_raw_w`：10 指尖兼容视图；
- `palms_object_raw_w`、`palms_object_n`：双掌 filtered 视图；
- `auto_clamp_hands`：每手 12 body 聚合后的完整夹箱控制视图。

录制目录的 `sensor_data.jsonl` 每帧保存同一 `contact_forces` 结构，并额外保存双臂
抬升协调器状态。调试双臂夹箱应优先读取 `auto_clamp_hands`，不要只看
`fingertips_object_n`。

## 6. 前提与限制

| 约束 | 说明 |
|---|---|
| `activate_contact_sensors=True` | 必须在机器人 USD spawn 配置中启用 |
| 更新频率 | `update_period=0.0`，随 120 Hz physics step 更新 |
| 仿真阈值 | `force_threshold=0.05 N` |
| 空间分辨率 | 每 link 一个合力，不能表示真实阵列压力分布 |
| Broad 来源 | 不区分物体、桌面等碰撞源 |
| Filtered 数量 | 24 个独立 sensor，场景初始化成本高于旧 10-sensor 方案 |

## 7. Smoke Test

先按 [`OPERATIONS.md`](OPERATIONS.md) 设置 `PROJECT_DIR` 和 `ISAACLAB_DIR`、生成
机器人 USD，然后运行：

```bash
cd "$ISAACLAB_DIR"
./isaaclab.sh -p "$PROJECT_DIR/tests/smoke_test_tactile.py" --headless \
  --object courier_box_m
```

当前 smoke test 验证 24-body broad sensor 的 body 集合和张量有限性，但 filtered
部分仍只逐项检查 10 个指尖 sensor。它不能证明另外 14 个掌心/proximal/thumb-base
filtered sensor 全部工作；这部分目前由服务启动和双臂夹箱运行数据覆盖，后续应扩展
smoke test 做完整 24-sensor 枚举断言。
