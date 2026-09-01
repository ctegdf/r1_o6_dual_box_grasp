# 大方箱双臂夹取与抬升 — 技术文档

## 1. 文档口径

本文以 flat-palm v19 实验成功为基线，覆盖 `courier_box_m` 和 `foam_box` 双臂
预抓、触觉夹紧、协调抬升和结果验证的完整流程化模块化说明。核心实现位于：

- `auto_clamp.py`：无 Isaac Lab/CUDA 依赖的几何、力控和双臂抬升协调器；
- `pose_grasp_server.py`：仿真状态机、FK、局部 IK、传感器读取和关节目标下发；
- `r1_o6_scene_cfg.py`：机器人、桌面、物体和触觉传感器配置；
- `pose_grasp_cli.py`：JSONL 客户端和人工操作入口；
- `tests/test_auto_clamp.py`：可离线运行的几何与控制回归测试。

任务的真实需求是把箱体抬升约 15 cm。`courier_box_m` 命令高度 `0.16 m`，
`foam_box` 命令高度 `0.20 m`；最终成功阈值均为箱体实际上升至少 `0.15 m`。

### 当前实验结果

| 对象 | 基线版本 | 结果 | 实际上升 | 最终力 | 时长 |
|---|---|---|---:|---:|---:|
| `courier_box_m` | V50 | `done(status=lifted)` | ~165 mm | L≈10.5/R≈10.5N | ~90s |
| `foam_box` | flat-palm v19 | `done(status=lifted)` | 198 mm | L=20.3/R=20.3N | 1548s |

历史 foam_box 录制包含 3376 帧 PNG 和 `sensor_data.jsonl`；为控制 GitHub 仓库体积，
运行产物不包含在本项目目录中。

---

## 2. 系统架构

```text
auto_clamp.py (纯算法层)                pose_grasp_server.py (执行层)
┌─────────────────────────────┐         ┌──────────────────────────────────────┐
│ ClampConfig / profiles      │         │ Isaac Lab 仿真环境 (120 Hz)          │
│ build_clamp_targets()       │◄───────►│ AutoClampContext (状态机上下文)       │
│ evaluate_force_control()    │         │ step_auto_clamp() (每仿真步调用)      │
│ BimanualLiftState           │         │ _solve_clamp_pose() (GPU FK 搜索)    │
│ update_bimanual_lift()      │         │ _squeeze_joint_command() (力→关节)    │
│ aggregate_hand_normal_force │         │ _advance_unified_lift_nominal()       │
│ palm_alignment()            │         │ _unified_lift_joint_command()         │
│ 可离线单元测试              │         │ 24-body 触觉传感器                    │
└─────────────────────────────┘         └──────────────────────────────────────┘
```

两层之间的契约通过纯 Python 数据类型传递——`auto_clamp.py` 不依赖 torch/Isaac
Lab/CUDA。所有传感器读取、FK/IK、关节目标下发由 `pose_grasp_server.py` 负责。

---

## 3. 对象 profile、几何与手型

`auto_clamp.py` 中的 `AUTO_CLAMP_PROFILES` 维护两种对象配置：

| 对象 | 尺寸 XYZ | 目标力 | 力带 | 预抓 Y | 最大内夹 Y |
|---|---:|---:|---:|---:|---:|
| `courier_box_m` | `0.30 x 0.22 x 0.20 m` | `10.5 N` | `9.5-11.5 N` | `+/-0.180 m` | `+/-0.041 m` |
| `foam_box` | `0.35 x 0.24 x 0.25 m` | `11.0 N` | `9.0-13.0 N` | `+/-0.190 m` | `+/-0.051 m` |

两种 profile 共用 `0.07 m` 预抓净空和 `-0.069 m` hand-base/掌面补偿，理论内收
行程均为 `0.139 m`。以运行时箱体中心 `(ox, oy, oz)` 为基准，例如 foam_box
hand-base 目标为：

| 目标 | 左手 | 右手 |
|---|---|---|
| 预抓 | `(ox, oy + 0.190, oz + 0.100)` | `(ox, oy - 0.190, oz + 0.100)` |
| 最大内夹 | `(ox, oy + 0.051, oz + 0.100)` | `(ox, oy - 0.051, oz + 0.100)` |

O6 的掌面法向是 `lh/rh_hand_base_link` 的 local `+X`，伸直手指方向是 local
`+Z`。平掌手型配置：

| 参数 | `courier_box_m` | `foam_box` |
|---|---:|---:|
| `flat_hand_finger_mcp_rad` override | `0.0 rad`（不覆盖基准手型） | `0.3 rad` |
| 运行时有效四指 MCP | `0.1745 rad`（10°，来自 `CLAMP_FLAT_HAND_VALUES`） | `0.3 rad`（~17°） |
| 拇指 yaw/pitch | `0.0 / 0.0` | `0.0 / 0.0` |

`_clamp_flat_hand()` 只在 override 为非零时替换 `CLAMP_FLAT_HAND_VALUES` 中的
基准值；因此 `courier_box_m` 的配置值 `0.0` 不代表运行时四指关节目标为零。

---

## 4. 传感器与力的定义

配置为 24-body 双手触觉系统：

- **1 个 broad contact sensor**：覆盖 24 个手部刚体，检测任意碰撞，用于硬限位保护；
- **24 个 Object-filtered sensor**：每手 12 个 body（5 distal + 5 proximal + 1 掌心 +
  1 thumb base），只统计与 `Object` 的接触。

控制力不是力矢量模长，而是 **每个 Object-filtered body 在夹持 Y 轴上的单向压缩
分量之和**。左手只累计 `+Y` 方向箱体反力，右手只累计 `-Y` 方向，反方向样本截为
0。这避免切向冲击或反弹被误判为有效夹持力。

抬升阶段使用的接触质量指标：

| 指标 | 阈值 | 含义 |
|---|---:|---|
| `non_thumb_axis_y_n` | ≥ 3.0 N | 掌心/四指必须有实质支撑 |
| `object_contact_body_count` | ≥ 2 | 至少两个 body 有效接触 |

以上指标防止仅靠拇指悬挂的不稳定构型。

---

## 5. 状态机完整流程

### 5.0 总览

```text
                   ┌──── 净空不足，增加 roll ─────┐
                   │                              │
PREPARE_HANDS ─→ SIDE_RAISE ─→ OPEN_HANDS ─→ VERIFY_CLEARANCE
                                                   │ 净空通过
                                                   ↓
SOLVE_PREGRASP_LEFT ─→ SOLVE_PREGRASP_RIGHT ─→ MOVE_PREGRASP ─→ VERIFY_PREGRASP
                                                                        │
                                                                        ↓
SOLVE_CLAMP_LEFT ─→ SOLVE_CLAMP_RIGHT ─→ FORCE_CLAMP ─→ MOVE_LIFT ─→ VERIFY_LIFT
                                              │                            │
                                        力达标+稳定                  retained+lifted
                                        _begin_unified_lift()       done(status=lifted)
```

`AutoClampPhase` 仍保留 `SOLVE_LIFT_LEFT/RIGHT` 枚举，但当前主路径在夹紧稳定
后由 `_begin_unified_lift()` 直接进入 `MOVE_LIFT`，不经过旧的 SOLVE_LIFT 分支。

状态机在 **120 Hz** 仿真频率下运行，每物理步调用一次 `step_auto_clamp()`。
进入具体阶段分支前先执行两级超时检查：

- 总时长超过 `total_timeout_s` → `timeout`（courier 90s、foam 130s）；
- 当前阶段超过阶段超时 → `phase_timeout`。其中 `FORCE_CLAMP` 和 `MOVE_LIFT`
  使用 `force_phase_timeout_s`（70s），其他阶段使用 `phase_timeout_s`（30s）。

### 5.1 PREPARE_HANDS — 手部准备

| 项目 | 内容 |
|---|---|
| **目的** | 将双手设置为过渡蜷曲姿态，避免手指在后续侧抬中碰撞桌面 |
| **输入** | 无 |
| **动作** | 双手 active joints → `CLAMP_TRANSIT_HAND_VALUES = (0.5, 0.4, 0.8, 0.8, 0.8, 0.8)`（拇指半开，四指蜷曲） |
| **等待** | 20 步（~0.17s），让手指物理到位 |
| **转移** | → `SIDE_RAISE` |

### 5.2 SIDE_RAISE — 双臂侧抬

| 项目 | 内容 |
|---|---|
| **目的** | 双臂 shoulder_roll 向外展开，为手掌接近箱体侧面腾出空间 |
| **输入** | `side_raise_roll_rad`（初始 0.75 rad ≈ 43°） |
| **动作** | 只修改 `shoulder_roll_joint`：左臂 `+roll_rad`，右臂 `-roll_rad`；其余 4 个 DOF 保持不变 |
| **执行** | `_start_dual_linear()` 生成 120 个线性 waypoint + 默认 settle 步，双臂同步 |
| **转移** | 轨迹完成 → `OPEN_HANDS` |

### 5.3 OPEN_HANDS — 张开手掌

| 项目 | 内容 |
|---|---|
| **目的** | 从过渡蜷曲切换为 profile 定义的平掌姿态，准备以掌面接触箱体 |
| **动作** | `_clamp_flat_hand(config)` 根据 profile 生成手指关节目标 |
| **等待** | 30 步（~0.25s） |
| **转移** | → `VERIFY_CLEARANCE` |

手指关节构成：
```
四指 MCP    = config.flat_hand_finger_mcp_rad（非零时覆盖）
              否则使用 CLAMP_FLAT_HAND_VALUES 的 10° 基准值
拇指 yaw   = config.flat_hand_thumb_yaw_rad     (均 0°)
拇指 pitch = config.flat_hand_thumb_pitch_rad   (均 0°)
其余关节   = 0（全伸直）
```

### 5.4 VERIFY_CLEARANCE — 净空验证

| 项目 | 内容 |
|---|---|
| **目的** | 确保张开的指尖不会插入桌面 |
| **检查** | 所有 10 根指尖世界 Z ≥ `TABLE_SURFACE_Z + fingertip_table_clearance_m`（0.72 + 0.03 = 0.75m） |
| **通过** | 确认平掌姿态 → `SOLVE_PREGRASP_LEFT` |
| **不通过** | `roll += side_raise_increment_rad`（+0.10 rad），双手重新蜷曲，回到 `SIDE_RAISE` |
| **失败** | roll > `max_side_raise_roll_rad`（1.40 rad ≈ 80°）仍不通过 → `clearance_failed` |

递增机制确保在最小展臂角度下完成净空——不会过度展开导致后续运动学困难。

### 5.5 SOLVE_PREGRASP_LEFT → SOLVE_PREGRASP_RIGHT — 求解预抓位姿

| 项目 | 内容 |
|---|---|
| **目的** | 用确定性 GPU FK 搜索找到双手到达预抓位置的 5DOF 关节解 |
| **方法** | `_solve_clamp_pose()` 使用固定随机种子做 GPU Monte-Carlo FK 搜索：全局 500,000 个候选 + 两轮各 150,000 个缩小半径的局部随机候选，共 800,000 个；无梯度精化 |
| **后处理** | wrist_roll 做 2π 等价折叠，选最接近 seed 的角度，避免整圈旋转 |
| **路径验证** | pregrasp 使用 `_staged_pregrasp_arm_path()` 生成 261 个 staged waypoint，仅检查 hand-base 高度 > 桌面 + 10cm；clamp 使用 121 个线性插值点，额外检查 Y 单调内收（反向步 < 2mm）、掌面朝向和手指朝下 |
| **转移** | 两侧完成 → `MOVE_PREGRASP` |

### 5.6 MOVE_PREGRASP — 执行预抓运动

| 项目 | 内容 |
|---|---|
| **目的** | 双臂同步从侧抬位姿移动到箱体两侧预抓位置 |
| **执行** | `_start_dual_staged_pregrasp()` 生成 261 个 waypoint：先保持 shoulder_roll、移动其余 4 个关节至预抓终值，再将 shoulder_roll 移到预抓终值；末尾 30 步 settle |
| **安全** | 每步监控所有指尖世界 Z；如 < `TABLE_SURFACE_Z - 5mm` → `table_clearance` 失败 |
| **转移** | 轨迹完成 → `VERIFY_PREGRASP` |

### 5.7 VERIFY_PREGRASP — 预抓验证

三重检查：

| 检查项 | 条件 | 失败代码 |
|---|---|---|
| 位姿偏差 | 实际 TCP 与目标 FK 偏差 ≤ 25mm，掌面朝向满足 dot 阈值 | `pregrasp_mismatch` |
| 箱体位移 | 箱体当前位置与初始快照偏差 ≤ 20mm | `object_moved` |
| 意外接触 | 双手 object_axis_y_n 最大值 > 0.5N | `unexpected_contact` |

全部通过后记录当前关节为 `force_start`（alpha 插值起点），转入 `SOLVE_CLAMP_LEFT`。

### 5.8 SOLVE_CLAMP_LEFT → SOLVE_CLAMP_RIGHT — 求解夹紧终点

| 项目 | 内容 |
|---|---|
| **目的** | 计算双手完全贴合箱面的关节解 |
| **夹紧位置** | `clamp_y = half_width + clamp_surface_margin_m`（margin = -0.069m，负值让 FK 目标深入箱面，补偿 hand-base 到物理接触面的 ~30mm 偏移） |
| **验证** | 121 个线性插值点上检查高度、单调内收、掌面朝向和手指朝下 |
| **转移** | 两侧完成 → `FORCE_CLAMP` |

### 5.9 FORCE_CLAMP — 力控夹紧（核心阶段）

这是最复杂的阶段，包含多个并行子系统。每仿真步（120Hz）执行一次。

#### 5.9.1 Alpha 插值运动

双手沿 `force_start → clamp` 的关节空间路径用 alpha ∈ [0, 1] 参数化：

```
q_command = q_start + alpha * (q_end - q_start)
```

Alpha 推进速率由力控决策和进度区间决定：

| 区间 / 模式 | 步长 | 说明 |
|---|---:|---|
| `alpha < contact_zone_alpha (0.50)` | `clamp_alpha_step = 0.0004` | 快速接近 |
| `alpha ≥ 0.50`（接触区） | `contact_zone_alpha_step = 0.0001` | 减速接近 |
| 会合模式（一侧有接触另一侧无） | `rendezvous_alpha_step = 0.0002` | 中速，等另一手 |

两种 profile 都使用 `approach_progress_mode="tcp_y"`。Alpha 推进受
`bounded_inward_alpha()` 限制，命令最多领先实测进度
`max_command_alpha_lead=0.025`。双侧 alpha 差值 ≤
`max_alpha_difference`（默认 0.20，录制时放宽到 1.0）。

Squeeze anchor 捕获前，outward alpha 退让步长分别为：
`unilateral_relief_alpha_step=0.0001`（会合模式）、
`relief_alpha_step=0.004`（正常）和
`emergency_relief_alpha_step=0.04`（紧急）。这些 alpha 退让与 anchor 捕获后的
`squeeze_relief_step_m=0.5mm` 是两套不同机制。

#### 5.9.2 力控决策 (`evaluate_force_control()`)

每步读取双手 **object-filtered Y 轴压缩力**，分两种模式决策：

**单侧接触模式**（只有一手碰到箱子）——解决"箱子被推走"问题：

```
已接触侧:
  force < contact_detect_n(0.2N)  → inward（恢复接触）
  force ∈ [0.2, 0.8]N            → hold（保持轻预载）
  force > 0.8N                    → outward（卸载，别推箱子）
未接触侧:
  always → inward
```

首次达到 `contact_detect_n` 即锁存已接触。单侧的轻预载确保先碰到的手不把箱体
推离另一手。

**双侧接触模式**（两手都碰到箱子）——独立力带调节：

```
force < force_lower_n  → inward，stable_frames = 0
force ∈ [lower, upper] → hold，stable_frames += 1
force > force_upper_n  → outward，stable_frames = 0
```

两侧各自连续 `force_stable_frames`(12) 帧在带内 → **success**。

#### 5.9.3 虚拟压缩挤压 (Squeeze)

当双手都接触箱体后（依据 `contact_seen`，非 palm-only force），alpha 停止推进，
转为 **contact-local TCP-Y Jacobian 挤压**：

1. **锚点捕获**：在双侧 `contact_seen` 首次均为 True 时，捕获当前 alpha 插值
   命令位置（非实际位置）为 `squeeze_anchor_q`，并计算数值 TCP-Y Jacobian
   `J_y`（对 5 个关节分别施加 0.001 rad 扰动，测量 FK 末端 Y 变化）
2. **压缩量更新**：根据力控 action 更新 `squeeze_compression_m`：
   - inward: `+= squeeze_compression_step_m`（0.02mm/步）
   - outward: `-= squeeze_relief_step_m`（0.5mm/步，快速释放）
   - hold: 不变
3. **关节命令转换**：`_squeeze_joint_command()` 将压缩量→关节偏移：
   ```
   q = anchor_q + J_y^T * compression * gain / (J_y^T . J_y + damping)
   ```

| 参数 | `courier_box_m` | `foam_box` |
|---|---:|---:|
| 每次加压 | 0.02 mm | 0.02 mm |
| 每次卸载 | 0.5 mm | 0.5 mm |
| 最大虚拟压缩 | 30 mm | **100 mm** |
| 关节目标增益 | 6.0 | **1.0**（慢增压，防止落座冲击） |

#### 5.9.4 可选的掌面落座重锚（当前两个 profile 均关闭）

`squeeze_contact_guard_enabled` 默认为 `False`，两个 profile 都没有覆盖它。
因此当前运行不会进入落座检测分支。若通过程序化 override 显式启用：

```
任一侧 object_axis_y_n ≥ squeeze_seat_detect_n(12N)
  → hold，并进入 squeeze_seat_unload_frames(10步) 倒计时
  → 倒计时期间按实际关节冻结
  → 结束后以实际关节重新锚定，compression 归零

或者双侧力平滑进入目标带并稳定 12 帧
  → 直接以实际关节重新锚定，compression 归零
```

重锚前后始终使用同一个 `squeeze_joint_target_gain`——没有"低增益→正常增益"
切换。

#### 5.9.5 可选的接触丢失防护（当前两个 profile 均关闭）

仅当 guard 显式启用，且任一侧 compression 严格大于
`squeeze_contact_loss_min_compression_m`(1mm) 后：

- 任一侧总 `object_axis_y_n < squeeze_contact_loss_n`(0.05N) → 按实际关节冻结
- 连续冻结 ≥ `squeeze_contact_loss_frames`(30帧) → `squeeze_contact_lost`
- 力恢复 → 以实际关节重新锚定后继续

默认两个 profile 下，`squeeze_contact_lost` 失败码不可达。

#### 5.9.6 安全保护（每步执行）

| 检查 | 条件 | 失败代码 |
|---|---|---|
| 广域力限位 | 24 个 broad-sensor body 中最大的 **单-body force-vector norm** ≥ `force_hard_limit_n`，连续 `broad_force_limit_frames` 帧 | `force_limit` |
| 箱体位移 | 位移 > `max_object_displacement_m`（courier 20mm / foam 50mm）或旋转 > `max_object_rotation_rad`(10°) | `object_moved` |
| 指尖桌面 | 任何指尖 Z < TABLE_SURFACE_Z - 5mm | `table_clearance` |
| 掌面朝向 | 每 30 步：inward_dot < `hard_min_palm_inward_dot`(0.70) 或 down_dot < `hard_min_finger_down_dot`(0.60) | `orientation_lost` |
| 路径耗尽 | 任一臂 action 仍为 inward、alpha=1.0、TCP tracking error ≤ `endpoint_position_tolerance_m`(20mm)，连续 `endpoint_no_contact_frames`(120) 帧 | `no_contact` |

#### 5.9.7 成功条件

```
双手 squeeze 力在目标带内连续稳定 12 帧
AND squeeze 锚点已捕获（双侧接触）
AND 非冻结/非重锚帧
AND (contact_guard 未启用 OR 落座已完成)
→ 调用 _begin_unified_lift()，进入 MOVE_LIFT
```

### 5.10 MOVE_LIFT — 统一触觉抬升（核心阶段）

在维持夹持力的同时，双手同步垂直抬升箱体。不使用一次性求解终点的旧方案，而是
每步重新线性化的局部 IK + 触觉力闭环。

**注意**：`MOVE_LIFT` 不再执行 broad-sensor 去抖、箱体位移/旋转、指尖桌面净空
和掌面朝向检查。安全保护仅依赖 per-side object-filtered 力限位和接触退化检测。

#### 5.10.1 初始化 (`_begin_unified_lift()`)

1. 以当前 squeeze 命令关节位为 lift 起点 `lift_nominal_q`
2. 提取任务特征锚点 `lift_anchor_features`（Y 坐标、Z 坐标、掌面法向分量）
3. 创建 `BimanualLiftState` 协调器，以当前力为 EWMA 滤波器初值
4. 记录箱体初始高度 `lift_start_object_z` 和双手初始 Z

#### 5.10.2 双 EWMA 滤波架构

```
raw force (120 Hz)
    |
    |---> 快滤波器 (alpha=0.08, tau~12步~0.1s)
    |       用于: squeeze/center 力控决策，响应快
    |
    +---> 慢滤波器 (alpha=0.005 foam / 0.08 courier, tau~200步~1.7s)
            用于: advance gate 决策，抵抗 PhysX 接触共振
```

courier_box 两个滤波器相同（alpha=0.08），foam_box 慢滤波器 alpha=0.005 是
专门为对抗 70-80mm 高度的 PhysX 接触共振设计的（~120 步周期，右侧峰值 ~42N，
时均 15-27N）。

#### 5.10.3 三自由度力控分解

`BimanualLiftState` 将耦合问题拆成三个独立控制自由度：

| 状态量 | 含义 | 控制方式 |
|---|---|---|
| `progress_m` | 双臂共享的垂直进度 | 只在 advance gate 通过时前进 |
| `squeeze_offset_m` | 对称改变双掌间距 | 有符号方向证据积分器调节平均力 |
| `center_offset_m` | 双掌整体横移 | 力差驱动，从强侧向弱侧转移 |

这种分解避免用"再夹紧"同时处理平均力和左右不平衡：两侧都偏弱时才缩小间距；
一侧强、一侧弱时保持间距并移动双掌中心。

#### 5.10.4 对称挤压控制 — 有符号方向证据积分器

源码用快 EWMA 滤波力生成有符号方向证据。积分器只在双侧完整接触，或双侧仍
满足恢复接触下限（`non_thumb ≥ recovery_min`(0.5N) 且 `body_count ≥ 2`）时
更新：

```python
if 弱侧滤波力 < force_lower AND 强侧 <= force_upper:
    counter += 1   # 需要更紧
elif 强侧滤波力 > force_upper AND 弱侧 >= force_lower:
    counter -= 1   # 需要更松
# 带内/交叉帧保持 counter 不变；相反方向逐帧抵消已有证据
# 接触门控不满足或出现硬限位时，counter 也保持不变

counter = clamp(counter, -threshold, +threshold)

if counter >= threshold:
    squeeze_offset += lift_squeeze_slew_m   # +0.04mm
    counter = 0                              # 归零重新积累
elif counter <= -threshold:
    squeeze_offset -= lift_squeeze_slew_m
    counter = 0
```

| 参数 | `courier_box_m` | `foam_box` |
|---|---:|---:|
| threshold (persist_frames) | 180 | **20** |
| slew 步长 | 0.04 mm | 0.04 mm |

threshold 表示达到一次动作所需的净方向证据量，并不要求严格连续同向帧——
中性帧不清零，反方向帧只是逐帧抵消。每次动作后归零，限制 squeeze offset
的变化速率。

#### 5.10.5 差分中心控制

左右力差驱动双手整体平移（不改变间距）：

```python
# 紧急模式: raw 力达到 soft_limit 或绝对差异达到 2N
if soft_rebalance:
    force_diff = raw_left - raw_right
    center_slew = lift_center_slew_m * emergency_multiplier(2.0)
else:
    force_diff = filtered_left - filtered_right
    center_slew = lift_center_slew_m  # 0.02mm

if diff > balance_tolerance(0.5N):
    center_offset += center_slew    # 向右推，减轻左手
elif diff < -balance_tolerance:
    center_offset -= center_slew    # 向左推，减轻右手
```

中心控制在一侧接触退化时仍保持活跃——只要至少一侧 `contact_ok`，这是恢复
接触的安全方式。

#### 5.10.6 前进门控 (Advance Gate)

决定"现在可以安全抬升 0.25mm 吗？"使用 **慢滤波力**：

```python
balanced = (
    error is None                                              # 无硬限位
    AND left_contact_ok AND right_contact_ok                   # 双侧接触完好
    AND force_lower <= advance_filtered_L <= advance_upper     # 慢滤波力在宽带内
    AND force_lower <= advance_filtered_R <= advance_upper
    AND |advance_L - advance_R| <= advance_balance_tolerance   # 慢滤波力平衡
    AND vertical_tracking_ok                                    # 手臂跟上命令高度
)

advance = balanced AND stable_frames >= lift_progress_stable_frames(8)
```

advance gate 的力带上界可以独立配置：

| 参数 | `courier_box_m` | `foam_box` |
|---|---:|---:|
| advance upper | 11.5N（= force_upper） | **35.0N**（容忍共振时均） |
| advance balance | 0.5N | **10.0N**（容忍共振反相残差） |

通过后执行一次微抬升：
```
delta_z = lift_height_m / lift_steps = 0.20/800 = 0.25mm (foam)
progress += delta_z
stable_frames = 0  # 每步消耗稳定窗口，需重新积累
```

#### 5.10.7 局部 IK — 任务优先级 (`_advance_unified_lift_nominal()`)

局部 IK 只在至少一侧 `contact_ok` 时调用。任务特征是
`[Y, Z, local_x_x, local_x_z]`，绕掌面法向的 roll 自由。

```
j = J[0]                                      # Y 方向 Jacobian 行
dq1 = j * error_y / (j . j + lambda1)         # 一级任务: Y
N = I - outer(j, j) / (j . j + lambda1)       # 近似零空间投影

J2N = J[1:] @ N                               # 二级 Jacobian 投影
dq2 = N @ J2N.T @ solve(J2N @ J2N.T + lambda2*I, error2)

dq = dq1 + dq2
若 max(abs(dq)) > 0.003 rad，则整体等比例缩放到 0.003 rad
```

`full_task=False` 时（接触恢复中）把 Z/方向目标设为当前值，只修正 Y；
两侧都不满足 `contact_ok` 时不更新 nominal。由于 `lambda1` 非零，`N` 不是
精确零空间投影，但 Y 方向仍然是 damped-priority 一级任务。

#### 5.10.8 力覆盖层 (`_unified_lift_joint_command()`)

在 nominal_q 上叠加 squeeze + center 的 Y 方向 Jacobian 偏移：

```python
desired_delta_y = center_offset + (-squeeze_offset if left else +squeeze_offset)
delta_q = J_y * delta_y / (J_y . J_y + damping)
q_command = clamp(nominal_q + delta_q, joint_lower, joint_upper)
```

**关键设计**：力反馈 **不** 积分回 `nominal_q`。这保证一次无法实现的力补偿只是
有界的位置控制误差，而不会递归改变后续抬升姿态。

#### 5.10.9 安全保护

| 保护 | 条件 | 反应 |
|---|---|---|
| 硬力限位 | 任一侧 raw `object_axis_y_n` ≥ `force_hard_limit_n` 连续 `lift_hard_limit_frames`(3) 帧 | `force_limit` 失败（注意：这里检查的是 per-side object-filtered 力，不是 broad sensor） |
| 接触退化 | 双侧完整 contact 时 recovery 归零；否则递增，达到 `lift_force_drop_frames` | `contact_degraded` 失败 |
| 紧急释压 | 硬限位触发当帧 | squeeze_offset -= 2mm, center 步进 0.2mm |

foam_box 的 `lift_force_drop_frames=180`（1.5 个振荡周期），确保共振低相位的
瞬时力降不会误触发失败。

#### 5.10.10 完成条件

```
progress >= lift_height_m           # 命令高度到位
AND balanced 连续 >= lift_settle_steps(60)   # 60帧稳定
→ 转入 VERIFY_LIFT
```

### 5.11 VERIFY_LIFT — 抬升验证

| 项目 | 内容 |
|---|---|
| **目的** | 确认箱体确实被抬起且力保持住 |
| **测量** | `object_rise = current_object_z - lift_start_object_z` |
| **retained** | advance_filtered 力在宽带内 AND 双侧 non_thumb ≥ 3.0N |
| **lifted** | `object_rise >= lift_min_object_rise_m`（0.15m） |
| **成功** | retained AND lifted 连续 ≥ `lift_verify_frames`(30帧) → `done(status=lifted)` |
| **失败** | 总验证超过 180 帧仍未通过 → `lift_not_retained` |

### 5.12 超时与错误码全集

| 代码 | 触发位置/条件 |
|---|---|
| `busy` / `no_profile` / `object_mismatch` | 请求校验失败（clamp 命令入口） |
| `timeout` | 总时长超过 `total_timeout_s` |
| `phase_timeout` | 当前阶段超过阶段超时 |
| `clearance_failed` | 净空重试 roll 耗尽 |
| `table_clearance` | MOVE_PREGRASP 或 FORCE_CLAMP 路径指尖进入桌面 |
| `pregrasp_mismatch` | 预抓位姿偏差过大 |
| `object_moved` | 预抓或 FORCE_CLAMP 阶段箱体位移/旋转超限 |
| `unexpected_contact` | 预抓完成时手上有意外接触力 |
| `orientation_lost` | FORCE_CLAMP 阶段掌面朝向偏离 |
| `no_contact` | FORCE_CLAMP 路径耗尽仍无力接触 |
| `force_limit` | FORCE_CLAMP 的 per-body broad guard，或 MOVE_LIFT 的 per-side raw object-filtered guard |
| `squeeze_contact_lost` | 仅 guard 显式启用时可达 |
| `contact_degraded` | 抬升中接触质量未恢复 |
| `lift_state` | 抬升协调器或起始高度缺失（内部状态错误） |
| `lift_not_retained` | 抬升到位但验证阶段未保持住箱体 |
| `execution_failed` / `exception` | 轨迹或状态机运行时异常 |
| `cancelled` | `stop` 命令清除正在运行的任务 |

---

## 6. Profile 关键参数对比

本节不是 `ClampConfig` 的完整字段清单；未覆盖字段继承 dataclass 默认值。

### 6.1 几何与手型

| 参数 | `courier_box_m` | `foam_box` |
|---|---:|---:|
| box_size XYZ | 0.30 x 0.22 x 0.20 m | 0.35 x 0.24 x 0.25 m |
| palm_center_offset_z | 0.10 m | 0.10 m |
| flat_hand_finger_mcp_rad override | 0.0 rad（不覆盖） | 0.3 rad |
| 运行时有效 finger MCP | 0.1745 rad（10°） | 0.3 rad |
| thumb yaw/pitch | 0.0 / 0.0 | 0.0 / 0.0 |
| thumb_is_structural | off | off |

### 6.2 力控

| 参数 | `courier_box_m` | `foam_box` |
|---|---:|---:|
| force_target | 10.5 N | 11.0 N |
| force_tolerance | 1.0 N | **2.0 N** |
| force_band | 9.5-11.5 N | 9.0-13.0 N |
| force_hard_limit | 15 N | **100 N** |
| lift_soft_force_limit | 12.5 N | **14.0 N** |
| broad_force_limit_frames | 3 | **15** |
| contact_detect | 0.2 N | **1.0 N** |
| unilateral_preload_upper | 0.8 N | **3.0 N** |

### 6.3 挤压

| 参数 | `courier_box_m` | `foam_box` |
|---|---:|---:|
| squeeze_joint_target_gain | 6.0 | **1.0** |
| max_squeeze_compression | 30 mm | **100 mm** |
| squeeze_contact_guard | off | off |
| squeeze_seat_detect | 12.0 N（guard 开启时才消费） | 12.0 N（guard 开启时才消费） |

### 6.4 抬升

| 参数 | `courier_box_m` | `foam_box` |
|---|---:|---:|
| lift_height | 0.16 m | **0.20 m** |
| force_filter_alpha (快) | 0.08 | 0.08 |
| advance_force_filter_alpha (慢) | 0.08 | **0.005** |
| advance_force_upper | 11.5N (= upper) | **35.0 N** |
| advance_force_balance_tolerance | 0.5 N | **10.0 N** |
| lift_force_drop_frames | 60 | **180** |
| lift_squeeze_persist_frames | 180 | **20** |
| max_object_displacement | 20 mm | **50 mm** |
| total_timeout | 90 s | **130 s** |

### 6.5 foam_box profile 设计理念

foam_box 面对 PhysX 在 70-80mm 高度的接触共振（~120 步周期，右侧峰值 ~42N /
左侧 ~24N，时均 15-27N 远高于 11N 目标），核心策略是：

1. **宽力带 + 高容忍**：advance upper 35N、balance 10N，让慢滤波值在共振时均附近仍能通过 gate
2. **慢 EWMA**：alpha=0.005（tau~200步），120 步周期振荡只穿透 ~9.5%，滤波后接近真实时均
3. **快方向积分器**：persist=20（而非 180），在振荡中快速追踪时均力的变化
4. **长恢复超时**：drop_frames=180，跨越 1.5 个振荡周期，防止低相位 non_thumb 瞬降触发失败
5. **低增益挤压**：gain=1.0 + 100mm 范围，以 ~2.4mm/s 缓慢接近掌面，避免落座冲击
6. **分阶段限位**：FORCE_CLAMP 使用 per-body broad 100N / 15 帧；MOVE_LIFT 使用 per-side object-filtered raw 100N / 3 帧，并以 raw 14N soft limit 提前再平衡

---

## 7. 启动与操作

可移植的资产生成、cuRobo 路径准备、两类箱体启动命令和成功判定统一维护在
[`OPERATIONS.md`](OPERATIONS.md)。泡沫箱默认目标力为每侧 `11.0 N`。

录制模式会自动把 total timeout 放宽到 1200 s、force phase timeout 放宽到
1100 s，并把 `max_alpha_difference` 放宽到 1.0，因此录制与非录制不是相同控制条件。

---

## 8. 离线验证

```bash
# 控制回归测试 (61 个测试用例)
python3 -m unittest discover -s tests -p 'test_auto_clamp.py' -v

# 语法检查
python3 -m compileall -q \
  auto_clamp.py pose_grasp_server.py pose_grasp_cli.py \
  r1_o6_scene_cfg.py delivery_objects_cfg.py tests
```

离线测试通过不代表 Isaac Lab 接触闭环通过。完整成功必须收到 `done(status=lifted)`。

---

## 9. 已知限制

1. foam_box 抬升中箱体旋转约 30-35°；`MOVE_LIFT`/`VERIFY_LIFT` 不检查箱体
   位移/姿态，也不继续检查指尖桌面净空或掌面朝向。
2. `VERIFY_LIFT` 使用冻结的 advance filter 值，不做实时更新。
3. 以下 `ClampConfig` 字段已声明但当前主路径未消费：`lift_min_force_n`、
   `endpoint_joint_tolerance_rad`、`tracking_position_tolerance_m`、
   `tracking_min_palm_inward_dot`、`tracking_min_finger_down_dot`、
   `lift_squeeze_inward_step_m`、`lift_squeeze_outward_step_m`。
4. 服务无认证，只能部署在受控网络。
5. `stop` 的手指和录制语义不完整。
6. cuRobo 世界模型不包含桌面，桌面安全依赖状态机自身的指尖净空检查。
7. 5DOF 无法精确 6D 位姿，局部 IK 依赖任务优先级和容差设计。
8. 录制模式改变控制参数，A/B 对比必须标注模式。

---

## 10. 文档导航

| 文档 | 内容 |
|---|---|
| `docs/auto_clamp.md` | 本文：完整状态机流程、算法、参数和操作 |
| `docs/auto_clamp_handoff.md` | 当前证据、风险和后续 AI 接管清单 |
| `docs/foam_box_auto_clamp.md` | EPS 泡沫箱策略迁移历史和首轮调参 |
| `docs/tactile_sensors.md` | 触觉系统背景 |
| `docs/auto_grasp.md` | 单臂自动抓取技术文档 |
| `docs/state_broadcast.md` | state 消息结构 |
