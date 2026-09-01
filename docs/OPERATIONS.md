# 部署与操作

以下命令假设项目已经克隆到 Isaac Lab 服务器。所有路径都通过变量显式指定，不依赖原开发机目录。

## 1. 准备路径

```bash
PROJECT_DIR="/absolute/path/to/r1_o6_dual_box_grasp"
ISAACLAB_DIR="/absolute/path/to/IsaacLab"

python3 "$PROJECT_DIR/scripts/prepare_curobo_configs.py"
```

生成文件位于 `$PROJECT_DIR/.runtime/configs/`，不会修改受 Git 跟踪的模板。

本页的箱体位置、X 偏移和目标力来自服务器验证基线，不是示例参数。机器可读记录见
`configs/validated_runs.json`。其中 `auto_clamp.py` 与验证服务器文件 SHA-256 一致，
三份 cuRobo YAML 仅把服务器绝对路径替换成了 `__PROJECT_ROOT__`。

服务默认绑定 `127.0.0.1`。只有客户端确实位于另一台受信任主机时，才应在启动命令中加入 `--host <受信任接口地址>`。控制协议没有认证和 TLS，不得将端口直接暴露到公网或不受控的共享网段。

## 2. 首次生成机器人 USD

```bash
cd "$ISAACLAB_DIR"
./isaaclab.sh -p "$PROJECT_DIR/scripts/convert_r1_o6_urdf.py" --force
```

完成后必须存在：

```text
assets/R1-o6/r1_o6_arms_hands_fixed.usd
```

## 3. 启动中号快递箱服务

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

## 4. 启动泡沫箱服务

```bash
cd "$ISAACLAB_DIR"
LIVESTREAM=2 ./isaaclab.sh -p "$PROJECT_DIR/pose_grasp_server.py" \
  --headless --livestream 2 --enable_cameras \
  --object foam_box \
  --robot-config "$PROJECT_DIR/.runtime/configs/r1_o6.yml" \
  --left-config "$PROJECT_DIR/.runtime/configs/r1_o6_left.yml" \
  --right-config "$PROJECT_DIR/.runtime/configs/r1_o6_right.yml" \
  --robot-root-x-offset-m -0.03 \
  --clamp-x-offset-m -0.03 \
  --clamp-force-target-n 11.0
```

flat-palm v19 成功录制中的泡沫箱初始世界位置是 `X=0.295 m`，实际抬升约
`0.1989 m`。这里显式写出每侧 `11.0 N`，避免以后 profile 默认值变化后静默偏离
成功基线。请求对象必须与服务启动时的 `--object` 一致，否则服务返回
`object_mismatch`。

服务器清理运行目录时删除了 flat-palm v19 的 `server.log`；保留下来的证据是
`sensor_data.jsonl`、视频、服务器启动约定和相邻实验日志。这个限制记录在
`configs/validated_runs.json`，不得把它描述成完整命令审计链。

## 5. CLI 操作

在控制终端中：

```bash
SERVER_IP="127.0.0.1"
python3 "$PROJECT_DIR/pose_grasp_cli.py" --host "$SERVER_IP" --port 5560
```

依次输入：

```text
state
home
state
clamp
```

泡沫箱也可显式输入 `clamp foam_box`。执行 `clamp` 的 CLI 会同步等待，实验前应打开第二个 CLI 用于 `state` 监控和 `stop`。

## 6. 成功与失败口径

只有以下终态算完整成功：

```text
Auto-clamp COMPLETE (lifted)
done status=lifted
```

箱体最高上升超过 15 cm，但最终返回 `force_limit`、`contact_degraded` 或其他错误，仍属于保护性失败。

录制时可以增加 `--record-dir "$PROJECT_DIR/runs/<run-name>" --record-fps 30`。录制模式会自动放宽超时和 `max_alpha_difference`，实验报告必须标注是否录制。

## 7. 仿真冒烟测试

生成 USD 和 `.runtime/configs` 后，可以先验证 cuRobo FK/IK/规划链路：

```bash
cd "$ISAACLAB_DIR"
./isaaclab.sh -p "$PROJECT_DIR/tests/smoke_test_r1_o6_curobo.py" \
  --headless \
  --robot-config "$PROJECT_DIR/.runtime/configs/r1_o6.yml"
```

触觉传感器冒烟测试见 [`tactile_sensors.md`](tactile_sensors.md)。冒烟测试通过仍不等于
双臂接触闭环成功，最终证据必须来自完整 `auto_clamp` 运行。
