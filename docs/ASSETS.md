# 机器人资产

## 仓库内文件

- `R1-o6.urdf`：原始机器人描述；
- `meshes/`：URDF 引用的 STL 网格；
- `scripts/convert_r1_o6_urdf.py`：锁定非双臂/双手关节并生成 Isaac Lab USD。

生成过程保留 10 个手臂关节、每手 6 个主动关节和每手 5 个 mimic 关节，其余身体关节转换为 fixed joint。

## 生成 USD

```bash
PROJECT_DIR="/absolute/path/to/r1_o6_dual_box_grasp"
ISAACLAB_DIR="/absolute/path/to/IsaacLab"

cd "$ISAACLAB_DIR"
./isaaclab.sh -p "$PROJECT_DIR/scripts/convert_r1_o6_urdf.py" --force
```

脚本会创建 `assets/R1-o6/`，同步网格并生成 `r1_o6_arms_hands_fixed.usd`。`assets/` 是可再生的大型产物，已被 `.gitignore` 排除。

## 上游来源

`R1-o6.urdf` 是面向本项目的组合模型：R1 本体来自宇树官方仓库，双手来自灵心巧手官方 O6 模型。组合文件调整了关节、命名和网格路径，因此不是任一上游 URDF 的原样副本。

| 本仓库内容 | 上游文件 | 固定版本 | 上游许可证 |
|---|---|---|---|
| `R1-o6.urdf` 的 R1 本体、`meshes/` 根目录下的 43 个 R1 网格 | [Unitree `robots/r1_description/`](https://github.com/unitreerobotics/unitree_ros/tree/bbd833d6ce9826c3ae4e3d44174d99d940111c32/robots/r1_description) | [`bbd833d6`](https://github.com/unitreerobotics/unitree_ros/commit/bbd833d6ce9826c3ae4e3d44174d99d940111c32) | [BSD-3-Clause](https://github.com/unitreerobotics/unitree_ros/blob/bbd833d6ce9826c3ae4e3d44174d99d940111c32/LICENSE) |
| `R1-o6.urdf` 的左右手、`meshes/o6_left/` 和 `meshes/o6_right/` 下的 24 个网格 | [Linker Hand `O6/`](https://github.com/linker-bot/linkerhand-urdf/tree/075cc7d42cc1e756bdcbece0fc069a0779fc5237/O6) | [`075cc7d4`](https://github.com/linker-bot/linkerhand-urdf/commit/075cc7d42cc1e756bdcbece0fc069a0779fc5237) | [Apache-2.0](https://github.com/linker-bot/linkerhand-urdf/blob/075cc7d42cc1e756bdcbece0fc069a0779fc5237/LICENSE) |

2026-09-01 逐文件核对 Git blob 哈希：上述 43 个 R1 网格和 24 个 O6 网格均与表中固定版本一致。上游归属和完整条款见 [`THIRD_PARTY_LICENSES.md`](../THIRD_PARTY_LICENSES.md)；仓库自行编写的代码和文档使用根目录的 Apache-2.0 许可证。

STL 总量约 49 MB，单文件均低于 GitHub 的 100 MB 限制。
