# HyPhy-GLNN：JOREK Model 502 非均匀时间插值代理模型

本项目用 CfC LNN 在 JOREK Model 502 二维数据上训练连续时间代理模型。目标不是"固定步长的下一帧回归"，
而是：

- 在记录帧之间任意时间点给出自洽的场（**非均匀时间插值**）；
- 让"场 → 碎片 → 源项 → 场"的耦合闭环成立；
- 只在没有真值的插值点上使用 Physics Loss 修正轨迹；记录帧上的训练保持**纯数据驱动**。

## 职责分工

| 对象 | 角色 |
|---|---|
| 记录帧 `(u_k, p_k, S_k)` | data loss，纯数据驱动 |
| 插值点 `t* ∈ (t_k, t_{k+1})` | Physics Loss（完整方程残差）修正插值轨迹 |
| 碎片 → 投影 | 源项映射子网络（NGS/Parks 解析先验 + 学习修正） |
| 场 → 碎片 | 混合碎片动力学（JOREK 解析骨干 + 学习修正） |
| 主物理场 | 现有 GNN encoder + CfC processor + decoder |

Physics Loss 不用于有标签的数据点。把物理项当作记录帧一步回归的正则，是上一版路线的错误；
现行语义见 [ADR-0010](docs/adr/0010-physics-loss-only-at-interpolated-times.md)。

## 当前范围

- 数据：`data/EHL-2` 下 10 个纯 2D run（`ntor=1`，只取 n=0 分量）。`ntor11*` 的 11 个环向分量
  保留给 3D 研究，暂不启用（[ADR-0004](docs/adr/0004-ehl2-data-selection-and-phased-training.md)）。
- 10 个 run 全部是 SPI run：九场 `jorek*.h5`、碎片 `part*.h5`（neon marker）、五场源项投影
  `projections*.h5`、shard 级状态 `spi_*` 与逐时间步烧蚀序列。注入时刻 `t_ns = 8380`：
  `t < t_ns` 的帧没有碎片（`n_spi_tot = 0`），`t ≥ t_ns` 的帧有完整 shard 状态
  （`n_spi` = 45–846）。窗口不得跨越 `t_ns`。
- 增广状态：`u` = 九场（每帧 `[nodes, 9]`），`p` = 碎片集合，`S` = 五场投影。
- 2D 数据是极向 FEM + 单一环向模；环向模耦合（3D）不在当前范围。

## 训练原则（CfC 原生）

- 真实逐帧 `dt`；随机 `α` 把帧间隔切成 `α·Δt` 与 `(1−α)·Δt` 两段，hidden state 连续，
  中间场不 detach。
- 记录帧只算 data loss；插值点由 CfC 给 `u(t*)` 与 `∂u/∂t`、碎片模块给 `p(t*)`、
  源项网络给 `S(t*)`，在 `t*` 评估完整方程残差。
- 闭环训练分阶段：教师源 → 自生成源，开环 → 闭环，每步有量化 gate。

## 仓库状态（2026-09-24 清理后）

- 保留：数据/物理/接口核心（`src/data`、`src/physics`、`src/models`、`src/evaluation`、
  `src/training`），包括九场 schema、参数通道、Grad-Shafranov 与图微分算子、残差层。
- 退役：旧的九场训练工作流（stage0/baseline/rollout 脚本与其产物）已删除或归档；
  `ADR-0008/0009` 已归档。
- 待重建：数据窗口与划分、碎片/源项模块、插值训练循环、闭环训练与 benchmark。
- 五场时代的结果只保留在 git 历史与旧脚本中，作为历史证据，不再驱动设计。

## 文档

- 术语：[CONTEXT.md](CONTEXT.md)
- 路线图（单一来源）：[new_version.md](new_version.md)
- 报告索引：[report/README.md](report/README.md)
- 物理约定：[src/physics/definitions.md](src/physics/definitions.md)
- 数据契约：[report/model502-schema-and-source-protocol.md](report/model502-schema-and-source-protocol.md)
- 决策：[docs/adr/](docs/adr/)
