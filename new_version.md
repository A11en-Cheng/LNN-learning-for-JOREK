# HyPhy-GLNN Model 502 路线图（单一来源）

本文是路线图唯一来源（ADR-0002），取代原 4-Phase 月度版。审计证据保留在
[report/hyphy-glnn-audit-roadmap.md](report/hyphy-glnn-audit-roadmap.md)；数据契约以
[report/model502-schema-and-source-protocol.md](report/model502-schema-and-source-protocol.md) 为起点（Phase 0.2 将升级到 v2）；
术语见 [CONTEXT.md](CONTEXT.md)。所有任务带阻塞边与量化验收；未过 gate 冻结下游扩展。

## 全局约定

- 数据：`data/EHL-2` 下纯 2D（`ntor_1`）run 的 `jorek*.h5` 九场帧；有效帧 = 与 `part*.h5` 编号对齐的帧
  （ADR-0004）。`ntor11` 多环向 run 移出近期训练集，留作 3D 研究材料。
- 门槛（ADR-0002）：split manifest 与仅训练集统计；固定种子单样本过拟合 loss 下降 ≥99% 且无 NaN；
  checkpoint reload 输出一致 ≤1e-6；至少一个 free-run horizon 优于 persistence；checkpointing 梯度最大绝对差
  ≤1e-5；峰值显存低于预算（记录并发占用）。
- 环境：`.torch-cuda-env`（ADR-0003），torch 2.11.0+cu130，RTX 5070 Ti。

## Phase 0 — 数据与接口契约对齐

**0.1 数据 fixture 与 split manifest**（阻塞：无）

- 锁定真实 HDF5 形状（状态 `values (9,4,1,N)`、坐标 `x (2,4,1,N)`）与谐波维语义；核对 2D run 清单、
  有效帧范围（`part*.h5` 对齐后）、跨 run 网格节点数一致性；旧 smoke test 数据路径全部迁移。
- 产出：更新 `tests/test_model502_schema.py` 等 fixture；序列级 train/val/test manifest（附数据版本/hash）。
- 验收：schema/source/迁移测试全绿；manifest 可复现。

**0.2 source 接口对齐（协议 v2）**（阻塞：0.1）

- 按 ADR-0005 建协议 v2：以 marker_spi_source 真实输出为准（`part*.h5` markers + `projections*.h5`
  五场投影 `[dE_ion, rad_sink, Z_eff, Z_imp, N_imp]`），model502 背景源为可选第二通道。
- 对 `mod_elt_matrix(_fft).f90` 做参量面审计：列出装配 9 场方程所需的全部零维参量与闭包，逐个标注
  "已导出 / 需导出 / 由网络吸收（hyper-*）"；给每个数据字段打"原生 / 导出 / 控制"标签。
- 产出：协议 v2 文档 + 参量清单；网络学习目标只保留原生 9 场。
- 验收：源项 schema 测试通过；参量清单与 h5 实际内容一致。

**0.3 参数读取通道**（阻塞：0.1；含 JOREK 侧外部依赖）

- `jorek*.h5` 已含 `eta/visco/visco_par/tstep/central_density/central_mass`；补齐 JOREK 导出
  `ZK_i_par/ZK_e_par`、`D_perp` 基准值等缺失的 Markovian 零维参数（`mod_export_restart.f90` 改动，
  属外部 Fortran 依赖，单独登记）。hyper-* 截断项不导出、不辨识，由网络隐式吸收。
- 缺失系数的兜底策略（待 ADR-0006 确认）：主路径是 JOREK 补导出；用相邻两帧在 CPU 上做约束最小二乘
  辨识仅作交叉校验，不作为精确推导来源。
- parser 读取全部零维参数 → condition vector + mask；数值 0 配 `mask=false`，禁止 1.0 回退冒充真实值。
- 验收：参数解析测试通过；每个参数有"来自 h5 / 缺失"标志。

## Phase 1 — 无源九场单体基线（gate：ADR-0002 全项）

**1.1 非均匀时间步长支持**（阻塞：0.1、0.3）

- 每步 `dt` 取相邻 `t_now` 之差，同时读取每帧 `tstep`；把 `time_steps` 从 trainer 传入 processor（CfC）
  与 PDE loss；明确隐藏状态跨步传递与 BPTT 截断边界（当前 trainer 每步丢弃 state）。
- 跳采样训练：构造 `(s_t, s_{t+k})` 跨步样本（`k=1,2,4,8,...`），以真实 `Δt` 作为 timespan；不重采样。
- 验收：变 dt 前向/反向无 NaN；dt 契约与 state 连续性测试。

容量策略（已确认）：维持现有约 186k 参数架构过 gate；单样本过拟合 gate 不过再扩容，不提前加复杂度。

**1.2 单样本过拟合**（阻塞：1.1）

- 固定 seed，单个小图/短序列（`SEQ_LEN=3` 起步）：训练 loss 下降 ≥99%，无 NaN；reload 后曲线可复现。

**1.3 梯度检查点与显存预算**（阻塞：1.1）

- message passing 封装为可 checkpoint 块；梯度等价 ≤1e-5；峰值显存报告含硬件、batch、mesh、并发占用。

**1.4 free-run rollout 与 persistence 基准**（阻塞：1.2）

- 逐场误差、rollout 曲线、质量漂移、PDE residual；至少 1 个 horizon 优于 persistence；指标无 NaN。

## Phase 2 — 二维含源项训练

**2.1 源项归一化与注入**（阻塞：Phase 1 gate、0.2）

- 源项仅训练集统计；把已存在的 `control_encoder` 接入（`require_source=True`）；无源帧不填充零。

**2.2 条件注入与消融**（阻塞：0.3、Phase 1 gate）

- 零维参数 condition encoder + 消融：固定状态、改变条件，输出/latent 有可检测响应；no-condition 对照。

**2.3 含源 rollout 对比**（阻塞：2.1、2.2）

- 强源项下无 NaN；相对无源 baseline 的误差/稳定性改善可解释并登记。

## Phase 3 — MoE（Phase 2 验收后）

- **3.1 gradient firewall 消融**（阻塞：Phase 2）：先证明 `.detach()` 方向与收益，梯度图测试。
- **3.2 专家拆分 + 最小握手**（阻塞：3.1）：EM/Thermo/Momentum 专家，先 concat/add MLP；Cross-Attention 仅作后续消融。
- **3.3 MoE vs monolithic 消融**（阻塞：3.2）：同 seed/split 对比误差、稳定性、显存、吞吐。

## 研究假设（不承诺时间，各自带数据/单位/阈值 gate）

- R1 完整九方程与 charge-state-aware closure（需物理专家确认单位与压强约定）。
- R2 3D 模态与复频谱（用 `ntor11` 数据；先定模态数据契约与实值 baseline）。
- R3 JOREK runtime 算子级替代（先接口 sandbox 与延迟/误差预算）。
- R4 跨装置泛化（外部数据到位后另立）。

## 外部依赖（阻塞项，需要人）

- marker_spi_source 对齐后的源项数据契约（Phase 0.2 决策 + 数据交付清单）。
- JOREK 侧参数导出改动（Phase 0.3 的 Fortran 依赖，或后处理导出器）。
