# HyPhy-GLNN 领域词汇

本文件是这个仓库的单一领域词汇表。所有 ticket、spec、代码注释都应使用这里的词，不要另造同义词。
权威数据契约见 [report/model502-schema-and-source-protocol.md](report/model502-schema-and-source-protocol.md)；
已结晶的决定见 [docs/adr/](docs/adr/)。

## 词汇

- **Model 502**：九场 JOREK 模型（`n_var=9`），本项目唯一的数据来源。
- **state contract（状态契约）**：每个图状态是 `float32 [nodes, 9]`，字段顺序固定为
  `psi, Phi, Z_j, omega, rho, T_i, v_par, rho_imp, T_e`（对应 HDF5 `values` 索引 2,1,0,3,4,5,6,7,8）。
- **原生量（native）vs 导出量（derived）**：原生量是 model502 的 9 个状态场本身；导出量是投影、粒子统计、
  诊断序列、输运闭包（如 `eta_T, ZK_*_T, D_prof, dcoef`）。网络只学原生量；导出量只作控制输入或诊断（ADR-0005）。
- **高阶截断（hyper-*）**：方程里的超电阻/超粘性/超扩散项（`eta_num, visco_num, ZK_par_num` 及其 T 依赖），
  由网络隐式吸收，不作为输入或辨识目标。
- **Markovian closure（马尔可夫闭包）**：给定当前 9 场状态 + 方程所需的全部零维参数，就能推出下一个（甚至
  后续多个）状态。训练目标正是学这个映射，不依赖更长的历史。
- **skip-sampling（跳采样）**：不做重采样，而是构造 `(s_t, s_{t+k})` 的跨步样本（`k=1,2,4,8,...`），
  以真实 `Δt=t_{t+k}-t_t` 作为 CfC 的 timespan，让网络拟合多尺度演化。
- **transform（变换）**：热力学场 `rho/T_i/rho_imp/T_e` 先 `log1p(clamp(x,0))` 再 Z-score；其余场线性 Z-score；
  统计量只由训练 split 计算。负值被 clamp 属于数据质量事件，必须在数据报告里计数并解释。
- **source-control contract（源项契约 v2）**：两条独立控制通道，都只是控制量、永不作为学习目标。
  主通道 = `projections*.h5` 五场投影 `[dE_ion, rad_sink, Z_eff, Z_imp, N_imp]`（`Data.source [nodes,5]`），
  按帧索引/节点数/时间对齐（投影 t_now 是物理秒，状态 t_now 是 JOREK 单位，换算用 `t_norm`）；
  次通道 = 背景源 `[particle_source, heat_i_source, heat_e_source]`（`Data.background_source [nodes,3]`），
  三者要么全有要么全无。无投影帧 `source_available=false` 是合法无源样本；强制训练用 `require_source=True`。
- **part*.h5（粒子快照）**：marker_spi_source 每帧写出的原始 markers（位置/速度/权重/电荷态分布/出生时间），
  文件名编号与 `jorek*.h5` 对齐；这是源项信息的最原始来源。
- **projections*.h5（源项投影）**：粒子源项投影到 JOREK 场上的五场输出
  `[dE_ion, rad_sink, Z_eff, Z_imp, N_imp]`（use_marker 模式，`n_var=5`）；协议 v2 已以它为主控制通道。
- **marker_spi_source**：JOREK 粒子-流体耦合程序（`particles/examples/marker_spi_source.f90`），
  是 part/projections 数据的产生方。
- **参数通道（parameter channel）**：所有影响下一步状态的零维参数（粘性、热导、高阶平滑等）必须由 H5 输出。
  `jorek*.h5` 已含 `eta/visco/visco_par/tstep` 等标量；`ZK_par/ZK_perp` 与 hyper-* 仍需在 JOREK 侧补导出。
  注意 `dcoef` 是 Dommaschk 平衡位形系数，不是输运系数。
- **tstep 与帧间 dt**：`tstep` 是每帧记录的 JOREK 内部时间步（自适应，约 0.05–5）；相邻帧的 `dt` 是多个
  tstep 之和，实测在耦合阶段约 0.45–50，**非均匀**。物理秒 = JOREK 单位 × `t_norm`；
  训练按每步真实 dt 进行（`step_dt_seconds`），跳采样 `(s_t, s_{t+k})` 以累计 dt 作为 CfC timespan，
  hidden state 跨步携带、BPTT 段边界 detach（#0006）。
- **condition vector（条件向量）**：零维参数通道（`src/data/parameter_channel.py`）输出的固定顺序向量：
  h5 标量 → run namelist（`jorek_namelist_spitest` 优先）→ JOREK 默认值（`preset_parameters.f90`）。
  缺失项以默认值填充并带 `provenance ∈ {h5, namelist, default, derived}`；消费方凭 provenance 判真伪，
  不以值是否为 0/1.0 判断。`time_evol_theta/zeta` 是派生量，用 JOREK 同款逻辑计算（ADR-0007）。尚未接入网络。
- **residual/delta（残差）**：decoder 输出归一化的单步残差 `delta_x [N,9]`；更新规则恒为
  `x_{t+1} = x_t + delta_x`；物理 `dt` 单独传给 LNN 与 PDE 诊断，不混进残差。
- **monolithic baseline（单体基线）**：九场单网络（encoder/LNN/decoder），可无源训练，是 MoE 消融的唯一对照。
- **persistence baseline（持续性基线）**：free-run 指标的下界，即“下一步 = 当前步”的预测。
- **free-run rollout（自由推演）**：超出 teacher forcing 的自回归预测；benchmark 按 horizon 报逐场误差并与 persistence 比较。
- **single-sample overfit（单样本过拟合）**：固定种子在单个小图/短序列上训练；验收是训练 loss 下降 ≥99% 且无 NaN。
- **gradient firewall（梯度防火墙）**：用 `.detach()` 明确每个 loss 项允许更新哪些模块参数，禁止路径梯度严格为 0。
- **MoE handshake（专家握手）**：电磁/热力学/动量专家 latent 的融合；最小形式是 concat/add MLP；Cross-Attention 只是后续可选消融。
- **gate（验收门槛）**：进入下一阶段前必须通过的量化标准，见 ADR-0002。未过 gate 就冻结下游扩展。
- **split manifest（划分清单）**：序列级 train/val/test 分配（禁止按帧随机划分），附数据版本/hash。
- **provenance（来源溯源）**：任何报告的数字必须能回溯到 checkpoint、config、随机种子和数据版本。

## 主要文档

- 路线图：[new_version.md](new_version.md)（单一来源；审计证据在 report/hyphy-glnn-audit-roadmap.md）
- 进度与验证记录：[report/implementation-progress.md](report/implementation-progress.md)
- 运行环境：`.torch-cuda-env`（ADR-0003）
