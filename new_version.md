# HyPhy-GLNN Model 502 路线图（单一来源）

> **2026-10-03：本文件已由 [roadmap_v2.md](roadmap_v2.md) 取代（v2 工作路线图）。**
> 本文件保留为 v1 记录——其中的数据契约、Phase 1.5/2 的历史结论与 ADR 索引仍然有效；
> **目标、训练语义、Gate 口径以 v2 为准**（v2 的 §1 是"v1 留下了什么"的盘点）。

（v1 时期）本文是路线图唯一来源（[ADR-0002](docs/adr/0002-gated-roadmap-and-acceptance-criteria.md)），取代旧版九场
路线（其训练语义见已归档的 ADR-0008/0009）。术语见 [CONTEXT.md](CONTEXT.md)；路线决策见
[ADR-0010](docs/adr/0010-physics-loss-only-at-interpolated-times.md)（Physics Loss 的适用范围）与
[ADR-0011](docs/adr/0011-fragment-source-coupling-architecture.md)（场-碎片-源耦合架构）、
[ADR-0012](docs/adr/0012-evaluation-protocol-blocked-splits-and-loo.md)（评估协议）与
[ADR-0013](docs/adr/0013-subinterval-integral-residual.md)（子区间积分残差）；数据契约见
[report/model502-schema-and-source-protocol.md](report/model502-schema-and-source-protocol.md)；物理约定见
[src/physics/definitions.md](src/physics/definitions.md)。

本路线图不写时间估计：每个阶段只有依赖、产出与 gate；未过 gate 冻结下游扩展。

## 全局约定

- 数据范围：`data/EHL-2` 下 8 个纯 2D run（`ntor=1`，只取 n=0 分量；两个 `*_v200` 为
  标注错误 + 重复数据已于 2026-09-29 删除）；`ntor11*` 的 11 分量数据保留，
  不在本路线图内（ADR-0004）。
- 数据三件套：`jorek*.h5`（九场）、`part*.h5`（neon marker）、`projections*.h5`（五场投影）；
  shard 级 `spi_*` 与逐时间步烧蚀序列在 8 个 run 都存在。所有 run 的注入时刻 `t_ns = 8380`：
  `t < t_ns` 的帧无碎片（`n_spi_tot = 0`），`t ≥ t_ns` 的帧有完整 shard 状态（`n_spi` = 45–846）。
- 单位：模型内部一律 JOREK 单位；物理秒只在报告与投影对齐时出现。
- 耦合时序：目标帧约定（CONTEXT.md）；训练与评估必须按同一配对规则，禁止隐式错位。
- 训练语义：记录帧 = data loss；插值点 = physics loss（ADR-0010）。
- 所有报告数字附 provenance（checkpoint/config/seed/数据版本）。

## 当前状态与决策（2026-10-03）

- **实时物理计算全部停止，全部走 data**（用户 2026-10-03 决定，延续 ADR-0017）：
  Phase 4 的 physics loss / 闭环**暂停**；残差模块只作离线诊断，不进训练回路。
- **背景三源不存在**：namelist 里 `particlesource = 0 / heatsource_i = 0 / heatsource_e = 0`
  （8 个 run 均如此），`projections*.h5` 也只有 5 个 marker 投影通道
  ⇒ **"可选背景三源"从数据需求中删除**；唯一在用的源项就是
  `S = [dE_ion, rad_sink, Z_eff, Z_imp, N_imp]`（按**目标帧**配对，教师源）。
- **主路径**：LTC 解析传播子（`rate=softplus`、`target=tanh`，无 LSTM）+ 残差解码 +
  动力学配平 loss + 一/二阶导数特征 + **显式 dt 特征**；`--save-best-metric closer`。
- **口径（强制）**：报告必须给 **loss / val_mse / R² / MSE / relL2 / `derr/|u|` / closer
  / skill_pers**，并按 **dt 分桶**给出；**必须写清评测窗口集合**（`--inner val|all`）；
  硬 argmax 磁轴已失效，改用"真轴节点 psi 值差"。
- **已撤销/不需做的**：Phase 1.5.1 矩反组装 gate（ADR-0016）、1.5.4 的 P0b（非局域表无需导出）、
  "背景三源导出"（本文件上一版列为 P1，现删除）。
- **主要缺口**（详见下方各 Phase 与 `report/pitfalls.md`）：LOO 全折评测、多 seed 复核、
  参数通道（61 维）未接入网络、shard 状态 `p(t)` 未接入网络、源项自生成（Phase 4.3）、
  空间耦合（GNN/谱）、子帧数据（插值点没有真值）。

## Phase 1 — 数据与状态契约（u, p, S）

**1.1 三件套契约**

- `u`：九场状态，沿用现有 schema 与 parser。
- `p`：碎片状态——shard 级与 marker 归约统一为集合编码，含存活 mask（8 个 run 都可用；
  注入前帧为空集）；注入时刻 `t_ns` 是物理 regime 边界，窗口与 burn-in 均不得跨越；
  单位与 provenance 逐字段登记。
- `S`：五场投影 `[dE_ion, rad_sink, Z_eff, Z_imp, N_imp]`；按目标帧约定与 `(u, p)` 配对。
  ~~可选背景三源 `[particle_source, heat_i_source, heat_e_source]`~~
  —— **2026-10-03 核实：namelist 里这三项恒为 0，H5 也未导出，已从契约中删除。**
- 产出：契约文档与 fixture；数据覆盖报告（每 run 的帧数、p/S 可用性、SPI 事件时刻）。
- Gate：schema 与对齐测试通过；同一窗口内 `(u, p, S)` 帧号一致；覆盖报告可复现。

**1.2 窗口、burn-in 与划分**

- 只用 2D run；窗口必须物理一致：零维参数在窗口内稳定，参数实质变化处切分；短段按登记规则丢弃。
- SPI 注入时刻 `t_ns` 必须作为切分点：`t < t_ns`（无碎片）与 `t ≥ t_ns`（有碎片）是两个 regime，
  窗口、burn-in 与统计量都不得跨过它。
- 每个窗口 = 30 帧 burn-in（只作输入、不算 loss、hidden state 延续）+ 训练/验证段。
- 数据构造只有一套：统一窗口语料；train/val 只是语料上的分配标签（ADR-0012）。
  - 插值协议（模型选择与 gate）：同一 run 内按连续时间块划分 train/val；purge（训练与验证窗口不共享
    任何帧）是硬要求，窗口充足的 run 追加一个窗口跨度的 embargo，不足的 run 保持 purge-only 并记入 manifest；
  - 场景协议（最终 gate）：Leave-One-Run-Out（10 折，每折留一个 run），报告逐 run 指标与分布；
    LOO 结果不得用于调参。
  - 成本受限时可用 5 折分组（每折 2 个 run）近似；"留一个场景家族"（如 v400）只作外推分析。
- 统计量只由该折的训练窗口计算；manifest 附数据版本、hash 与折号。
- Gate：窗口参数不变性测试通过；blocked 划分的块间无帧重叠；LOO 折覆盖全部 8 个 run；
  manifest 可复现；burn-in 段与训练段的分界在数据与代码中一致。

**1.3 数据管线与高频校验**

- **语料更新（2026-09-29）**：两个 `*_v200` run 为标注错误且重复的数据，已删除；
  当前纯 2D 语料为 **8 个 scan_* run**（manifest：29 segments / 697 windows，LOO 折数 8）。
  各 run 的 `abl_history.dat` / `fragments_*` 已用 `jorek2_diagno_spi` 按最新帧重新生成
  （与 H5 的 `xtime_spi_ablation*` 一致到输出精度），细节见 `report/implementation-progress.md`。
- 抽取每帧 `(u, p, S)` 到缓存，同时抽取高频标量序列（`xtime_spi_ablation_rate`、辐射功率、
  能量收支）作为后续预算校验。
- 数据目录里的 `fragments_position.dat`、`fragments_plume_total.dat`、`abl_history.dat`
  在 run 间内容相同（样板文件），不得当作逐 run 真值；逐 run 真值以 HDF5 为准。
- Gate：缓存可重建；帧号/时间对齐测试通过；缓存 I/O 与显存预算有测量记录。

## Phase 1.5 — 状态与源契约（2026-09-29，ADR-0014 前置）

**1.5.1 Eulerian 矩提取与反向组装 gate**

- 从 `part*.h5` 提取逐节点电荷态分辨密度 `n_Z (Z=0..10)` + 速率矩（局部冷却率、电离能率）
  + 空间矩；投影必须使用 JOREK 的 `HH(l,m)*size(l,m)` 组装。
- **硬性 gate**：用提取的矩反向重新组装五个投影通道，必须与 `projections*.h5` 逐帧对齐到
  机器精度；不达标不得进入网络训练（否则网络会把 FEM 截断误差当物理规律去拟合）。

**1.5.2 投影窗口语义与 timebox/fallback**

- 已核实：投影是**单个流体步的时间平均**，一个帧间隔含 50 个流体步。
- Timebox：在限定轮次内争取拿到逐流体步（或帧区间累计）的投影；
- Fallback：用现有投影在帧内做二阶梯形积分近似，并在评估体系中显式记录截断误差；
  用 `macroscopic_vars.dat` 的逐时间步聚合量做交叉校验。

**1.5.3 已存在的高频资产（无需 JOREK 改动）**

- `macroscopic_vars.dat`：逐时间步宏观历史（约 6500 步/run），含 particlecontent、
  thermalenergy、heatingpower、particlesource、bnd_fluxes、energy_conservation 等；
  用作能量/粒子通道的历史特征与预算 gate。
- `nonlocal_array.dat`：非局域烧蚀的面平均表（`psi_n`、`nl_avg_Te`、`nl_avg_ne`，1000 点），
  **不是逐帧表**：JOREK 只在 `nl_avg_Te(1)==0` 时调用一次 `int3d_new`
  （`core/mod_jorek_timestepping.f90:414`），所以非局域表是**初始冻结的一次性快照**，
  整个 run 的非局域烧蚀都用它（在碎片 ψ 处插值）。
  面平均公式已核对（`ne = rho + alpha_e*rho_imp`、`p_e = ne*T_e`、40 个等 ψ 面）。
  **结论：非局域烧蚀可以离线计算**——用这份冻结表 + 记录的 shard `spi_psi_arr`/半径/species
  代入模型 3 的非局域分支，再取 `max(local, nonlocal)`；不需要逐帧导出，也不需要 eqdsk。

**1.5.4 JOREK 侧修改清单（按优先级）**

| 优先级 | 修改 | 位置 | Fallback |
|---|---|---|---|
| P0a | 帧区间平均（或逐流体步）投影导出 —— **未做**（当前用 fallback） | 粒子投影 event / `mod_project_particles.f90` | 帧内梯形聚合 + 截断误差记录（用 `macroscopic_vars.dat` 交叉校验） |
| P0b | **取消**：非局域表是冻结常量，直接用现有 `nonlocal_array.dat` + 离线实现非局域分支 | 无需 JOREK 改动 | — |
| P1 | 逐 shard **逐流体步**烧蚀历史导出（**未做**；每帧粒度的 `spi_abl_arr` 已有） | `communication/mod_export_restart.f90`（`n_inj` 维度改为 `n_spi_tot`） | 现有每帧 `spi_abl_arr` + `xtime_*` 聚合 |
| P2 | `t_birth` / RNG 状态（不做） | 粒子 writer | 统计等价即可 |
| ~~P1'~~ | ~~背景三源导出~~ **删除**：namelist 恒为 0（2026-10-03 核实） | — | — |

**1.5.5 训练与安全 gate（ADR-0014）**

- **L3 消融（强制）**：对比"纯隐式 CfC 潜状态记忆"与"CfC + 显式过去 3 帧源项特征"。
  若纯隐式记忆在长窗口下无法维持 R²，立即回退到显式历史输入方案，不在黑盒环节死磕。
- **L4 防线（硬性）**：所有学习修正乘子（辐射耗散、非局域烧蚀等）必须按先验物理极值
  硬性 clamping；闭环自回归出现异常值时强制截断，保证解析骨架不会在长于一个记录间隔的
  预测中崩溃。

**1.5.6 已确认并已修复的 bug：parser 的 ψ/Z_j 映射对调**

- JOREK `models/model502/mod_model_settings.f90` 定义
  `var_psi=1, var_u=2, var_zj=3, var_w=4, var_rho=5, var_Ti=6, var_Vpar=7, var_rhon=8, var_Te=9`；
  `communication/mod_export_restart.f90:492` 直接拷贝 `node%values`，无重排。
  因此 H5 通道顺序就是 `[psi, Phi, Z_j, omega, rho, T_i, v_par, rho_imp, T_e]`（identity）。
- 仓库 `src/data/model502_schema.py` 的 `H5_VALUES_INDICES=(2,1,0,3,4,5,6,7,8)`
  **把 ψ 与 Z_j 对调**。数据佐证：H5 第 0 通道 max=0.5032 ≈ `psi_axis_t`=0.5035，
  且 2673/4125 个节点 ψ_n∈[0,1]；第 2 通道（被当作 ψ）范围 −15~0.7，是 Z_j。
- 影响范围：parser 输出、manifest 统计、physics loss（current/vorticity definition 等）、
  椭圆代理（其"psi->Z_j"实际输入输出被交换）、逐场诊断。**修复后必须重跑受影响的验证与旧结论**。
- 修复动作（P0，**2026-09-29 已执行**）：`H5_VALUES_INDICES` 改为 identity（并加回归测试），
  重建 manifest 与 frame cache，重新检查 `Delta* psi -> Z_j`、物理残差与代理结果。
- ψ 语义补充（源码核实）：H5 通道 0 是**当前总极向磁通**（平衡 + 扰动），
  `get_psi_n` 直接用它配 `ES%psi_axis/psi_bnd`（后者由 `bootstrap_find_minRad` 从当前场求出）；
  平衡分量是单独的数据集 `psi_eq`。`eqdsk` 只在建网格时使用（`finish_grid.f90` 等给新网格填
  初始 psi 与壁几何）；重启文件已含 `values` 与 `psi_eq`，重启演化与离线 ψ_n 重建都不需要 eqdsk。

## Phase 2 — 碎片→投影与碎片动力学

**2.1 统一碎片编码**

- shard 级字段与 marker 归约特征共用一个 permutation-invariant 编码器（DeepSets/attention），
  支持变长碎片数与存活 mask；所有 8 个 run 的 post-`t_ns` 帧都可参与，且都带 shard 级监督。

**2.2 源项映射子网络**

**2026-10-03 评估：shard → 投影场（`S_pred`）可行性**

- **数据上可行**：`part*.h5` 每帧都带 16 个 shard 字段
  （`R, Z, phi, phi_init, Vel_R, Vel_Z, Vel_RxZ, radius, abl, species, vol, psi, grad_psi`
  + 3 个 drift 变体）与存活 mask；8 个 run 共约 1417 个 post-`t_ns` 帧，
  逐帧都有 `projections*.h5` 的 5 通道作为监督 ⇒ **配对样本现成**。
  shard 数 45–846 且随时间减少（消融死亡）⇒ 编码器必须 permutation-invariant（DeepSets/attention）。
- **物理上大头是"确定性装配"**：记录的投影 = JOREK `mod_project_particles.f90`
  用 `HH(l,m)*size(l,m)` 把**逐个 shard 的源强**装配到 4125 节点上的结果。
  真正不确定的是**源强的分配**（烧蚀率、离/电子能量分配、辐射、电离），
  而这一层解析骨干（Parks/NGS + 非局域）已能做到逐 shard 中位比 1.00–1.09、
  log-corr ≥0.98、总烧蚀率误差 ~5%（Phase 2.2/2.3）。
- **结论**：让神经网络"从碎片直接回归投影场"在数据上完全可行，但**不建议端到端替代**——
  我们早先的学习修正版本反而更差（ADR-0015：中位比 0.52–0.66、总原子量 −47%）。
  推荐结构 = **解析骨干给出每 shard 源强 → 网络只学小的乘性修正（烧蚀率/能量分配）
  → 再用同一套 FEM 基函数确定性装配**（装配部分绝不能学，否则会把 FEM 截断误差
  当成物理规律拟合，正是 1.5.1 原本要 gate 的东西）。
- **语义前提**：投影是"单流体步的时间平均"，而 shard 状态是逐帧的 ⇒ 配对时必须以
  该帧的 shard + 该帧记录的投影为准，并登记窗口语义（1.5.2）。
- **收益**：一旦 shard → `S_pred` 可用，就**不再需要"未来帧的投影"**（当前网络是插值器：
  控制项取目标帧的教师源）⇒ 自回归/长 horizon 预报与 Phase 4.3 闭环才成立。

- 输入：碎片编码 + 碎片邻域的局域场特征；输出：五场投影。
- 以 NGS/Parks 解析值为基线，网络只学修正（如乘性/对数残差），保证非源区输出可忽略。
- 监督：`projections*.h5`；辅助监督：`xtime_spi_ablation_rate`（当前 HDF5 导出是**全碎片
  求和的总量**，逐碎片逐内部步需要 JOREK 侧补导出）。
- 解析基线（`report/spi-analytic-baseline.md`，2026-09-29 更新）：模型 3 的**非局域分支已可离线
  重建**（`src/data/nonlocal_table.py` 逐帧重算 40 面 `<T_e>/<n_e>` 表 + JOREK 的查表与
  `max(local, nonlocal)`），8 个 run 上逐 shard 中位比 1.00–1.09、log-corr ≥0.98、
  85–96% 落在 2× 内、总烧蚀率误差 ~5%。（磁盘上的 `nonlocal_array.dat` 只是最后一次调用
  的快照，不能当逐帧表用。）
  学习修正因此只覆盖残余模型误差（k-NN vs FEM 采样、电荷态细节等），不再承担结构性偏差。
- Gate：在 held-out SPI run 上逐场相对 L2 优于解析基线；烧蚀率误差与原子/能量预算闭合；
  子网络输出对碎片位置/半径的响应方向正确。

**2.3 混合碎片动力学**

- 场→碎片一步 = JOREK 解析骨干（位置推进、RKT 加速度、半径侵蚀、NGS 烧蚀）+ 学习修正
  （乘性烧蚀修正、可选力修正）。
- 解析骨干已含非局域烧蚀分支（2026-09-29）：held-out run 上解析-only 即可达
  半径 MAE 1.33e-7、烧蚀中位比 1.02、within-2x 90.7%、总烧蚀量误差 +2.0%、区间中位误差 4.0%
  （`output/fragment_dynamics/v402`）。**学习修正的定位随之改变**：它不再负责补结构误差，
  需要重新调参或降级为可选残余项；v402 中学习修正反而使上述指标变差（中位比 0.52–0.66、
  总原子量误差 −47%）。
- **收口（ADR-0015，2026-09-29）**：Phase 2.3 交付物 = 非局域解析骨干；学习修正暂停、
  仅作消融（`--analytic-only` 登记交付物，见 ADR-0015 的重新启用条件）。帧粒度 gate 的
  登记数字取 `output/fragment_dynamics/v402_analytic/metrics.json`；速度增量精度、帧内 50 步
  演化与长窗口自回归稳定性留给 Phase 3/4。
- 监督：全部 8 个 run 的 post-`t_ns` shard 轨迹与逐时间步烧蚀序列；死亡碎片按 mask 处理。
- 生成数据使用的模型已核对：`spi_abl_model = 3`（Parks/NGS + `nonlocal_abl = .t.`）；
  `spi_rkt_model = 6` 是 PELOTON 的 JET 标定（`p_asymm(1) = 1e5·(T_eV/2e3)^1.26·(n_e/1e20)^0.4` Pa，
  加速度 = `p_asymm·πr²/m_frag`），已在 JOREK 工作树 `models/pellet_module.f90` 中；
  实现时需要局域 `T_e/n_e`、shard 半径、species 与 `spi_rkt_H_m_mod`（默认 1.0）。
- 监督来源是全部 8 个 run 的 post-`t_ns` 帧（共约 1417 帧）；注入前帧只可用于场侧预训练。
- Gate：解析-only 基线 vs 混合的对照；半径/烧蚀率保持非负；预测长于一个记录间隔时不发散。

**2.4 与 ADR-0005 的边界**

- 主状态模型仍只以九场原生量为学习目标；本阶段两个模块是辅助闭包，允许以导出量为监督目标（ADR-0011）。

**2.5 隐式 CfC 架构（2026-09-29 更新，ADR-0014）**

- **2026-09-29 修订（ADR-0016）**：Phase 1.5.1 的 marker→projection 逐系数反组装 gate
  **取消**；Eulerian 矩从"CfC 前置条件"降级为"仅 Phase 4.3 闭环自生成源的可选路径"，
  ADR-0014 的"矩充分性"开放契约随之撤销。源项路线 = 记录投影（教师源）+ 确定性解析/ADAS
  （闭环）。投影窗口语义（单流体步时间平均）保留为已知事实，对 Phase 3 不构成阻塞。

- 2.1-2.3 的编码器/解析骨干/xtime 监督保留为组件与消融；主路径改为：
  Eulerian 网格投影矩作为粒子物理子状态 + CfC 潜演化（含增广源项积分器 `A(t)`）+
  物理约束解码（`S_pred = S_phys * exp(M)`）。
- 融合采用双向局部 cross-attention（碎片->节点用 FEM 形函数插值 `T_e/n_e`；
  节点->碎片限定 `R_cutoff`），相对位置编码包含磁通坐标 `rho` 偏离。
- 训练采用课程式 teacher forcing（强 teacher forcing -> Dagger 自回归 -> 长窗口纯预测）。
- 实现前必须确定两个数据契约：投影窗口语义（单流体步平均 vs 帧间隔累计）与
  矩集合的充分性（电荷态分辨 vs 压缩矩）。见 ADR-0014。

## Phase 3 — 场模型与非均匀时间插值（纯数据）

**3.0 CfC 的接入时间线与输入清单（2026-09-29 澄清）**

CfC 不是"现在就吃碎片"的模块；它按阶段逐步接入，输入的边界如下表。

| 阶段 | 由谁演化 | CfC 的输入 | 该阶段谁提供 p 与 S |
|---|---|---|---|
| Phase 3（纯数据场模型） | 九场 `u` | burn-in 段的记录帧九场编码（潜状态）+ 0D 参数通道 + 真实逐帧 `dt`（随机 `α` 两段积分） | 不出现粒子：源项用**教师源**（记录投影） |
| Phase 4.1–4.2（插值点物理损失） | `u`（CfC）+ 源项残差 | 同上 + 子区间 `δ`；RHS 需要 `p(t*)`、`S(t*)` | `p` 由解析弹丸骨干给，`S` 由源项网络给（教师源对照） |
| Phase 4.3（闭环） | `u`、`p`、`S` | 同上 + 上一步自生成的 `p_prev/S_prev`；粒子侧为 Eulerian 矩（节点特征） | 全部由模型自生成（教师 → 半闭环 → 全闭环） |

解析弹丸骨干（含非局域烧蚀）的角色：Phase 3 不出现；Phase 4 起是 `p` 的先验/托底与
`S_phys` 基线（`S_pred = S_phys · exp(M)`，修正有界），不会因为 CfC 接入而废弃（ADR-0015）。

**3.1 CfC 随机两段积分**

- 真实逐帧 `dt`；随机 `α`；两次 CfC cell 调用；hidden state 连续；中间场不 detach；
  跨物理一致窗口 reset。

**3.2 纯数据目标**

- 只在记录帧计算 data loss；中间时刻没有标签，其约束来自组合自洽：两段积分必须复现下一记录帧。

**3.3 源项输入模式**

- 本阶段显式使用教师源（记录投影），并在日志与 checkpoint 中标记；自生成源模式留到 Phase 4。

**3.4 损失与结构度量（2026-09-29 补充，必须执行）**

背景：Phase-3 当前的目标是 z-score 空间的**逐节点 × 逐场 MSE**。它有两个已知盲区：

- **"MSE 小但拓扑错"**：位置/相位错位、极性/符号错误、小尺度结构被抹平，都可以让逐点均方误差
  很小；z-score 又把大振幅场（`rho`、`T`）的误差放大、结构关键的场（`Phi`、`omega`、`Z_j`）
  淹没。
- z-score 按**全场 std** 归一化，磁轴 / LCFS / X 点这些信息量最大的区域没有任何额外保护。

因此 Phase 3 的评估与训练分层补充：

1. **结构指标（先做，仅评估）**：逐场 **H1/梯度相对误差**
   `‖∇u_pred − ∇u‖ / ‖∇u‖`；**通量面与拓扑量**（磁轴位置 = 最大 ψ 节点、`ψ_n` 等值线 /
   LCFS、X 点位置，与帧内 `psi_axis_t`/`psi_bnd_t` 对比）；**谱度量**（逐场 2D 功率谱或
   沿 ψ 的模分解）；**积分预算**（总电流、粒子数、热能、辐射总量 vs `xtime_*`）。
2. **结构损失（后做，必须消融）**：H1/Sobolev 项 `λ_H1·Σ_fields‖∇u_pred−∇u‖²`；
   逐场权重与活动度加权；Phase 4 的物理残差本身就是拓扑敏感约束（见 4.1/4.2）。

- Gate：不同 `α` 划分的插值自洽性在登记容差内；无 NaN；checkpoint reload 输出一致；
  单样本过拟合 gate 沿用 ADR-0002；**新增：逐场 R² 与 H1 相对误差同时达标、
  磁轴/`ψ_n` 结构量在登记容差内**（防止"MSE 达标但拓扑错"）。

**3.5 空间耦合与导数表示（2026-09-29 补充）**

背景与事实核实：

- 当前 Phase-3/4 模型是**逐节点 CfC**（无消息传递），`edge_index/edge_attr/pos` 只进损失/算子；
  网络内部拿不到空间梯度 ⇒ H1 误差与磁轴偏差的根源之一。
- **JOREK 已经把导数存进切片**：`jorek*.h5` 的 `values` 是 `(9 var, 4 dof, 1 tor, 4125 node)`，
  4 个 dof 是 bicubic Hermite 系数（值、`∂/∂s`、`∂/∂t`、`∂²/∂s∂t`；实测 psi 四分量 RMS
  0.252/0.752/0.056/0.454）。**当前 parser 只读 dof 0**，导数信息被丢弃后再用图最小二乘重算。
- 导数还原只需要**单元局部几何**（`vertex`、`size`、节点坐标 `x`，全在 H5 里），不需要整张图；
  公式与质量矩阵同源（`u_s,u_t → ∂u/∂R,∂u/∂Z`，`J = x_s y_t − x_t y_s`）。
- 成本：图 LSQ `compute_gradient` = **2.73 ms/场**（32,186 条边，9 场 ≈ 25 ms，无法每步调用）；
  Hermite dof 的 FEM 一致导数 ≈ **微秒级**，二阶算子（`Δ*ψ`、`Δ+Φ`）同量级。
- 环向结构：JOREK 极向导数在极向面内**实空间**（Hermite），环向导数**谱空间**（`∂/∂φ→i·n`），
  非线性项在 Gauss 点回到实空间做乘（伪谱）。本数据 `ntor=1`（n=0）⇒ 环向导数恒为零。

决定：

1. **扩 parser/store**：读全部 4 个 dof，并把 `vertex/size/x` 带入 `RunFrameStore`；
2. **写 FEM 一致导数模块**（∇、`Δ*`、`Δ+`），用于 jet 输入与损失算子，替换图 LSQ；
   图 LSQ 保留为消融/兜底；
3. **局部导数特征输入消融**（local-derivative features；不叫 "jet"，避免与 JET 装置混淆）：
   A=`[u]`(9) / B=`[u,∇u]`(27) / C=`[u,∇u,Δu]`(36)；梯度在**模型解码场**上现算
   （形成空间反馈）。教师强制步的输入 jet 不提供新信息（确定性函数），价值在反馈与输出约束；
4. **jet 输出头**：多任务输出 `[u, ∇u, Δu]`，用记录 dof 监督，与 `λ_H1` 做对照；
   `λ_H1` 保留为消融项，由数据决定是否降级；
5. **(j, ω) 不替代 (ψ, Φ)**：`j=Δ*ψ`、`ω=Δ+Φ` 是椭圆算子的像，反解需要全局求解且演化方程
   本就写在原始场上；它们只作为**辅助输出头**与**定义类残差**使用（后者已是物理损失三通道之二）；
   **口径澄清（2026-09-30）**：这里的"辅助输出头"指**额外**的多任务输出，**不是**把
   `Z_j`/`ω` 从状态里降级——它们仍是完整状态通道（实测通道、携带最大增量，且
   `η(Z_j − Δ*ψ)` 正是驱动 ψ 电阻扩散的项；其偏差比场本身粗 8.8×/14×，
   说明该偏差目前被我们算子的节点噪声主导，见 `pitfalls.md` #21 口径更正）。
6. （可选，另立 ADR）**磁通坐标 `(ψ_n, θ)` 谱分支**：θ 方向 FFT + ψ_n 方向 Chebyshev/样条，
   用于全局拓扑（磁轴、X 点、模数 m）；极向面直接的 2D FFT 因网格弯曲/非周期不适用。

## Phase 4 — 插值点 Physics Loss 与闭环

> **2026-10-03 状态：暂停（用户决定，实时物理计算全部停止，全部走 data）。**
> 本 Phase 的内容保留为"将来重启时需要什么"的清单，不进入当前训练回路；
> 重启前置条件见 4.2 补充二末尾（整体隐式矩阵或弱形式演化）。
> 另外：`particle_source / heat_i_source / heat_e_source` 在 namelist 里恒为 0，
> 4.1 的源项清单里**只有 5 个 marker 投影通道**要接。

**4.1 完整残差与源项**

- 启用全部九个残差；把五场投影接入对应方程：密度粒子源、杂质 `N_imp`、离子 `heat_i`、
  电子 `heat_e / dE_ion / rad_sink`、平行动量源；`Z_eff` 进电阻率/辐射闭合。
- 使用相对归一化；验收用"扰动单调性 + 预算校验"，不是零本底。

**4.1 补充（2026-09-29）：系数来源、残差本底与通道启用顺序**

- **物理系数一律从当前帧的 H5/parameter channel 读取**（`eta, visco, visco_par,
  visco_par_par, ZK_i_par, ZK_e_par, D_perp[1], ZK_i_perp[1], ZK_e_perp[1], F0`），
  不再使用残差模块的内部默认值；窗口必须落在同一组系数内（manifest 已按 `material`
  判据切分，`rel_threshold` 收到 1%）。
- **残差"本底"的定义**：在两个相邻**记录帧**上评估同一套残差（与模型无关）——
  它是物理损失的下界，也是 λ_phys 的标定基准；模型侧的对照是在同一区间内由 CfC
  插值出的两个状态（`t*=α·dt`、`δ=ρ·dt`）上的残差。
- 实测（真实系数）：`current_definition` 0.077→0.946、`induction` 0.134→0.981、
  `vorticity_definition` 0.231→4.637（本底 ≪ 信号，可直接用作损失）；
  而 `ion_energy`/`electron_energy` 本底**饱和到 ~1.0**——原因是
  `ZK_i_par=1.67e5`、`ZK_e_par=4.1e6` 使 `κ∥·div_par(T)` 成为刚性主导项，
  节点配点算子的离散误差同时进入分子与分母。
- **第一轮 `λ_phys` 的启用集合**：`induction` / `current_definition` /
  `vorticity_definition`（`density`/`impurity` 作为边际项后加）；
  `ion_energy`/`electron_energy`/`vorticity`/`parallel_momentum`
  **在 κ∥ 离散改进前不进入损失**。
- 工作项：κ∥ 的物理一致离散（通量坐标 / FEM 一致的平行算子；或把 κ∥ 项从相对范数中
  剥离、单独做弱形式约束），完成后回归全九方程。

**4.1 补充二（2026-09-30）：本底改成"分布 + 逐通道归一化"**

- 上面那组 0.134/0.077/0.231 只是**一个窗口一个区间**的读数（旧 30+1 口径），不足以标定
  λ_phys。新工具 `tests/check_residual_floor_stats.py` 在**全部 8 个 run 的 196 个 val
  窗口 / 588 个监督区间**上逐区间评估（系数取区间起始帧、源取记录投影），产物
  `.scratch/n/residual_floor_stats.json`：

  | 通道 | 中位 | p10 | p90 | 逐 run 中位范围 |
  |---|---:|---:|---:|---|
  | `induction` | 0.422 | 0.119 | 0.677 | 0.17–0.53 |
  | `current_definition` | 0.085 | 0.074 | 0.152 | 0.07–0.15 |
  | `vorticity_definition` | 0.048 | 0.027 | 0.159 | 0.016–0.44 |
  | `vorticity` / `density` / `impurity` | 0.95 / 0.69 / 0.74 | — | — | — |
  | `parallel_momentum` / `ion_energy` / `electron_energy` | ≈1.00 | — | — | 本底饱和 |
  | `charge_state` | 0.000 | — | — | 恒等式，**不进损失** |

- 本底是"记录轨迹在我们这套离散算子下的不自洽度"。**来源拆分已实测**
  （`tests/check_residual_time_scheme.py`：同套空间算子，仅换时间差分）：中心差分相对后向
  差分只把 `induction` 本底从 0.177 降到 0.164（−7%），两条定义类通道与时间差分无关
  ⇒ **本底由空间算子（图 LSQ / 节点配点）的离散误差支配，不是 JOREK 的时间格式**。
  因此 θ/Crank–Nicolson 残差**不做**；λ_phys 的靶子是"把模型残差压到**本底量级**"，
  而不是压到 0；要真正降低本底只能升级空间算子（FEM 弱形式刚度矩阵，独立工作项）。
- **逐通道归一化是必需的**：三条通道的残差**对模型输出的梯度尺度差 ~65×**
  （实测 `|∂r/∂u|`：induction 2.0e-2、current_definition 1.1、vorticity_definition 6.1，
  见 `report/pitfalls.md` #27），等权平均时 `induction` 的权重被压到 ~1%，λ_phys 扫到 0.1
  它都纹丝不动（比值恒为 7.39）。因此物理项改为
  `mean_c( w_c · r_c / floor_c )`（`floor_c` 用上表的中位/p90，`w_c` 默认 1），
  让 λ_phys 有"残差 = k 倍本底"的可解释含义。
- 旧结论的作废声明：修正前的 λ_phys=0.05 实验（30+1 口径）只对**窗口最后一个区间**生效，
  且被 pitfall #26 的变量覆盖 bug 影响，不能作为工作点依据。

**4.2 插值点评估（子区间积分残差）**

- 主路径：在 `t*` 取 micro-interval `δ = ρ·Δt`（`ρ` 随机，如 0.1–0.3）；CfC 闭式给出
  `u(t*)` 与 `u(t*+δ)`；碎片模块给 `p(t*)`，源项网络给 `S(t*)`；残差用
  `M·[u(t*+δ) − u(t*)]/δ − RHS(u(t*), p(t*), S(t*)) ≈ 0`（或两点梯形）。
  这是 JOREK 自身的弱形式时间离散，两个状态都由模型给出，不使用数据有限差分（ADR-0013）。
- 解析 `∂u/∂t` 降为消融路径；若启用，必须同时启用稳定化组合：全局梯度裁剪与逐残差梯度范数标定、
  `λ_phys` warmup/monitor-only、rate 的空间与子步平滑正则、导数可信度 gate
  （与模型自评的短区间差比较）、保留 float64 残差组装与 `_clamp_physical`。

**4.2 补充（2026-09-30）：信号/本底比是验收量**

- 工具 `tests/eval_phase4_physics.py`：对 held-out 每个监督区间同时给出
  **记录轨迹本底**与**模型插值轨迹（t*=α·dt → t*+ρ·dt）残差**，输出比值。
  dense 工作点（λ_H1=0.7、λ_axis=0.3）data-only 基线（held-out v400，18 区间）：

  | 通道 | 本底（中位） | 模型（中位） | 比值 |
  |---|---:|---:|---:|
  | `induction` | 0.135 | 0.995 | 7.4 |
  | `current_definition` | 0.077 | 0.866 | 11.3 |
  | `vorticity_definition` | 0.223 | 1.938 | 8.7 |

  即插值轨迹"违反离散方程"的程度是记录数据自身的 7–11 倍——这部分正是模型**导数级**
  精度（梯度/H1 相对误差仍 36–60%）的直接体现，也是 Phase 4 要压的目标。
- 训练时 `train_phys` 会随数据拟合**上升**（0.61 → 0.95）：模型越锐利，子区间残差越大，
  说明物理项在等权平均下压不住高频误差。归一化（上一节）之后要重新登记这条曲线。

**4.2 补充二（2026-09-30，同日更晚）：残差参数化 —— Phase-3 基线重定**

- 触发：`induction` 通道在 8 次训练里恒饱和（pitfall #28），追下去发现
  **增量根本没学会**（pitfall #29）：新增 `dR2`（增量 R²，0 = persistence，
  见 `tests/eval_phase3_structure.py`）后，绝对参数化模型**9 个场的 dR2 全为负**
  （psi −155、v_par −6.6、Phi −3.6 …），而场级 R² 0.85–0.99 全靠"记录场帧间几乎不变
  （1–20%）"这一事实撑着。**此前所有 Gate（R²/H1/磁轴）都在测"像不像上一帧"。**
- 修法：`--residual`，即 `u(t+dt) = u(t) + decode(...)`（decoder 输出增量）。
  20 epoch 就全面超越绝对参数化的 100 epoch：

  | 指标 | 绝对 20ep | 绝对 100ep | **残差 20ep** |
  |---|---|---|---|
  | psi R² / H1 / dR2 | 0.971 / 0.44 / −155 | 0.995 / 0.19 / −26.5 | **0.9998 / 0.035 / −0.28** |
  | z 轴中位 [cm] | 3.20 | 1.67 | **0.99** |
  | residual 比 (ind/cur/vort) | 7.38 / 11.31 / 8.69 | 7.33 / 9.97 / 10.85 | **5.95 / 2.29 / 6.73** |

- 同时修的链长不一致：数据链改为 `α·dt + (1−α)·dt`（原来收尾用的是 `ρ·dt`，
  总时长 `(α+ρ)dt ≠ dt`，与评估用的 `two_substep` 不一致）；物理残差直接用第二个子步
  （`δ=(1−α)dt`，两端都是模型状态）。`--phys-rho` 因此废弃。
- **验收口径变更**：Phase-3/4 的 Gate 一律以 **dR2**（增量）为主轴，场级 R² 只作参考；
  物理损失在残差参数化基线上重扫（本底归一化 + 分母 detach 默认开）。

**3.4 补充三（2026-09-30，Gate 重跑结果与新基线）**

- **归一化/物理性核查**：反归一化链路（`destandardize` + `inverse_transform_state`）
  本身正确，物理残差一直在真实单位上算；但 **z-score 空间的 loss 没有按动力学配平**——
  `E[(Δu_norm)²]` 逐场相差 5105×（Phi 0.87 vs psi 1.7e-4），平 MSE 等于只学 Phi/omega。
  已加 `--loss-scale dynamics`（`w_f = 1/E[Δ_f²]`，训练集统计），目标函数＝场平均增量 R²。
- **Gate 重跑**（残差网络 + 动力学配平，λ_H1=0.7 / λ_axis=0.3，20 epoch，held-out v400）：

  | 判据 | 阈值 | 实测 |
  |---|---|---|
  | 场级 R² | ≥ 0.85 | 0.938–0.9998 ✅ |
  | H1_rel | ≤ 0.7 | ≤ 0.440 ✅ |
  | 磁轴 中位/p90/max | ≤ 4 / 6 / 8 cm | **0.77 / 3.38 / 3.75 cm** ✅ |
  | dR2（新增） | > 0 | 6/9 场为正；**omega −0.45** ❌ |
  | 物理比 ind/cur/vort | — | **5.91 / 2.06 / 2.13**（旧 7.38/11.31/8.69） |

- **待办（按优先级）**：① H1 项逐场配平（现 365× 失衡，rho+rho_imp 占 86%）；
  ② omega（涡量）动力学专项；③ 物理损失在新基线上重扫；④ `log1p(clamp)` 的
  噪声级不可逆（1.7% rho_imp 节点）→ 可选改保号 symlog（数据契约变更）。

**3.4 补充四（2026-09-30 晚）：symlog 契约 + H1 配平 + 新工作点**

- **数据契约 → `model502-state-v2`**：4 个正量场改保号 symlog（`sign·log1p(|x|)`），
  往返处处精确（原 `log1p(clamp)` 丢 1.68% 的 rho_imp 噪声级负值）。
  对训练数值影响 ~1e-7（纯正确性修复）。
- **H1 逐场配平**：`mean_c(‖∇e_c‖²/‖∇t_c‖²)` 取代全局比（旧隐含权重 365× 失衡，
  rho+rho_imp 占 86%）。配平后 λ_H1 有效强度变化大，重扫结果（平均 dR2）：
  0 → 0.138、**0.01 → 0.151（最优）**、0.03 → 0.119、0.1 → −0.074、0.3/0.7 → ≈−0.09。
- **新工作点**：`--residual --loss-scale dynamics --lambda-h1 0.01 --lambda-axis 0.3`
  （`gate4_h1pc_0.01`）。Gate：R² 0.958–0.9998、H1 ≤0.451、轴 **0.39/3.43/3.79 cm**、
  dR2 **7/9 为正**（Phi +0.07、omega +0.02、rho_imp +0.44、T_i +0.39、rho +0.39、T_e +0.25）、
  物理比 6.74/3.19/**1.16**。
- **omega 归因**（pitfalls #33）：主因是**正则化/损失配平**（H1 配平 + λ↓ 后 dR2
  −0.45→+0.02）；但模型**完全没有空间耦合**（无 edge_index/消息传递），
  输运项不在模型类内，线性局部上界只有 −0.29 ⇒ **空间耦合是把 omega/Phi 做上去的
  必要条件**。下一步：① 把物理输运项（`{Φ,ψ}`、`{u,ψ}`、advection）作为输入特征
  （最便宜，先验证）；② 极向面 GNN/stencil 分支。

**3.4 补充五（2026-09-30 晚二）：单位制、导数特征实验、psi/v_par 失败模式**

- **单位制（已核实，写入 `CONTEXT.md`）**：全链路 JOREK 内部单位——长度**米**
  （网格/R_geo/R_axis 一致 ⇒ 磁轴 cm 标签有效）、时间 Alfvén 单位
  （`t_norm=4.0858e-7 s`，dt≈10 ⇒ 4.1 μs）、场量 JOREK 归一化（psi_axis 0.506）。
- **导数特征（用户假设）成立但增益有限**：`{Φ,ψ}` 是一阶导数的局部函数，喂入 18 个一阶导后
  逐点 MLP 原理上可表示输运项。实测 `gate5_inputderiv`：psi dR2 −0.212→**+0.079**
  （amp 0.726→0.537，过冲收敛）、omega +0.022→**+0.049**；代价是 4 个热力学场各降
  0.02–0.04，平均 dR2 0.151→0.166，物理比略变差（1.16→1.43）。⇒ 局部 Taylor 只是
  输运算子的粗糙代理，**真正需要的是模型内有空间算子**（或把 RHS 输运项本身作输入）。
- **psi / v_par 的正确读法**（新增 `corr`/`amp` 两列，`dR2 = 2·corr·amp − amp²`）：
  psi 是"方向弱 + **幅度过大**（amp 0.73）"，且其增量仅占自身 1.35%，
  与解码噪声底同量级 ⇒ **被精度底卡住，不是被权重卡住**（其动力学权重已是全通道最大）；
  正确解法是**降低解码噪声底**（谱/FEM 低维解码）。v_par 相反：增量占自身 9.1% 不小，
  但输出 amp 0.05 ≈ 纯 persistence ⇒ 与输运项同源，走空间信息那条路。

**3.4 补充六（2026-09-30 晚三）：pitfall #21 重验 + 全通道一/二阶导数特征**

- **#21 重验（旧结论取自大 bug 之前，不可靠）**：几何自检变成**机器精度**
  （`lap(R)` rms 1.7e-11），旧的"链式法则有 5% 误差"作废；**FEM(dof) 二阶装配仍坏**
  （`lap(psi)` p50 = 1.5e3，应 ~2，系统性量级错误，待修，已标注禁用）；
  但**图 LSQ 二阶算子是对的**：`Δ*ψ` vs 记录 `Z_j` corr **0.994**/relL2 0.11、
  `Δ+Φ` vs 记录 `omega` corr **0.997**/relL2 0.08 ⇒ "Z_j/omega 不是逐点二阶导"
  的旧结论**过强**，它们数值上就是逐点 `Δ*ψ`/`Δ+Φ`（~10% 来自 JOREK 的电阻松弛）。
- **二阶信息可作特征**：新增 `--input-deriv2`（9 值 + 18 一阶导 + 9 图 LSQ 拉普拉斯 = 36 维）。
  实测（同工作点，20 epoch）：**平均 dR2 0.151 → 0.166（仅一阶）→ 0.194（一阶+二阶）**，
  热力学四场 corr 升到 0.74–0.84、dR2 0.31–0.49，**v_par 首次转正**；
  代价是 Phi/omega 略退、物理比变差（更锐利的场 ⇒ 二阶残差更大）。
  ⇒ 下一步用物理损失（本底归一化 + 分母 detach）压这些残差，并把二阶特征纳入基线候选。

**4.3 闭环**

- 教师源 → 半闭环（教师 p、预测 S）→ 全闭环（预测 p 与 S）；每步有稳定性 gate。
- Gate：残差有界且随预测偏离单调增大；原子/能量/辐射预算与高频序列一致；
  闭环在登记 horizon 内有限；data loss 相对纯数据阶段不劣化超过登记容差。

## Phase 5 — 验证与 benchmark

> **2026-10-03 现状**：协议（blocked 内层 + LOO 折）在 manifest 里已就绪，
> 但**只用过 v400 一个折**；horizon 阶梯（H1–H4）与"可接受降级 gate"尚未登记；
> 所有结论都是**单 seed**（路线图要求多 seed）。**当前可做、不需要新数据**的三件事：
> ① 8 折 LOO 全量评测；② 关键配置 2–3 个 seed 复核；③ 把"评测窗口集合 + dt 桶构成"
> 写进每张 gate 表（见 pitfalls #58：换窗口集合会让 R² 从 0.994 掉到 0.974）。

- 协议：插值协议（blocked 内层划分）用于模型选择；场景协议 LOO（10 折）用于最终 gate（ADR-0012）。
- 装置只有一个（EHL-2），本阶段不做装置/几何泛化；run 之间的差异是注入场景
  （速度 −289/−385/−391 m/s、角度 12°/15.5°、`n_spi` 45–846、`spi_quantity` 7e21–0.922e22）。
- horizon 阶梯（相对 `t_ns` 与数据定义的热猝灭窗口登记，JU 与 ms 双记）：
  H1 插值 1–2 个帧间隔（0.4–40 µs）；H2 猝灭前段 0.1 ms；H3 猝灭窗口（热能 10–90% 下降，
  约 0.4–1.0 ms）；H4 全 post-`t_ns` 记录（0.8–1.9 ms / 77–621 帧）。
- 可接受降级 gate（Phase 1 校准，且必须让 data-only/persistence 基线失败）：全程无 NaN；
  猝灭时刻 ±0.1 ms 或 ±20%；累计烧蚀原子数与电子热能降幅 ±10–20%、辐射能量 ±20%；
  无虚假猝灭/恢复、无冻结坍缩；失相关之后按分布与预算评判。
- 指标：记录帧逐场误差、不同 `α` 的插值自洽性、源项误差、烧蚀率/能量/辐射预算、
  猝灭时刻与降幅、闭环稳定 horizon、推理成本。
- 对照：data-only CfC（无 physics loss）、解析源闭环、混合源闭环；ROM 对照按需引入。
- 多 seed 报告；禁止单 seed 能力声明。

## 研究假设（不承诺）

- `ntor11` 的环向模耦合（`ELM_p/ELM_n/ELM_k/ELM_kn` 为参考结构）另行立项。
- EXL-50U 跨装置验证是外部数据依赖。
- ROM/降阶模型对照按需引入。

## 开放问题

- train/val（可选 locked test）的分配单位：run 级（场景泛化）还是窗口级（插值自洽）。


## 当前基线（2026-10-01 起）：data-only CfC

**物理损失暂停**（见 ADR-0017 的决策块）。数据侧保留既定修正：**残差解码 + 动力学配平 loss +
一/二阶导数特征**。

```
./.torch-cuda-env/bin/python tests/train_cfc_interpolation.py \
  --manifest .scratch/n/manifest.json --cache-dir .scratch/n/frames \
  --held-out scan_1d22_ang15_5_massloss_neon_v400 \
  --epochs 20 --steps-per-epoch 11 --batch-size 32 --num-workers 0 \
  --lambda-h1 0.01 --lambda-axis 0.3 --lambda-phys 0.0 \
  --residual --loss-scale dynamics --input-deriv2 \
  --latent 64 --hidden 128 --dt-scale 50.0 --seed 0 --device cuda \
  --out output/cfc_interp/<name>
```

**基线指标**（held-out v400、15 窗口、α=0.5）：下表左列是 220 步的旧预算
（产物 `output/cfc_interp/gate6_deriv2`），右列是 **1200 步**的当前基线
（`--epochs 60 --steps-per-epoch 20 --save-best`，产物 `output/cfc_interp/long60x20`，
best = epoch 58）：

| 判据 | 220 步（20 epoch） | **1200 步（60 × 20，当前）** |
|---|---:|---:|
| 平均 **dR2**（增量口径，主轴） | +0.194 | **+0.416** |
| psi / Phi / Z_j / omega | +0.081 / −0.012 / +0.026 / +0.011 | **+0.275 / +0.281 / +0.120 / +0.222** |
| rho / T_i / rho_imp / T_e | +0.488 / +0.405 / +0.430 / +0.310 | **+0.701 / +0.718 / +0.718 / +0.691** |
| v_par | +0.011 | +0.016（唯一没动的场） |
| 场级 R² 范围 | 0.958 – 0.9998 | 0.973 – 0.9999 |
| psi H1_rel / H1_rel 最大 | 0.060 / 0.451 | **0.034** / 0.439 |
| 磁轴 中位 / p90 / max | 0.38 / 3.27 / 3.48 cm | 0.45 / 3.37 / 5.11 cm |
| 物理比（仅供记录，未参与训练） | 5.96 / 8.51 / 3.53 | —（未重测） |

**训练预算与选型（重要）**：220 步的读数**不是模型能力**，是"没训完"（pitfall #44）；
同一配方 1200 步几乎把所有场翻倍，并把 Phi/omega 从 ≈0 推到 0.22–0.28。
**必须按 Gate 指标（内层 val 的 `mean_dr2`）选 checkpoint**：同一实验的最后一轮
在 held-out 上 omega = −0.399。⇒ 配方固定加 `--save-best`。

**为什么用这套**：`--residual` 修掉"场级 R² 被 persistence 支配"（pitfall #29）、
`--loss-scale dynamics` 修掉逐场增量方差 5000× 失衡（#30）、`--input-deriv2` 给出
一阶导（FEM dof，corr ≥0.992）与二阶导（图 LSQ 拉普拉斯，与记录 `Z_j/ω` corr 0.987–0.997）。

**已知短板**：omega/v_par 的 dR2 仍≈0（输运主导）；psi 受解码精度底限制（#34）；
CfC 逐节点无空间耦合、单步跨整帧（#39）——这些是下一阶段网络侧的题目。

**子步消融（2026-10-01，否证 #39b）**：把"每帧一个 CfC 步"细分成 S 个子步**没有帮助**，
反而单调变差（held-out v400、20 epoch、同 seed）：

| 配置 | 平均 dR2 | 轴中位 [cm] |
|---|---:|---:|
| **S=2 / burn-in 1 步（基线）** | **+0.194** | **0.38** |
| S=2 / burn-in 2 步 | +0.176 | 0.72 |
| S=8 / burn-in 8 步 | +0.081 | 0.77 |
| S=25 / burn-in 25 步 | +0.027 | 1.06 |

根因（pitfall #43）：ncps CfC 的 cell **没有小步长极限**（`ts→0` 不趋于恒等），
所以"切细区间"不是数值加密，而是"多算几次循环更新"；同时 `--dt-scale 50` 让每步的时间门
趋近常数，dt 信息被抹掉。⇒ 子步保持 2（`--sub-steps 2 --burn-in-substeps 1`）。

**由此得到的结构结论（下一阶段第一优先）**：当前 CfC 的**时间语义是弱的**——同一个 S 内对
切分不敏感（随机 α 训练 = 时间自洽正则，ADR-0013 有效），但**跨 S 不可替换**
（一次 cell 走 `dt` vs 两次 `dt/2` 差别 ≈ 1.03 × ‖Δu‖）。
因此"任意 `t*` 上的插值 + 物理损失修正"这条主线要求先让**积分器时间一致**
（以 `hx` 为初值的闭式解 / ODE 传播子），否则 physics loss 修正的是一个没有时间语义的映射。
探针留在 `tests/probe_time_consistency.py`。

**训练器新工具**：`--save-best` 按**内层 val 的 `mean_dr2`**（与 Gate 同口径）保留
`checkpoint_best.pt`；`_evaluate` 现在同时报 `mean_dr2` 与逐场 `dR2`。

**时间一致性 vs 表达力（2026-10-01，`--cell ltc` 对照）**：当前 ncps CfC cell **不是时间
传播子**（`ts=0` 不是恒等；一次调用走完 `dt` 与两次 `dt/2` 的差别 ≈ 1.03×‖Δu‖），
而路线图要在插值点 `t*` 放物理损失、就要求积分器有时间语义。于是实现了
`--cell ltc`（闭式 LTC 传播子 `h(ts)=h0·e^{−rate·ts}+target·(1−e^{−rate·ts})`，
恒等缺陷 0、半群缺陷 <0.1%，由 `tests/probe_cell_semigroup.py` +
`tests/test_ltc_processor.py` 守）。**同一预算（1200 步）下的代价**：

| 平均 dR2 | psi | Phi | Z_j | omega | rho | T_i | rho_imp | T_e |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ncps **+0.416** | +0.275 | +0.281 | +0.120 | +0.222 | +0.701 | +0.718 | +0.718 | +0.691 |
| ltc **+0.210** | +0.077 | +0.030 | +0.045 | +0.010 | +0.479 | +0.396 | +0.508 | +0.315 |

⇒ **时间一致性目前要花掉一半精度**（pitfall #45）。两个混淆项未分离：
① ncps 处理器带 LSTM mixed-memory 捷径、LTC 没有；② LTC 每次只走 `rate·ts ≈ 10–20%`
（时间尺度/初始化可能卡住表达力）。

**下一步候选（未决，按性价比）**：
a) 给 LTC 加**可学习时间尺度 / cell 内 2–4 次解析积分**，看能否把平台从 0.21 抬到 0.35+；
b) 分离混淆项（保留 LSTM 捷径、只换 cell）定位那 2× 差距的来源；
c) ncps 与一致性路径**双轨**：数据拟合走 ncps、插值/物理修正走传播子（接口需对齐）。

## 网络（2026-10-01 起）：LTC 解析传播子（取代 CfC 插值网络）

**报告口径（2026-10-01 定稿，用户意见后修订）**：

| 指标 | 定义 | 看什么 |
|---|---|---|
| `loss` / `train_mse` | 训练目标里的**数据项**（z-score 空间、逐场按 `1/E[Δ²]` 配平） | 是否优化到位（欠拟合 vs 过拟合） |
| `val_mse` | 内层 val 的**普通** MSE（z 空间，未配平） | 训练侧泛化趋势；选 checkpoint 的旁证 |
| `R²` | held-out **场级** R²（物理单位，逐场） | **有没有复现下一帧**（注意 persistence 就能拿 0.98–0.999） |
| `MSE` | held-out **逐场物理 MSE**（= (1−R²)·Var，逐场量纲不同，只比同场） | 误差的绝对量级 |
| **`closer`** | `d_true/(d_true+d_prev)`，`d_true=‖u_pred−u_true‖`、`d_prev=‖u_pred−u_prev‖` | **主指标**：0=正中下一帧，0.5=正好在两帧中间（模糊），1=就是前一帧（照抄） |
| `skill_pers` | 旧称 dR2：`1 − ‖Δ_pred−Δ_true‖²/‖Δ_true‖²`（**不是 ΔR²**） | 仅作参考；细 dt 上分母进入噪声底，会失真（细 dt run 上 −40…−60） |

**`closer` 不需要按 dt 分层**（它用邻帧距离做参照，随 dt 同比缩放）：细 dt run 上它给出
0.47–0.64 的可读数字，而 `skill_pers` 在同一处是 −50。dt 分桶
（`dt<3 / 3–15 / 15–60 / ≥60`）保留作**诊断**与损失权重/物理采样的依据。
`tests/eval_phase3_structure.py` 输出全部列 + `MEAN` 行 + dt 分桶表（`--json` 落盘）；
`tests/report_run_table.py` 汇总训练侧与 held-out 侧。

**dt 是否要一致？——不要**（pitfalls #50）：连续时间传播子的解只依赖 `(h,x,dt)`，
非均匀采样是信息；要控制的是 ①损失权重（逐区间等权 vs 按 dt/分桶加权，**待定**）、
②度量按 dt 分层（诊断用）、③物理残差限定在可分辨的 dt 区间。dt 已显式进 encoder（`--input-dt`）。

**物理残差：本轮不做**（用户 2026-10-01 明确）。所有 run 都是 `--lambda-phys 0`、
不启用 `--predict-source`，插值点不负责任何损失项；物理模块仅作离线诊断。

**动机**：旧 ncps cell 不是时间传播子（`Φ(0)≠恒等`，切分一个区间会改变结果 10–41% 的更新量），
而路线图要在插值点 `t*` 放物理损失。已按用户要求改造为 LTC 微分方程的**精确解析解**。

**结构**：`src/models/ltc_cell.py::LTCCell`

- 网络输出不再是下一时刻隐状态，而是两个系数：`rate = softplus(head) + 1e-6`（>0，耗散）、
  `target = tanh(head)`（可切 `linear`）；状态用
  `h(ts) = h₀·e^{−rate·ts} + target·(1−e^{−rate·ts})` 推进；
- **LSTM/GRU 混合记忆已从默认路径移除**（`mixed_memory` 默认 False；旧 checkpoint 才开）；
- backbone 默认 **256×2**（补偿 LSTM 让出的容量）；`CfCInterpolator` 默认 `cell="ltc"`；
- **区间内输入固定、只链 h**（回灌解码 latent 会让系数依赖子步序号，破坏切分不变性）。

**两条设计约定**（决定精度与严格性的取舍）：

- `state_dependent=False`（默认）：`rate/target` 只看输入 ⇒ 半群性到浮点精度；
  `--state-dependent` 恢复 h 依赖（真 LTC gating），代价是 split 缺陷回到 `O(ts²)`。
- 区间切分（`--sub-steps`）对 latent **完全没有影响**（S=1/2/10 逐位相同）。

**验收（用户协议，未训练随机初始化）**：`tests/probe_ltc_semigroup.py`

| level | variant | ‖Φ(0)h−h‖ | ‖h₁−h₂‖（0→1 vs 0→0.3→1） | /‖update‖ |
|---|---|---:|---:|---:|
| cell | **ltc（默认）** | 0 | **3.3e-07**（float64：6.3e-16） | 3.7e-09 |
| cell | ltc（state-dependent） | 0 | 3.6 | 4.0e-02 |
| cell | ncps（旧） | 1.107 | — | — |
| model | **ltc** | 0 | **2.9e-07** | 5.0e-08 |
| model | ncps（旧） | 32.7 | 4.43 | 1.3e-01 |

**增广源积分器**（用户第 3 点，主路径暂不启用）：`predict_source=True` +
`advance_with_source_integral(..., num_substeps=5)` 在区间内高频解码 `S(τ)` 并梯形求积
`A=∫S dτ`（与记录的**区间平均**投影同语义）；常数源误差 0.0、可加性误差 0.0。

**精度代价（同预算 60×20=1200 步、同 seed、held-out v400、内层 val 选 best）**：

| 方案 | 平均 dR2 | psi | Phi | Z_j | omega | rho/T_i/rho_imp/T_e | v_par |
|---|---:|---:|---:|---:|---:|---:|---:|
| ncps（旧，非传播子） | **+0.416** | +0.275 | +0.281 | +0.120 | +0.222 | 0.70/0.72/0.72/0.69 | +0.016 |
| ltc v1（旧公式） | +0.210 | +0.077 | +0.030 | +0.045 | +0.010 | 0.48/0.40/0.51/0.32 | +0.026 |
| **ltc v2（本次改造）** | +0.147 | +0.133 | +0.046 | +0.087 | −0.191 | 0.24/0.37/0.31/0.34 | −0.007 |

⇒ **严格时间一致性当前要付 ~2.8× 精度**。两个补救选项也量完了（同预算）：

| 变体 | 平均 dR2 | psi | Phi | Z_j | omega | rho/T_i/rho_imp/T_e | v_par |
|---|---:|---:|---:|---:|---:|---:|---:|
| `--state-dependent` | +0.197 | +0.078 | +0.078 | +0.063 | −0.016 | 0.42/0.43/0.36/0.37 | −0.005 |
| `--target-activation linear` | +0.167 | +0.064 | +0.021 | +0.070 | −0.157 | 0.30/0.40/0.35/0.42 | +0.037 |
| 默认（input-only + tanh） | +0.147 | +0.133 | +0.046 | +0.087 | −0.191 | 0.24/0.37/0.31/0.34 | −0.007 |
| `--hidden 256`（加宽状态） | +0.156 | +0.169 | +0.033 | +0.087 | −0.118 | 0.26/0.36/0.32/0.29 | +0.006 |

四个 LTC 变体全部落在 0.15–0.20，**target 激活、h 依赖、状态宽度都补不上**；omega 是唯一
在所有 LTC 变体里都翻负的场（−0.016…−0.276 vs ncps +0.222）。**关键线索：LTC 家族始终
欠拟合**（末轮 `train_mse` 5.0–5.1e-4 vs ncps 3.66e-4），所以下一步先查**优化**
（lr / 训练预算 / rate-target 参数化），再考虑"数据拟合（ncps）+ 插值传播子（ltc）"双轨。

**驱动项接线（2026-10-01 核对）**：进网络的驱动项只有 5 通道 marker 投影
`[dE_ion, rad_sink, Z_eff, Z_imp, N_imp]`（`controls = source[:, k+1]`，**目标帧**，经
`control_encoder` 加到 latent 上）。**没有**接：背景三源 `[particle_source, heat_i_source,
heat_e_source]`（`projections*.h5` 只有 `n_var=5`，导出文件里根本不存在）、61 维参数通道
（只在物理残差里用）、碎片状态 `p(t)`（`with_shards=False`）。⇒ 当前网络是**插值器**：
它拿到目标帧的投影源，所以不是纯自回归预报器。

**损失接线（按用户 2026-10-01 要求，保持纯 data）**：插值点 `t*` **不承担任何损失**
（既不走进 physics residual，也不走增广积分器）；唯一的监督是**记录帧上的 data loss**
（+ λ_H1 结构项 + λ_axis）；`--lambda-phys 0`、`--predict-source` 不开。
完整结论与审查发现（checkpoint 架构错配、α 闸门取错状态、回灌 latent、
**目标帧教师源 ⇒ 网络是插值器而非预报器**）见 pitfalls #46。
