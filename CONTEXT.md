# HyPhy-GLNN 领域词汇

本文件是 LNN-learning-for-JOREK 的单一领域词汇表：只定义概念，不记录实现或决策。
权威数据契约见 [report/model502-schema-and-source-protocol.md](report/model502-schema-and-source-protocol.md)；
物理约定见 [src/physics/definitions.md](src/physics/definitions.md)；决策见 [docs/adr/](docs/adr/)。

## 数据与状态

**Model 502**：九场 JOREK reduced-MHD 模型（`n_var=9`），本项目唯一数据来源。

**state contract（状态契约）**：每个图状态是 `float32 [nodes, 9]`，字段顺序为
`psi, Phi, Z_j, omega, rho, T_i, v_par, rho_imp, T_e`。

**原生量（native）vs 导出量（derived）**：原生量是 Model 502 的九个状态场；导出量是投影、粒子统计、
诊断序列与闭包。主状态模型只学原生量；导出量只作控制/诊断，或在辅助模块中作为目标（ADR-0011）。

**transform（变换）**：热力学场（rho/T_i/rho_imp/T_e）保号 symlog `sign(x)·log1p(|x|)` 后 Z-score，
其余场线性 Z-score；统计量只由训练 split 计算。反变换 `sign(y)·expm1(|y|)` 与 destandardize 串联，
是物理残差的入口（契约测试 `tests/test_normalization_contract.py`）。契约版本 `model502-state-v2`。

**单位制（2026-09-30 核实）**：全链路都在 **JOREK 内部单位**下，不转 SI：

| 量 | 单位 | 实测锚点 |
|---|---|---|
| 长度 / 网格坐标 `x` | **米** | `R_geo = 1.11`，网格 R ∈ [0.4626, 2.0200]，`R_axis_t = 1.2237`，`R_bnd_t = 0.8268` |
| 时间 `t_now` / `dt` | JOREK Alfvén 单位 | `t_norm = 4.0858e-7 s`（= `sqrt_mu0_rho0`）；帧间隔 `dt ≈ 10` ⇒ **4.1 μs**；记录跨 8423–10461 ⇒ 3.4–4.3 ms |
| `psi` | JOREK 归一化磁通 | `psi_axis_t = +0.5059`、`psi_bnd_t = 0.1142`、`psi_axis_init = −0.5034` |
| `rho`/`rho_imp` | 以 `central_density = 0.8` 为单位 | 峰值 93.6（≈1.2e20 m⁻³ 量级） |
| `eta` | Alfvén 归一化电阻率 | `1.443e-10`（H5 直接读） |
| `F0` | 真空环向场函数 | `−3.3087`（H5 直接读） |

磁轴误差因此可以按"米→cm"报（阈值 4/6/8 cm 有效）；物理残差模块用的就是这些单位
（系数 F0/eta 等全部从当前帧 H5 读，见 `_condition_from_frame`）。

**parameter channel（参数通道）**：影响下一步状态的零维参数向量（h5 → namelist → JOREK 默认值），
每项带 provenance。

**source-control contract（源项控制契约）**：主通道是 `projections*.h5` 的五场投影
`[dE_ion, rad_sink, Z_eff, Z_imp, N_imp]`；次通道是背景源 `[particle_source, heat_i_source, heat_e_source]`。
它们只是控制/源项，不是主状态模型的学习目标。

**marker（粒子标记）**：`part*.h5` 中的粒子记录（位置、速度、权重、寿命、电荷态）。
本项目 10 个 2D run 都有 neon marker。

**shard（碎片）**：SPI 弹丸碎裂后的球形碎片（JOREK `type_SPI`）；所有 10 个 2D run 都启用 SPI，
shard 状态只在 `t ≥ t_ns` 的帧存在。

**t_ns（注入时刻）**：碎片开始存在的时刻（当前数据 `t_ns = 8380`）；它把 run 分成无碎片段与有碎片段，
是窗口切分与 burn-in 的物理边界。

**fragment state p(t)（碎片状态）**：描述碎片集合的增广状态（位置、速度、半径、烧蚀率、species 等）；
由 shard 记录或 marker 归约得到。

**augmented state（增广状态）**：主物理场 `u(t)` 与碎片状态 `p(t)` 的并置；两者通过源项双向耦合。

**projection source field S（投影源场）**：五场投影在 JOREK 网格上的节点场，是碎片烧蚀/辐射在流体方程中的源项。

### 数据文件（一个"帧"包含什么）

**frame（帧）**：JOREK 的一次输出时刻，对应磁盘上同一编号的一组文件：
`jorek######.h5`（九场状态快照 + `spi_*_arr` 碎片状态 + `xtime_*` 时间序列）、
`part######.h5`（marker 快照：位置/速度/权重/电荷态/出生时间）、
`projections######.h5`（五场投影源，marker 投影事件写出）。
**aligned frame（对齐帧）**：上述文件都存在、编号一致的帧；数据集只使用对齐帧。
一个帧与下一帧相差 `nout = 50` 个流体步，帧内发生在 50 个流体步上的演化没有被逐帧记录。

**snapshot（快照）**：没有时间维、只反映"写出/调用那一刻"的文件。`nonlocal_array.dat`
就是快照（每次 `int3d_new` 调用都会覆盖），不属于任何特定帧；用时必须自己按帧重算。

**nonlocal table（非局域烧蚀表）**：`nonlocal_array.dat` 的内容——1000 行
`(index, psi_n, <T_e>, <n_e>)`，其中 `<·>` 是该磁通面上的体积平均（JOREK 单位）。
JOREK 的 `max(local, nonlocal)` 烧蚀分支用它把"磁面平均温度"映射到碎片处的烧蚀率；
逐帧重建见 `src/data/nonlocal_table.py`。

**postproc diagnostics（后处理诊断）**：`util/auto_spi_diagnostics.sh` 编排的官方诊断流程：
`util/convert2diagno_spi.py` + `jorek2_diagno_spi` 逐帧导出碎片轨迹 CSV（`fragments_position_*.csv`），
并在最后一个 restart 上导出 `abl_history.dat`（逐时间步的注入率/累计烧蚀原子数，来自
restart 里的 `xtime_spi_ablation*` 数组）。它不产出非局域表，也不改写 `part*.h5` 的 marker 数据。

**shard（碎片）**：`type_SPI` 碎片，状态存在 `jorek*.h5` 的 `spi_*_arr`
（R, Z, phi, Vel_R/Z/RxZ, radius, abl [atom/s], species, vol, psi, grad_psi…）。

**marker（粒子标记）**：`part*.h5` 中的杂质宏粒子（本例 1e6 个 neon），带权重与出生时间；
它是烧蚀云的空间分布采样，不是碎片本身。

## 时间与耦合

**staggered explicit coupling（错步显式耦合）**：JOREK 的推进顺序：由 `(u_k, p_k)` 组装源项 → 解出场更新 →
更新碎片；即 `(u_k, p_k) → S_k → u_{k+1}` 与 `(u_k, p_k) → p_{k+1}`。

**JOREK 的"隐式"到底隐式在哪（2026-10-03 源码核实）**：

1. **九场之间是隐式的**：每步 `construct_matrix` 装配九场耦合的大稀疏矩阵 + RHS，
   `solve_sparse_system` 一次解出九场增量（`core/mod_jorek_timestepping.f90`）。
2. **源项全部在 RHS 上、显式**：`models/model502/mod_elt_matrix.f90` 里
   `particle_source / source_pellet / source_bg / source_imp` 只出现在 `rhs_ij_*`
   （乘 `xjac·tstep`，部分项带 θ 权重），**不进 LHS 矩阵** ⇒ 源没有对场做隐式线性化。
3. **碎片不进矩阵，且在场解完之后用显式前向 Euler 单独推进**：
   `solve_sparse_system` →（`update_pellet` / `update_spi`）→ `update_values` → `t += dt`；
   `models/pellet_module.f90::update_spi` 里位置/速度/半径都用**整个流体步 `tstep`** 更新
   （`R += V_R·tstep/V_norm`、`radius -= …·tstep`、`V += Acc·tstep`），烧蚀率用刚解出的场
   代入解析定标律（`spi_abl_model`，含 `max(local, nonlocal)` 非局域分支）。无子循环、无隐式。
4. **投影链条是另一处隐式求解**：marker → `sample_rhs`（逐粒子逐子步累积
   `HH·size·f(particle)·weight`）→ 除以 `rhs_gather_time = n_steps·timesteps`
   （⇒ 记录的是**帧区间时间平均**）→ `project_only` 解投影矩阵 `M x = b`
   （`assemble_projection_matrix` + `solve_sparse_projection_system`）→ `save_to_h5`
   写 `projections*.h5`。所以 `projections*.h5` 的 `values` 是**解出来的 FEM 节点系数**，
   不是原始 RHS（这正是当年反组装 gate 差 1e3 的原因，ADR-0016）。
5. **`use_marker=.t.` 只切换电离/辐射闭合**（用 marker 电荷态而非 ADAS 日冕平衡，
   `mod_elt_matrix.f90` 的 `delta_alpha_e` 等分支），**不改场矩阵结构、也不改源沉积路径**：
   model502 的场 RHS 仍由 `total_imp_source → inj_source` 在 Gauss 点上按解析源形状逐点沉积。
6. ⇒ 我们数据里的 `projections*.h5`（marker 投影、帧平均、诊断/导出通道，也被下游 IMAS 读）
   与 model502 内部实际驱动场的解析沉积**不是同一个离散对象**：物理同源、离散/归一化不同。

**target-frame convention（目标帧约定）**：记录的投影 `S_k` 是驱动 `u_{k-1} → u_k` 的源；
训练时源项与目标帧配对，而不是与当前帧配对。

**NGS/Parks ablation（NGS 烧蚀标定）**：JOREK 内部的弹丸烧蚀标定律
（`abl ∝ n^-0.81 · T^1.64 · r^0.33 · V^0.44 · A^0.5`）与配套的碎片运动/半径侵蚀更新。

**source mapping sub-network（源项映射子网络）**：以碎片状态与局域背景场为输入、输出五场投影的小网络；
以解析 NGS 先验为基线、网络学残差。

**hybrid fragment dynamics（混合碎片动力学）**：场→碎片一步采用"JOREK 解析骨干 + 学习修正"，
而不是纯解析或纯学习。

**Eulerian moments（网格投影矩）**：把粒子（marker）的物理量按 FEM 基函数投影到网格节点上的
低维状态（粒子数、电荷矩、辐射/冷却权重、空间展宽）；作为图节点特征，不使用全粒子集合。

**augmented source accumulator（增广源项积分器）**：CfC 状态中专门累积源项的通道 `A(t)`，
`dA/dt = S_pred(t)`；损失用 `(A(t+dt)-A(t))/dt` 对齐 JOREK 的 gather 时间平均语义。

**curriculum teacher forcing（课程式教师强迫）**：先用真实粒子矩逐帧初始化学习单区间，
再自回归展开并用阈值式 Dagger 纠偏，最后长窗口纯预测。

**flux-coordinate attention bias（磁通坐标注意力偏置）**：cross-attention 中用磁通坐标
`rho` 的相对偏离（以及 `(dR,dZ)` 距离）作为偏置，表达沿磁力线远快于跨磁面的输运各向异性。

## 训练语义

**recorded frame（记录帧）**：JOREK 实际写出的场帧，是唯一有真值的时刻。

**data loss（数据损失）**：记录帧上的监督损失；记录帧训练保持纯数据驱动。

**interpolation point（插值点）**：记录帧之间随机采样的无标签时刻 `t*`。

**interpolation-time physics loss（插值点物理损失）**：只在插值点上评估的方程残差损失；
由完整源项保证残差在任意 `t` 都有定义（ADR-0010）。

**sub-interval integral residual（子区间积分残差）**：插值点物理残差的主形式——取 `δ = ρ·Δt`，
用 CfC 给出的 `u(t*)` 与 `u(t*+δ)` 组装 `M·Δu/δ − RHS`，不使用解析瞬时导数（ADR-0013）。

**CfC random two-substep integration（CfC 随机两段积分）**：把帧间隔按随机 `α` 切成 `α·Δt` 与
`(1−α)·Δt`，连续两次 CfC 单元调用；hidden state 连续，中间场不 detach。

**burn-in**：窗口开头只作输入、不算损失的帧段（当前约定 30 帧），用于把 hidden state 暖到窗口的物理状态。

**physics-consistent window（物理一致窗口）**：窗口内零维参数保持稳定；参数实质变化处切分窗口。

**teacher-forced source vs self-generated source（教师源 vs 自生成源）**：用记录投影驱动场，
还是用源项网络自己生成的投影驱动场；闭环训练必须使用后者。

**closed loop（闭环）**：场 → 碎片 → 源 → 场的自洽推演，不使用记录的碎片状态或投影。

## 验证与治理

**physics floor（物理残差本底）**：`pred = target` 时残差的不可约本底；验收用扰动单调性，
而不是"本底为零"。

**gate（验收门槛）**：进入下一阶段前必须通过的量化标准；未过 gate 冻结下游扩展。

**split manifest（划分清单）**：序列级 train/val/test 分配与数据版本/hash；禁止按帧随机划分。

**blocked split（分块划分）**：同一 run 内按连续时间块划分 train/val，块间留不小于窗口跨度的
embargo，防止重叠窗口泄漏；用于插值协议（ADR-0012）。

**Leave-One-Run-Out（LOO，留一 run）**：每次留出一个完整 run 做验证、其余 9 个训练，共 10 折；
用于场景协议的最终 gate，且不得用于调参（ADR-0012）。

**quench window（猝灭窗口）**：由数据定义的热猝灭区间，当前用热能 10%–90% 下降界定；
horizon 阶梯 H1–H4 相对 `t_ns` 与该窗口登记。

**provenance（来源溯源）**：任何报告数字必须能回溯到 checkpoint、config、随机种子与数据版本。

## 主要文档

- 路线图（单一来源）：[new_version.md](new_version.md)
- 报告索引：[report/README.md](report/README.md)
- 物理约定：[src/physics/definitions.md](src/physics/definitions.md)
- 决策记录：[docs/adr/](docs/adr/)
