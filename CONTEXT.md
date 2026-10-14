# HyPhy-GLNN 领域词汇

本文件是 LNN-learning-for-JOREK 的单一领域词汇表：只定义概念，不记录实现或决策。
权威数据契约见 [report/model502-schema-and-source-protocol.md](report/model502-schema-and-source-protocol.md)；
物理约定见 [src/physics/definitions.md](src/physics/definitions.md)；决策见 [docs/adr/](docs/adr/)。

> **2026-10-04 文档隔离**：本文件同时收录现行词汇与 v1 插值器路线词汇。标注 `[v1]` 的
> 条目/章节只作历史参考；现行契约以 [roadmap_v2.md](roadmap_v2.md)、
> ADR-0018/0019/0020/0021/0022（物理流程按0020，状态学习/参数/课程按0021定向修订）与
> [report/model502-schema-and-source-protocol.md](report/model502-schema-and-source-protocol.md) §3.1a 为准。

## 数据与状态

**Model 502**：九场 JOREK reduced-MHD 模型（`n_var=9`），本项目唯一数据来源。

**state contract（状态契约）**：既有节点视图为 `float32 [nodes, 9]`，字段顺序为
`psi, Phi, Z_j, omega, rho, T_i, v_par, rho_imp, T_e`；此视图不等于完整原生场状态。

**native field state（原生场状态）**：九场在全部原生节点自由度和保留环向谐波上的系数集合；
节点 dof-0 九场视图只是其有损观察，不是完整原生场状态。

**compatible field-gradient observation（兼容场值—梯度观察）**：由同一完整原生场系数、
基函数及几何共同定义的场值和一阶空间导数；不是独立可任意指定的两份物理状态。

**residual field advancement（残差场推进）**：保留当前完整物理场并叠加本步增量的推进表示；
它不等于已经求出完整JOREK解后的误差修正，也不同于marker的有界残差。

**source-conditioned joint field advancement（含源联合场推进）**：在当前完整物理场、独立流体历史与双源条件下联合预测九场状态变化的学习映射；它不要求先求一个近似物理解，也不等于已精确执行九场物理方程。

**field-only supervision（仅场监督）**：不消费显式源反馈的场轨迹学习任务；其标签仍可能来自受源驱动的物理过程，不能据此称数据为纯流体。

**pure-fluid task（纯流体任务）**：不含 SPI/shard/marker 演化反馈的流体状态推进任务；其它外加驱动与边界仍须明确，不等于将强源数据的源输入删除。
_Avoid_: 把 field-only、无反馈输入或无粒子文件直接等同于无源流体数据。

**verified exterior exit（已核验外边界退出）**：当前粒子轨迹首次穿过已记录网格外边界的事件，具有拓扑与几何依据；普通定位失败不构成外边界退出。

**terminal residence interval（末段驻留区间）**：粒子最后一次采样到确认出界之间、参与电荷演化及旧位置沉积的时间区间；它不同于整个粒子步长或整个投影窗口。

**frozen-capacity reference response（冻结容量参考响应）**：用当前电子容量将电子弱式载荷转换为近似场增量的参考响应；它不同于联合学习场推进，也不等于完整联立求解。

**learned field rate（学习场速率）**：单位物理时间的完整原生系数变化量；时间适配后得到本步增量。
它不是弱式载荷，也不是已积分的场增量；与潜特征及原始历史增量分别定义。

**field temporal memory（场时间学习记忆）**：场学习模型为时间依赖维护的潜状态；
它不替代原始流体历史增量、marker云或投影历史。

**causal training window（因果训练窗口）**：包含初始化依据、历史预热段和目标预测段的时间样本；
每步可用观测与未来监督标签明确分离，完整时间支撑包含其初始化回放。

**conditional-source window（给定源条件窗口）**：当前场和独立双源、投影历史及时钟合法配对的监督窗口；给定源的合法性不等于具有可恢复的 marker 云或自生成源闭环资格。

**recorded schedule replay（记录步长回放）**：离线从真实输出整理控制步长后执行的回放；它不是自主续跑时由外部提供的下一步控制。

**coupling base step（耦合 base 步）**：源、九场和 OLD 场碎片完成一次反馈交换的实际更新间隔；它不同于 marker 子步和监督记录间隔。

**marker transport history（marker 输运数值历史）**：按粒子身份保存的前实际推进间隔及局部细分状态；它不同于物理电荷历史、投影历史或场学习记忆，也不是原生粒子快照必然含有的信息。

**local marker subcycling（marker 局部细分）**：在不改变耦合 base 窗和 gather 的条件下，对选中粒子的输运、电荷及沉积共同细分的数值近似；不是只重复定位或只细分粒子运动。

**real-node supervision（真实节点监督）**：在真实输出时刻对因果推进所得状态计算监督的任务；真实右端不是区间内部推进的输入。
_Avoid_: 将其称作双端点条件插值或宏步加速。

**burn-in（历史预热）**：在固定预测起点之前，依合法初态与历史因果演化完整物理状态和学习记忆的阶段；它不替代缺失物理初态。

**coherent teacher intervention（一致教师替换）**：在事件边界以相互匹配的物理状态及历史替换当前状态的显式干预；仅计算监督不构成教师替换。

**native pellet density（原生弹丸材料数密度）**：`pellet_density` 的记录数值以 `10^20 m^-3` 为单位；转成 SI 数密度需乘 `10^20`，原始数值不能直接标为 `m^-3`。

**physical parameter routing（物理参数路由）**：按场景条件、物理系数和数值时间信息区分参数
对推进的作用；参数来源、单位和有效时间属于其含义，运行身份不是物理条件。

**weak load（弱式载荷）**：对测试函数积分得到的方程右端贡献，属于测试空间的对偶量；
它不是节点场值、场增量或学习潜特征。

**fluid history increment（流体历史增量）**：上次场更新留下、下一次时间离散所需的原始节点增量；
它与 marker 投影历史增量不同，不能保证由两个已舍入场端点相减无损恢复。

**committed prediction state（已提交预测状态）**：一次事件成功后相互一致的场、历史增量、
碎片、marker 云、反馈和时钟集合；部分更新的候选结果不是可续跑状态。

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

完整、逐条带 JOREK 源码行号的归一化规范（含输运系数单位与显式稳定性判据）见
[docs/jorek-normalization.md](docs/jorek-normalization.md)。

**parameter channel（参数通道）**：影响下一步状态的零维参数向量（h5 → namelist → JOREK 默认值），
每项带 provenance。

**source-control contract（源项控制契约；2026-10-07 按 ADR-0020 决定 1 修订）**：
marker run（`use_marker=.t.`）里源对象拆成两条——**解析 shard 密度源**
（`total_imp_source → inj_source`，Gauss 点沉积、进 RHS、对场显式；负责杂质密度）与
**marker feedback 五通道** `[dE_ion, rad_sink, Z_eff, Z_imp, N_imp]`
（`projections*.h5`；是电子能量源与电荷/杂质闭合输入组成的**真实反馈对象**，
`mod_elt_matrix.f90:989-1003,1079,1129,1728`）。次通道是背景源
`[particle_source, heat_i_source, heat_e_source]`（namelist 恒 0）。v1 的
"投影只是控制/不是学习目标"只适用于旧插值器路线（历史）。

**marker（粒子标记）**：`part*.h5` 中的粒子记录（位置、速度、权重、寿命、电荷态）。
本项目 10 个 2D run 都有 neon marker。

**shard（碎片）**：SPI 弹丸碎裂后的球形碎片（JOREK `type_SPI`）；所有 10 个 2D run 都启用 SPI，
shard 状态只在 `t ≥ t_ns` 的帧存在。

**t_ns（注入时刻）**：碎片开始存在的时刻（当前数据 `t_ns = 8380`）；它把 run 分成无碎片段与有碎片段，
是窗口切分与 burn-in 的物理边界。

**fragment state p(t)（碎片状态）**：描述碎片集合的增广状态（位置、速度、半径、烧蚀率、species 等）；
由 shard 记录或 marker 归约得到。

**persistent marker state（持久 marker 状态）**：跨粒子采样窗口延续的活粒子云状态，
包括空间位置、权重、电荷态分布及推进所需的上一速度；它不同于某一窗口的五通道投影。

**marker feedback state（marker 反馈状态）**：当前和上一窗口相联系的投影节点场；
它既是流体求解的反馈输入，也参与下一窗口 marker 输运和投影增量的计算。

**learned residual memory（学习残差记忆）**：降阶模型为修正无法显式解析的误差而维护的
潜状态；它不是 JOREK 的粒子云或反馈状态。_Avoid_: 将其单独称作“marker 物理状态”。

**augmented state（增广状态；ADR-0020 决定 3）**：联合描述主物理场、碎片、持久 marker
云及反馈状态的跨窗口状态；降阶模型可以压缩其中的粒子云，但须区别物理状态与学习残差记忆。

**restart initial state（重启初态）**：按 JOREK 的实际时序相匹配的流体、碎片和
marker 检查点状态；粒子与流体的记录时间可以错步，完整性不由文件同名保证。

**projection source field S（投影源场）**：五场投影在 JOREK 网格上的节点 dof-0
视图，是 marker run 的真实反馈对象（不是代理）；前两通道为电子能量源，后三通道为
电荷/杂质闭合矩，不能整体视作九场方程的加性源。共享 S 接口
（顺序/布局/单位/gather-time 平均）冻结在
[src/models/source_contract.py](src/models/source_contract.py)（ADR-0020 决定 4）。

### 数据文件（一个"帧"包含什么）

**frame（帧）**：JOREK 的一次输出时刻，对应磁盘上同一编号的一组文件：
`jorek######.h5`（九场状态快照 + `spi_*_arr` 碎片状态 + `xtime_*` 时间序列）、
`part######.h5`（marker 快照：位置/速度/权重/电荷态/出生时间）、
`projections######.h5`（五场投影源，marker 投影事件写出）。
**aligned frame（对齐帧）**：按内部时间、单位及事件位置完成对象配对的记录集合；
不由文件同号或全文件齐备定义，流体输出粒度与marker快照粒度可以不同。

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
2. **独立密度/功率源在 RHS 上、显式**：`models/model502/mod_elt_matrix.f90` 里
   `particle_source / source_pellet / source_bg / source_imp` 只出现在 `rhs_ij_*`
   （乘 `xjac·tstep`，部分项带 θ 权重），**不进 LHS 矩阵** ⇒ 这些独立源没有对场做隐式线性化；marker 电荷闭合另参与 RHS/LHS 系数，不能泛化为所有反馈仅在 RHS。
3. **碎片不进矩阵，且在场解完之后用显式前向 Euler 单独推进**：
   `solve_sparse_system` →（`update_pellet` / `update_spi`）→ `update_values` → `t += dt`；
   `models/pellet_module.f90::update_spi` 里位置/速度/半径都用**整个流体步 `tstep`** 更新
   （`R += V_R·tstep/V_norm`、`radius -= …·tstep`、`V += Acc·tstep`），烧蚀率用尚未提交更新的 OLD 场
   代入解析定标律（`spi_abl_model`，含 `max(local, nonlocal)` 非局域分支）。无子循环、无隐式。
4. **投影链条是另一处隐式求解**：marker → `sample_rhs`（逐粒子逐子步累积
   `HH·size·f(particle)·weight`）→ 除以 `rhs_gather_time = n_steps·timesteps`
   （⇒ 记录的是**帧区间时间平均**）→ `project_only` 解投影矩阵 `M x = b`
   （`assemble_projection_matrix` + `solve_sparse_projection_system`）→ `save_to_h5`
   写 `projections*.h5`。所以 `projections*.h5` 的 `values` 是**解出来的 FEM 节点系数**，
   不是原始 RHS（这正是当年反组装 gate 差 1e3 的原因，ADR-0016）。
5. **`use_marker=.t.` 切换电离/辐射/电荷闭合与场源**（2026-10-07 复核修正，
   ADR-0020 决定 1）：marker 模式下 `mod_elt_matrix.f90` 跳过解析 ADAS 分支
   （`:944,972`：`E_ion=0`、`Lrad=0`），直接从投影 aux 变量取
   `aux_dEion_dt/aux_rad/Z_eff/Z_imp/n_imp`（`:989-1003`）并把 `Z_imp/Z_eff`
   回代进电子密度/电阻率/输运系数（`:1079,1129`）；方程 9 的源是
   `-v·BigR·(aux_dEion_dt+aux_rad)`（`:1728`）。**解析 `total_imp_source → inj_source`
   只负责杂质密度源**（`:1219`，`rhs_ij_5 :1542`、`rhs_ij_8 :1717`）。
6. ⇒ 对 marker run，`projections*.h5`（marker 投影、gather-time 平均）就是
   能量/辐射/电荷通道的真实场源；它与解析 shard 沉积是**两个不同的离散对象**
   （支撑/标度/marker 云分布不同），两者都进同一套场方程、由 ADR-0020 的双源对象
   契约区分。

**recorded-output timing contract（记录输出时序契约，2026-10-04 源码 + 数据核实，ADR-0019）**：
`jorek{k}` = `u_k`（时刻 `T_k`）；`projections{k}` = 采样窗口 `[T_{k−1}, T_k]` 的时间平均、
内部时间 `T_k`，被第 k+1 步求解消耗 ⇒ **`S_k` 驱动 `u_k → u_{k+1}`**；`part{k}` =
时刻 `T_{k−1}` 的粒子状态（窗口起点 `p_{k−1}`）。训练与闭环配对固定为
`(u_k, p_{k−1}=part{k}, S_k) → u_{k+1}`。旧文档里的 "target-frame convention（目标帧约定）"
（`S_k` 驱动 `u_{k-1} → u_k`）经核实**不成立**，旧管线的 `source[:, k+1]` 属一步未来的源；
证据见 `report/marker-projection-causality.md` 与 ADR-0019。

**registered restart event（登记 restart 事件）**：code restart 造成的流体输出重复时间
边界；它不是零长度的粒子采样窗口。边界上的粒子可由检查点延续，而投影反馈被重新初始化；
因此跨此边界的状态与预算必须分段解释。

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

**curriculum teacher forcing（课程式教师强迫）**：从真实历史输入的局部监督逐步转向使用
模型状态的连续自回归训练；场教师替换与源教师替换是不同策略，不等同于专家重新标注的DAgger。

**flux-coordinate attention bias（磁通坐标注意力偏置）**：cross-attention 中用磁通坐标
`rho` 的相对偏离（以及 `(dR,dZ)` 距离）作为偏置，表达沿磁力线远快于跨磁面的输运各向异性。

## 训练语义（v1 插值器路线，历史）

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

## 验证与治理（v1 评估协议，历史；多 seed/LOO 仅作可选泛化评估）

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

- 路线图（单一来源）：[roadmap_v2.md](roadmap_v2.md)（v1 记录见 [new_version.md](new_version.md)）
- 报告索引：[report/README.md](report/README.md)
- 物理约定：[src/physics/definitions.md](src/physics/definitions.md)
- JOREK 归一化（代码核实）：[docs/jorek-normalization.md](docs/jorek-normalization.md)
- 决策记录：[docs/adr/](docs/adr/)
