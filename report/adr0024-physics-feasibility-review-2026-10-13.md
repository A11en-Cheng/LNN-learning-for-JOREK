# ADR-0024 草案：物理性、数学保证与实现可行性评估（2026-10-13）

这是源码证据与理论评估笔记，不是 ADR 决策，也不代表草案获接受。评估对象是 [ADR-0024 草案](../docs/proposals/adr_0024_hybrid_neural_analytic_ode.md)，SHA256 为 f9d87a4a34178c5ca67fe9c057e55d388aab7b37bdcb398dbcb4f4492ff1e438；引用行号按本次文件计算，未改草案。
总体判断：联合连续云、ADAS 主项与原生几何值得保留；局部冻结反应的保证不能推广为任意大步的全系统闭式、自洽和守恒。建议先修订核心主张，再作有边界的研究验证。
前五节给实现证据；B 节给优先级、数学推导与建议；C 节记录自包含算例；D 节列一手文献。没有正式采用新架构。
已先完整读取 `AGENTS.md` 与 `docs/adr/README.md`；现行覆盖关系见 [ADR 索引:21](/home/ac_desktop/LNN-learning-for-JOREK/docs/adr/README.md:21)。
本轮只读源码并新增本文件；未运行仓库回归、JOREK、训练、benchmark 或安装，未修改草案、代码、配置、roadmap 或 ADR。主代理仅执行了 C 节的三个标准库标量解析核验。
证据按本机工作树实际内容核查，不把已有报告或 ADR 中“当时尚缺”的快照当作当前实现结论。
JOREK 源码根目录为 `/home/ac_desktop/JOREK_code/jorek`，HEAD 为 `54c675bcc`；下文 `J/` 表示该绝对根目录。
仓库 HEAD 为 `0611ce4`，已有大量用户工作树内容；下列行号指此次读取的真实文件。

## 1. 4125 节点、形函数 gather、定位与投影

草案第30行只写九场在4125节点求值，第69–71行写四顶点形函数采样，第74–76行写转置 scatter。
这些表述还不足以覆盖当前完整原生状态与投影接口；4125仅为草案指定的节点数，本轮不认证所有网格都有同一节点数，不能据此删去 degree 或模式轴。
已实现可复用：`NativeFieldState` 要求 `[9,4,H,N]` 的 values 和同形 raw_deltas，模式显式声明。
证据：[native_state.py:32](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_state.py:32)、[native_state.py:73](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_state.py:73)；第49–51行明确 dof0 是有损观察。
当前学习场步只支持 `[9,4,1,N]`，每节点联合编码/解码36系数；不能把接口支持 H 轴写成已经支持任意 H。
证据：[native_field_step.py:72](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_field_step.py:72)、[native_field_step.py:169](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_field_step.py:169)；现行规定见 `docs/adr/0021-native-state-cfc-window-curriculum.md:22`。
共享索引并非每节点 degree 独立编号：`NativeSourceMesh` 对原生 `indices[4,N]` 去重，在唯一 DOF 空间装配。
证据：[native_coupling.py:20](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_coupling.py:20)；第44行生成每单元16项 local_indices。
`NativeUpdateMapping` 按共享索引平均预测增量再广播；这是现有学习映射选择，不能称作重建 JOREK RHS。
证据：[native_field_step.py:54](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_field_step.py:54)；第33–44行还明确限制悬挂节点、轴变换及 JECCD 分支。
四顶点拓扑可以保留，但每顶点必须读取四个 degree、所有声明模式；H=1时每场是16个局部系数，非四个标量值。
JOREK 直接证据：[interp_PRZ.f90:31](/home/ac_desktop/JOREK_code/jorek/elements/interp_PRZ.f90:31)，循环 vertex、degree，乘 size，并在第55–69行叠加环向模式。
可复用的完整基函数/几何/同场梯度见 [native_gauss.py:105](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_gauss.py:105)；第127–137行包含 size、HZ、曲面 Jacobian 与 R/Z 导数变换。
点求值见 [native_coupling.py:140](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_coupling.py:140)；所谓 attention 必须明确为完整基函数收缩，不能额外 softmax 改写权重。
给定 cell/st 后采样支持固定大小局部访问；找到该 cell/st 仍是另一个工作步骤，草案第71行的总 lookup O(1) 未由此建立。
本地 locator 仍从 RZ 位置筛选单元并用阻尼 Newton 反解 st，见 [native_coupling.py:254](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_coupling.py:254)。
JOREK 移动后调用 `find_RZ_nearby`：`J/particles/examples/marker_spi_source.f90:657–661`；它以旧 cell/st 起步并迭代几何反解。
直接证据：[mod_find_rz_nearby.f90:92](/home/ac_desktop/JOREK_code/jorek/grids/grid_utils/mod_find_rz_nearby.f90:92)；全局单元搜索分支见 `J/grids/grid_utils/find_RZ.f90:101–205`。
投影的 transpose scatter 只生成载荷，现有 `project` 还解矩阵并还原共享系数，见 [native_coupling.py:391](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_coupling.py:391) 与第409–416行。
JOREK 按节点 index 装配/MPI 合并载荷并求解，再写 values 与相对前投影的 deltas：`J/particles/diagnostics/mod_uncoupled_projection.f90:134–155,202–225,258–294`。
其矩阵包含质量项、垂直/超扩散/平行滤波、几何积分权重及边界惩罚，见 [mod_uncoupled_projection.f90:1011](/home/ac_desktop/JOREK_code/jorek/particles/diagnostics/mod_uncoupled_projection.f90:1011) 与第1096–1128行。
可复用的 H=1 滤波/边界实现见 [native_projection_n0.py:55](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_projection_n0.py:55) 与第95–103行；它不是九场流体边界算子。
实际 builder 分别建立滤波 marker projector 和独立无滤波 density projector，见 [native_real_chain_builder.py:100](/home/ac_desktop/LNN-learning-for-JOREK/tools/native_real_chain_builder.py:100)。
需要明确：U/C 的完整系数布局、共享索引、几何/size/模式、定位策略，以及第75行输出究竟是弱载荷还是投影系数。
还须区分点粒子已带原子权重的 scatter 与连续体积分载荷；当前粒子沉积不能简单再乘一次体积 Jacobian，参见 `native_coupling.py:391–397,118–128`。

## 2. marker 原子步与五通道加独立 density

草案第49–54行提出 `KC+B` 与指数原子更新，并称消除原子物理的隐式 Newton–Krylov 迭代。
当前核查路径的 marker 原子步不是 Newton–Krylov：按 SCD/ACD 显式递推各荷态，再非负截断及归一化。
直接证据：[marker_spi_source.f90:543](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:543)，第555–581行更新 `P_tmp`，第598–602行归一化/提交；第98行设 `CE_marker=.false.`。
本仓库对应实现见 [marker_state.py:203](/home/ac_desktop/LNN-learning-for-JOREK/src/models/marker_state.py:203)，第209–215行显式通量交换、截断和归一化。
可复用 ADAS 表/单位/电荷态数据及源生成接口；指数更新将是新的数值核，不能叙述成替换当前原子 Newton–Krylov。
JOREK 流体侧另有装配及 sparse solve/GMRES 选择，见 `J/core/mod_jorek_timestepping.f90:380–393`；这不是上述 marker 原子递推。
五通道顺序为电离能变化、辐射损失、电荷平方矩、电荷矩、数密度矩，生产者见 `J/particles/examples/marker_spi_source.f90:535–548,598–603,629–642`。
3/4通道是投影矩，不能直接等同已经形成的最终 Zeff/平均荷态；消费端除以带下限的 c5，见 [mod_elt_matrix.f90:989](/home/ac_desktop/JOREK_code/jorek/models/model502/mod_elt_matrix.f90:989)。
1/2进入电子能量 RHS；3/4/5参与电荷闭合、电子密度/电阻率等系数，不是固定5→9加性源矩阵。
证据：`J/models/model502/mod_elt_matrix.f90:1128–1131,1160–1165,1728,1787–1788`；本地闭合消费见 `src/models/native_coupling.py:747–767`。
独立 `total_imp_source` 仍生成 bg/imp 密度源，见 `J/models/model502/mod_elt_matrix.f90:1214–1227,1542,1717`；c5不是该注入率，也不是原生 rho_imp。
已实现完整双源接口：marker `[5,4,1,N]`、projection rawD、density `[2,4,1,N]`；见 [native_coupling.py:900](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_coupling.py:900) 与第772–776行。
编码 schema 也分开保存两条源，见 [native_source_features.py:12](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_source_features.py:12) 与第155–181行。
需要明确：草案 C 如何生成五通道窗口反馈及独立质量密度源，如何处理辐射/电离能与电荷闭合；仅命名 charge-state density 尚未给出这些适配器。
现行契约依据：`docs/adr/0020-marker-source-contract.md:105–110,141–158` 和 `docs/adr/0022-source-conditioned-joint-native-advancement.md:15`。

## 3. 同步 ODE 与现行事件时序、出生、restart、rawD、memory、controls

草案第24–35行定义同步 Z，第44–46行冻结步初环境，第49–50行使用常量 B，第57–59行直接给 U 更新。
现行正常步是已有场历史采样→marker/gather/投影→流体候选→OLD场 shard 更新→统一提交；不是同时把新 U/P/C 喂入所有分量。
JOREK：`marker_spi_source.f90:482–497,674–676,749–753`；投影事件在 stepper 前，见该文件第175–180行。
流体候选求解后，`update_spi` 在 `update_values` 前执行，直接证据：[mod_jorek_timestepping.f90:393](/home/ac_desktop/JOREK_code/jorek/core/mod_jorek_timestepping.f90:393)，第418及424–426行。
已实现可复用事件 driver 见 [marker_state.py:732](/home/ac_desktop/LNN-learning-for-JOREK/src/models/marker_state.py:732) 与第787–804行；第771–775行明确 OLD场 shard 后提交 U/rawD/memory。
出生是关键离散差异：JOREK 以源率乘 `n_steps*timesteps` 计算本窗口新粒子数，在子步 loop 前统一初始化。
准确证据：[marker_spi_source.f90:423](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:423)，第433–437行调用 birth，第443行设 gather，第452–454行选粒子数组，第482行才开始子步 loop。
本地同样在 loop 前插入 births，见 [marker_state.py:301](/home/ac_desktop/LNN-learning-for-JOREK/src/models/marker_state.py:301) 的第307–316行；调用链 `native_coupling.py:1003–1011` 先 births 再 advance。
因此草案第49–50行常量 B 的区间内持续注入，不等价于当前窗口前统一出生；会改变新生云年龄、荷态及窗口反馈，必须明确选哪种时序。
gather 是实际粒子覆盖时长，五通道除以同一 `n_steps*particle_dt`；不是默认等于下一个流体 dt。
证据：`J/particles/examples/marker_spi_source.f90:350–355,443,674–676`；`src/models/marker_state.py:396–403` 保留余留时间与实际 gather。
`S_n` 驱动 `u_n→u_{n+1}`，不是 `S_{n+1}`；依据 `docs/adr/0019-recorded-output-timing-contract.md:25–34,46–53`，restart 解释以0020第161–172行覆盖。
已有场 rawD 与 projection rawD 是两种历史：源采样用前者，电荷闭合历史消费用后者，不能互换或从单一 U/C 快照猜测。
源时间采样可复用 [native_source_time.py:42](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_source_time.py:42)；JOREK 对应 `J/particles/mod_fields_linear.f90:69–89`，static 分支也显式存在。
流体/aux deltas 按 dt/dt_prev 缩放，Gears 的 zeta 也依赖步长比：`J/models/model502/mod_elt_matrix.f90:222–225,321–326`；本地 `marker_state.py:693–714`。
JOREK 程序重启恢复粒子，但投影节点清零、余留时间清零并执行启动事件；模型 checkpoint 恢复和连续 chunk 切分不应附加此重启。
证据：`J/particles/examples/marker_spi_source.f90:100–103,165,312–314`、`J/particles/diagnostics/mod_project_particles.f90:127–132`；可复用 [marker_state.py:166](/home/ac_desktop/LNN-learning-for-JOREK/src/models/marker_state.py:166)。
controls 是显式步长计划、起止时刻、origin 与 phase；[base_coupled.py:79](/home/ac_desktop/LNN-learning-for-JOREK/src/training/base_coupled.py:79) 校验，不能把训练窗口当物理时间步。
训练 driver 按每个 base control 运行，并保持 Torch 场图与 detached NumPy 源/shard；见 `src/training/base_coupled.py:124–136,192–228,237–254`。
当前 CfC 有显式 `[N,hidden]` memory；外层 memory gate 和场速率积分分开，见 [native_temporal.py:32](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_temporal.py:32) 与第49–75行；并非草案第58行已落地。
需要明确：保留旧事件 driver 仅替换原子/时间核，还是改成同步物理 ODE；后者连同出生、输出标签、restart 和 checkpoint 都需新契约。

## 4. “精确 state”何时必须包含历史：目标 A 与目标 B

草案第24–32行声称 `(U,P,C)` 完整且 Markov；这里 A 指复刻 JOREK 离散映射，B 指新的物理 ODE，与现有联合提案的目标 A/B 一致，不是 roadmap 的 Phase A/B。
A 必须保存下一事件实际消费的历史：完整 U/rawD、持久 marker 全荷态/位置/权重/V_prev、上一投影及 rawD、时钟/余留时间、上一流体 dt、格式与计划游标。
证据：[marker_state.py:52](/home/ac_desktop/LNN-learning-for-JOREK/src/models/marker_state.py:52)、第122–144及562–584行；`native_coupling.py:563–582` 明确采样上一投影。
V_prev 是数值历史而非可随意删掉的物理标签：JOREK 使用旧方向做 Adams–Bashforth 推进，见 `J/particles/pushers/mod_fieldline_euler.f90:67–77`。
rawD 存的是求解器实际更新历史；JOREK 更新值和保存 delta 分开赋值，见 `J/communication/update_values.f90:88–94`，不能用相邻舍入后 values 差替代。
若学习模型使用 memory，精确模型续跑还须恢复 memory；检查点绑定配置/controls/RNG，见 `src/training/base_coupled.py:323–354`。
若启用局部细分，还须保存独立前子步间隔及身份历史；这与原生固定步互斥，见 `docs/adr/0023-explicit-marker-local-subcycling.md:7–14`。
生产机制还消费缓存诊断与外部 ES；相关可复用接口见 `src/models/native_source_production.py:23–27,48–60`，不能未经声明从 Z 中删去。
B 不必仅因 A 使用多步格式就把 rawD/V_prev 永久定义成物理 state；是否足够由新方程、闭合与外部控制定义决定。
但若 B 继续调用这些旧历史消费者、保留 CfC memory 或事件缓存，其可执行/续跑状态仍需对应增广；换名称不会消除依赖。
当前持久云是 MarkerCloud，C 的网格电荷态系数尚非这条主链的直接替代接口；单靠网格投影没有源码依据宣称可逆恢复完整 marker 集合。
需要明确：追求 A 的离散保真还是 B 的新动力学近似、哪些历史是物理/数值/学习状态，以及两者初始化、输出比较和续跑如何隔离。

## 5. GNN 是否现行主路径

草案第16行把 GNN 成本列为当前动机；本轮确认现行主路径是 stencil/CfC 的完整原生九场联合学习，GNN并非待替换主干。
现行规定明确不用 GNN：`docs/adr/0018-architecture-first-and-physics-structured-build.md:20,44–45`；后续0021/0022保留该空间路线。
实际代码用 FEM 邻接 latent stencil、Richardson 迭代和36系数残差解码，见 `src/models/native_field_step.py:100–129,225–237`；该 latent P1 近似不等于原生 cubic 物理算子。
实际 builder 接入 CfCTimeCore 和默认 joint_learning，见 `tools/native_real_chain_builder.py:177–187`；默认/互斥参考模式见 `src/models/native_coupling.py:825–847`。
可复用这些空间、时间及双源条件接口；需要明确第16行若描述旧探索就标历史背景，不据此声称当前 GNN 性能瓶颈或本草案已有加速收益。

## B. 主代理总评：方向可保留，核心保证尚不成立

本节是主代理对草案数学与物理的评估；文献结论、直接源码证据、工程推导和研究建议分别标明。下面不是对草案的修改或批准。

**结论：适合作为修订后的研究原型，不宜按当前文字直接认定为统一闭式、任意大步自洽且全局守恒的求解核心。** 有依据的部分是完整连续荷态云、已知原子反应与学习闭合结合、确定 FEM 几何交换。需要重写的是将局部冻结子问题的性质推广到完整非线性耦合系统的主张。

| 优先级 | 草案位置 | 问题 | 修订方向 |
|---|---|---|---|
| P0 | §3.1，行 44–46 | 起点冻结温密/B，未控制源—冷却—荷态反馈误差 | 明确冻结范围、内部联合求值/误差控制与合法事件 |
| P0 | §3.2，行 53–54 | K 奇异；有出生时总原子数增长，不能称无条件总量守恒 | 用 φ 函数/增广系统；限定局部保正与源平衡 |
| P0 | §3.3，行 57–59 | 对角松弛不是完整 MHD 推进或稳定性证明 | 作为可比较 Adapter；补空间/约束与物理 rate 定义 |
| P0 | §4.2，行 74–76 | 转置载荷不等于原生投影，更不自动守全系统能量 | 规定体积内积、质量矩阵、滤波/边界和能量账 |
| P1 | §2，行 24–35 | Z=(U,P,C) 的 Markov 性与完整状态未被定义 | 区分连续物理状态和 JOREK 离散/学习续跑状态 |
| P1 | §3.2 / §2 | 未给出 C 空间通量、P 运动/耗尽与源龄时序 | 补独立动力学/事件；不以反应指数替代输运 |
| P1 | §4.1，行 69–71 | 全 cubic DOF、导数/度量和移动定位未完整说明 | 使用原生 gather；不留 softmax，不宣称定位 O(1) |
| P1 | §1，行 15–16 | splitting 全面失败、固定 ns/μs、GNN 当前成本无依据 | 限定工况，保留对照；按实际输入/实现核查 |

这里 P0 指正式采用此设计前应解决的物理/数学契约，不是现有代码新增了生产故障。

### B.1 原子指数更新：正确且有价值，但保证有条件

在一个物理求值点，令 c 为 11 荷态原子密度列向量；若固定 K 的非对角元素非负、列和为零，且固定 b≥0，则应写：

$$
c^{+}=e^{hK}c_0+h\varphi_1(hK)b,\qquad
\varphi_1(z)=\int_0^1 e^{(1-s)z}\,ds,\quad \varphi_1(0)=1.
$$

这里 h≥0，K 是 SI 每秒或明确转换后的时间单位；不能混用 SI 与 JU。K 因保存原子数存在零特征值，K^{-1} 在整个空间上不可定义。φ_1 是整函数；奇异 K 并不妨碍该更新。

可以使用增广系统：

$$
\exp\!\left[h
\begin{pmatrix}K&b\\0&0\end{pmatrix}\right]
\begin{pmatrix}c_0\\1\end{pmatrix}
=
\begin{pmatrix}c^{+}\\1\end{pmatrix}.
$$

增广指数与矩阵函数作用的算法背景见 [T2：Al-Mohy–Higham，§2](https://eprints.maths.manchester.ac.uk/1591/1/alhi11.pdf)。这不是要求每个点都显式形成密集指数矩阵。

以下保结构结论是有限生成矩阵的直接推导：选 α>0、α≥max_i(-K_ii)，则 Q=I+K/α 是非负、列随机矩阵；e^{hK} 是 Q 的幂按 Poisson 权重的组合。因此齐次反应在精确算术下保正、保存原子数，且在 l1 范数下非扩张。正出生的积分项也非负，但：

$$
\mathbf 1^Tc^{+}=\mathbf 1^Tc_0+h\,\mathbf 1^Tb.
$$

这应称为“反应保原子数、含出生时满足源平衡”。只有将 shard 质量损失及边界损失配对记账，才有更大系统的物质预算。不能宣称有 b 时云本身总量不变，也不能由此推出 MHD 稳定或总能量守恒。

这些性质适用于物理荷态值，不自动适用于有符号的 Hermite 系数；反应求值、空间表示和回投影需单独保结构。有限精度的 φ/expm 算法还须验证误差及非负/预算漂移。“Taylor”可以是可行算法的一部分，但要有缩放/阶数/误差控制，不是写一句 stable 就得到保证：两态纯电离的二阶 Taylor 在 ah=4 时可给中性分量 5、离子分量 −4。

指数积分对冻结线性部分精确、对非线性余项仍需近似；刚性误差阶与振荡问题另有条件。[T1：Hochbruck–Ostermann，§1/§2](https://na.math.kit.edu/download/papers/acta-final.pdf)

### B.2 最主要缺口：整宏步冻结环境，不等于联合自洽

实际依赖应至少写为：

    K = K(Te(U), ne(U,C,合法反馈历史), μ)
    B = B(P, U, 注入控制, 事件相位)
    U_rate = Fθ(U,C,P,H,几何,参数)
    C_rate = -div Jθ(U,C,必要历史) + Kc + B
    P_rate = 已知运动/消融/rocket 主项 + 明确的学习闭合

草案只在起点求 K、B、神经 target。区间中冷却改变 Te/ne，云改变电荷闭合/辐射，shard 移动和侵蚀改变出生位置/量；被冻结的 K0/B0 并未响应这些变化。把变量写在同一个 Z 中，不会自动更新依赖。

**自包含数学反例，非 phaseA 结果**：两荷态、总原子数 1。前半区间纯电离，后半纯复合，两个速率均为 a=1000 s^-1，h=0.02 s，初始全中性。两个阶段都各有非负、保数的精确指数解：

$$
n_{1,\mathrm{true}}=(1-e^{-ah/2})e^{-ah/2}
\approx4.5398\times10^{-5}.
$$

若整步冻结起点的纯电离矩阵，则：

$$
n_{1,\mathrm{frozen}}=1-e^{-ah}\approx0.999999998.
$$

两者都保正且保原子数，但荷态几乎相反。系数由给定温度路径改变已经足以反驳“局部稳定推出任意大步正确”；反馈决定温度时更需耦合误差控制。此例只作构造性证伪，不估算真实云误差。

不能以“原子反应快于流体 dt”证明冻结合法；数值 dt 不等于温度/电子密度的物理变化时长。需要实际 rates、温密轨迹、出生/移动尺度和容差。反应很快时，状态应跟随变化中的平衡流形，而不是起点的平衡值。

常数 B 还代表持续注入。当前 JOREK example 在子步循环前按 gather 生成物质包；若保持这一契约，更接近：

    c(0+) = c0 + Δc_birth
    随后推进反应/输运

它与 c_dot=Kc+B 的均匀出生具有不同年龄分布。前文已给源码锚点；必须选择目标 A 的离散时序，还是目标 B 的新连续源，不静默互换。

shard 耗尽、出界、注入开启、跨单元和诊断更新仍需处理。冻结 B 若跨过耗尽事件继续注入，会破坏质量账；消除原子显式稳定步限制，不等于消除所有内部更新/事件。

### B.3 流体指数门控：可做候选，不是已证明的场方程解

将草案 τ 作为“衰减率”时，其单位为 1/time；若想表示时间常数，应写 λ=1/τ。还应限制 λ≥0，并明确网络输出对应完整原生系数还是物理求值点。

$$
U^{+}=D\,U_0+(I-D)U^*,\qquad
D=\operatorname{diag}(e^{-h\lambda}).
$$

这个表达对固定 λ/U* 的仿射对角 ODE 精确。U* 若依赖邻域 U/C/P，整体已不是独立的线性松弛；节点间输运、MHD 波、驱动不稳定性和事件不因一个正门控消失。它可以通过条件 target 近似局部 rate，因此不能反过来宣称它永远无法拟合 MHD；但需要实验与结构，而不能直接称完整积分或物理稳态。

反例：λ=1、U*=2U0、h=1，则 U+/U0=2−e^-1≈1.6321。正衰减率不保证全闭环收缩；物理 MHD 本身也可能增长，稳定性应定义为正确有界误差/支持域，而不是强迫所有场衰减。

CfC 的正式名称是 Closed-form continuous-time neural networks。原论文 scalar theorem 有单元、输入及连接假设，实用模型也使用 sigmoid 等学习结构；草案这个 U 门控不自动继承原 CfC 定理。[T4：Hasani 等，Theorem 1、Eq.(3)–(4)](https://research-explorer.ista.ac.at/download/12147/12355/2022_NatureMachineIntelligence_Hasani.pdf)

**九场还有代数/弱形式约束。** 本地 model502 的第 3、4 方程分别是 current definition 与 vorticity definition，RHS 没有时间推进因子：[mod_elt_matrix.f90:1527](/home/ac_desktop/JOREK_code/jorek/models/model502/mod_elt_matrix.f90:1527)、[mod_elt_matrix.f90:1533](/home/ac_desktop/JOREK_code/jorek/models/model502/mod_elt_matrix.f90:1533)。所以全部九场不能被视为任意独立 ODE 坐标。

一个直接的离散算子推导：在固定网格/合法齐次边界下设 j=Lψ，初态和 target 都满足该约束。即使 ψ/j 使用同一个门控 D，独立门控后的约束残差仍为：

$$
j^{+}-L\psi^{+}=(DL-LD)(\psi_0-\psi^*).
$$

空间变化的 D 一般不与 L 对易。完整 DOF 与值/梯度兼容也不自动保存这个弱定义。需要选择可兼容的主变量推进/重建，或明确的约束检查/修正与误差预算；这不要求重写完整 JOREK 求解器，但不能省掉定义。

### B.4 几何转置与全局守恒之间还缺几个步骤

固定 cell/st 下的全 cubic gather 可以复用；不是四个标量顶点的标准 softmax attention。仍要覆盖 degree、element size、共享索引、Jacobian 导数变换和已声明环向模式。找 cell/st 的成本不属于这个已定位后的 O(1) basis 求值。

Galerkin 载荷与投影系数不同。对连续出生核 β(x)：

$$
b_i=\int_\Omega N_i(x)\,\beta(x)\,dV,\qquad
A_{\rm proj}s=b.
$$

对物理 Dirac 点源 a δ(x−xp)，载荷是 a N_i(xp)；不能没有测度推导就再次乘单元体积 Jacobian。物理体积、2πR、归一化和滤波按具体模式/通道定义。有限宽生产出生/密度核不自动等于点 scatter。

同一 shape 权重的 gather/scatter 可建立离散对偶配对，但全局预算取决于正确的内积、常数再现、投影/边界和时间积分。对能表示常数的系数向量 c_const，总量关系应按 c_const^T b 或 c_const^T M s 检查，不是直接把所有有符号 Hermite 系数相加。

云出生也不是辐射/电离能本身：消融给 B0，后续荷态与温密给能量反馈。不能凭 W_shape^T 同时保证粒子与总系统能量守恒。

### B.5 端点云状态不足以定义能量反馈

共享累计电离势 ε 后，物理反应能率示意为 ε^T Kc，不能从包含输运/出生的整个 C_dot 冒充电离功率。辐射率示意为 ℓ^Tc，ℓ_z=ne(PLT_z+PRB_z)。JOREK 消费的是对应窗口的积分/平均及历史，不能只取 C(h) 的瞬时反馈。

对固定 K/b/ℓ 的局部问题，可以进一步推导：

$$
\int_0^h c(t)\,dt
=h\varphi_1(hK)c_0+h^2\varphi_2(hK)b,
\qquad
\varphi_2(z)=\sum_{r=0}^{\infty}\frac{z^r}{(r+2)!}.
$$

从而辐射能为 ℓ^T 上述积分。该式是局部冻结模型的工程推导，非整个 coupled step 的闭式物理解；原生 marker 的显式子步/旧荷态取样还有离散误差。

能量账应区分电子/离子热、荷态势能、注入携能、辐射和开放边界等适用项。复合的势能释放与 PRB 辐射不能双算；受驱动系统不能无条件要求“总能量恒定”。学习的 U 更新也需与这份账相容，空间转置无法替它完成。

### B.6 splitting 的动机应重写为条件性判断

只有快慢时间尺度相差大，不足以推出 Lie/Strang 完全破坏物理。非对易、边界、刚性阶、源/环境变化和子问题解法决定误差；适当 splitting/IMEX/指数方法仍可构造有效方案。[T3：Blanes–Casas–Murua，§1.1/§6](https://arxiv.org/html/2401.01722)

草案自己的冻结 K/B 加独立 U 门控，仍有分离/滞后耦合近似，不能因名称改成 unified 就免除分析。改用少量内部阶段/预测校正是候选，但不自动得到二阶、刚性一致阶或保正；需要指定系数及验收。

### B.7 建议保留与重新定义的最小架构

建议称为“物理结构化联合推进：局部原子指数子步＋学习流体/输运生成器＋事件感知耦合”，不先把它写成统一闭式解。一个拟议连续目标的依赖示意是：

$$
\dot U=F_{\theta,U}(U,C,P,H,\mu),\quad
\partial_t C+\nabla\cdot J_\theta=K(U,C,\mu)C+B(P,U,\mu),
$$

配套 shard 运动/消融、学习记忆 H 与必要场约束。若要逼近现有 JOREK 离散流程，完整历史与源/OLD 时序仍由现行事件 Adapter 管理，不能直接拿错步标签验证新同步 ODE。

这里 Fθ 的九场记号必须限制在约束兼容的状态/推进上，或配套明确的代数约束处理；它不是将九个场重新定义为互不约束的自由 ODE。可研究主变量推进后重建，或九场联合预测后检查/修正相同弱定义，二者成本和近似分别登记。

优先做五件事，均为建议而非本轮实施：

1. 定义完整 U/P/C/H、控制、云/反馈历史、出生与约束；选择 A 的离散参照还是 B 的新连续时序。
2. 将原子更新规范为 φ 函数/增广系统，验证保正、源平衡、能量积分、零初值与极小 dt；比较数值实现而非只看公式。
3. 明确 C 的空间通量与生产出生核，并把 native gather、载荷、投影、滤波/边界分开。
4. 选择有明确耦合误差控制的内部交换点；以变化温密、移动/耗尽 source 和纯流体波/约束案例验证，允许与适当 splitting/IMEX 比较。
5. 在相近误差与固定物理时长下量完整 CPU/GPU forward/backward、内存和恢复成本；之后才另行批准训练/宏步。

小 11×11 系统/φ 作用与批处理有实现基础，但逐网格 dense exp、ADAS/几何 backward、C 通量及长 BPTT 成本均未测；不能把“避免原子 Newton”当当前性能收益，因为参考原子步本来不是 Newton–Krylov。

## C. 自包含算例与核验范围

本轮没有运行 JOREK、真实云、训练、优化器或性能 benchmark。只用标准库 math 对三个显式解析表达作 CPU 标量核验；没有创建/修改代码文件，数字不来自 phaseA：

    python3 -c 'import math; a=1000.; h=.02; q=math.exp(-a*h/2); print(1-math.exp(-a*h),(1-q)*q); print(1+3*2); print(2-math.exp(-1))'

输出分别为：

- 0.9999999979388464 与 4.5397868608862414e-05：冻结热区 vs 热后冷两态荷态；
- 7：初始 1 原子、每秒注入 3、持续 2 秒后的原子数；
- 1.6321205588285577：状态依赖 target 下的正门控增长因子。

数学反例只否定草案的无条件保证，不证明所有混合神经—解析模型不可行。草案及原有项目文件保持不变。

## D. 一手参考文献

| 编号 | 来源 | 本次使用范围 |
|---|---|---|
| T1 | Hochbruck & Ostermann (2010), Exponential integrators, Acta Numerica 19, 209–286. [作者全文](https://na.math.kit.edu/download/papers/acta-final.pdf)，[DOI](https://doi.org/10.1017/S0962492910000048) | §1 的局部线性精确/非线性近似与 φ_1；§2 的刚性条件；不提供本项目大步保证 |
| T2 | Al-Mohy & Higham (2011), Computing the Action of the Matrix Exponential, with an Application to Exponential Integrators, SISC 33(2), 488–511. [机构全文](https://eprints.maths.manchester.ac.uk/1591/1/alhi11.pdf)，[DOI](https://doi.org/10.1137/100788860) | §2/Thm.2.1 的增广矩阵和 φ 作用；不证明 GPU 或全系统收益 |
| T3 | Blanes, Casas & Murua (2024), Splitting methods for differential equations. [作者全文](https://arxiv.org/html/2401.01722)，[DOI](https://doi.org/10.1017/S0962492923000077) | §1.1/§6 的阶条件、非对易、边界/降阶；非“全部 splitting 失败” |
| T4 | Hasani et al. (2022), Closed-form continuous-time neural networks, Nature Machine Intelligence 4, 992–1003. [机构全文](https://research-explorer.ista.ac.at/download/12147/12355/2022_NatureMachineIntelligence_Hasani.pdf)，[DOI](https://doi.org/10.1038/s42256-022-00556-7) | Theorem 1、实用门控结构；不迁移为原生 MHD 的闭式物理保证 |

相关既有项目证据：[联合方案](../docs/proposals/source-aware-joint-dynamics-2026-10-12.md)、[CE/marker 核查](jorek-coronal-equilibrium-vs-marker-evidence-2026-10-12.md)、[原生事件/接口审计](source-aware-joint-dynamics-code-audit-2026-10-12.md)。本文不修改这些文件，不把已登记近似升级为物理认证。
