# model502 marker 五通道确定性消费层（2026-10-07）

依据现行 ADR-0018/0019/0020，尤其 ADR-0020 决定 4。`roadmap_v2.md` 保持原文；保留已有用户修改，未训练，未修改只读 JOREK。

**公式消费层已验证**：已实现高斯点投影矩解释、商导数/增量、alpha_e、两种电子密度修正、有效电荷、电子压力值、电子能量直接项与投影历史项。**完整 FEM 装配待接入**；**九场求解尚未验证**。本报告的公式组件测试不能称为自生成源闭环验证。

## 生产分支与证据

源码根为 `/home/ac_desktop/JOREK_code/jorek`。本次重新核查文件和行号，没有套用 FFT 公式。

- `matrix/construct_matrix_mod.f90:63–82` 仅在完整环向块且 `n_tor>=n_tor_fft_thresh`，或 unified 模式下使用 FFT。
- `models/mod_settings.f90:5` 为 `n_tor=1`；`models/preset_parameters.f90:14` 为阈值 2；`models/model502/mod_model_settings.f90:16–18,55` 为 Ti/Te、杂质开启、unified 关闭。
- 两个 phaseA run 的 `jorek_namelist_spitest:528` 都开启 `use_marker`，没有 `n_tor_fft_thresh` 覆盖。用户已确认各自运行期间除步长表外配置不变。
- 本次实读两 run 的 `jorek004110.h5`：均为 model502、`n_tor=1`、`n_order=3`、`central_density=0.8`、`central_mass=1`、`t_norm=4.085775225951833e-7 s`。因此生产分支按普通 `element_matrix` 核查。此结论来自源码/配置与原始文件，不声称做了运行时装配插桩。

源码 SHA-256：

| 文件 | SHA-256 |
|---|---|
| `models/model502/mod_elt_matrix.f90` | `5a47f3d314cf7251543d999e13109c5e781b31491eb94469fb826777432f814d` |
| `models/corr_neg_include.f90` | `5be1a4cb976426438068aaddb4c4aec52eb2157ed775b6cd53f4514203270c73` |
| `models/mod_plasma_functions.f90` | `1940e414f78a66c98fc08758d42cb3a814a955afde66938a31b252ca474d73d9` |

## 输入、公式与消费位置

记 `c1..c5` 为**已在高斯点求值**的五通道，`D` 为已在该点求值的原始投影 deltas，`r=dt/dt_prev`。`n_norm=central_density*1e20 m^-3`，`E_norm=1/[μ0(γ−1)] J/m³`，`t_norm=sqrt(μ0*central_mass*amu*n_norm) s`。`μ0`、电子电荷与 amu 按本 JOREK `models/constants.f90:11,13,17`，不偷偷改用其他版本常数。

下表源码路径中 `M` 指 `models/model502/mod_elt_matrix.f90`，`P` 指 `particles/examples/marker_spi_source.f90`。

| 输入 | 公式 | 消费位置/输出 | 单位 | 源码行号 |
|---|---|---|---|---|
| 完整投影节点 values | `Σ values*size*H*HZ`，s/t 用 `H_s/H_t`，phi 用 `HZ_p` | `eq_aux_g/s/t/p`；本组件的外部输入边界 | c1/2 为归一化功率密度；c3/4/5 为数密度矩除 n_norm；s/t 为局部坐标导数，phi 为每弧度 | M294–298 |
| 完整投影节点 deltas | 同基函数求值，装配局部 `D_scaled=r*D` | `delta_aux_g/s/t`；持久原始数组保持不变 | 与对应 values 相同，非时间导数 | M301–303,321–326 |
| c1 | 不作非负截断 | `aux_dEion_dt`，直接电子能量源的一部分 | 功率密度乘 `t_norm/E_norm`，有符号 | P257–261,628；M989 |
| c2 | `max(c2,0)` | `aux_rad`，直接电子能量损失 | 同 c1 | P631；M990 |
| c3、c4、c5 | `N=max(c5,.5)`；`Q=max(c4,0)/N`；`Q2=max(c3,0)/N` | 平均电荷、杂质电荷平方均值；c3/4 本身不是均值 | N 为数密度/n_norm；Q、Q2 无量纲 | P535–537,632–634；M991–996 |
| c4/c5 的 s/t/phi 导数与几何 | `Q_a=(c4_a−Q*c5_a)/N`；`f_x=(y_t*f_s−y_s*f_t)/J`，`f_y=(−x_t*f_s+x_s*f_t)/J`，`J=x_s*y_t−x_t*y_s` | Q 的 s/t/phi/x/y 导数 | s/t 局部坐标；phi 每弧度；x/y 每米 | M353,999–1020 |
| D4、D5 | `delta_Q=(r*D4−Q*r*D5)/N`，使用触顶前的 Q | 投影历史线性修正，非有限差分商 `(M1_new/N_new−M1_old/N_old)` | 无量纲增量 | M1005–1007 |
| Q、ADAS n_Z | 若 `Q>n_Z`，Q 设 n_Z，五个空间导数设零 | 电荷上限；等于上限时仍保留导数 | 无量纲 | M1021–1028 |
| 修正后的电子温度 | `Te_eV=Te_corr/(e*μ0*n_norm)`；若 `<.1`，Q 设零 | 低温覆盖在电荷上限之后执行；未清空间导数或 delta_Q | eV | M937,1043–1047 |
| Q、中心质量和杂质质量 A | `m=central_mass/A*(2/2.01455279245)`；`alpha_e=m*Q−1`；`alpha_e_a=m*Q_a`；`delta_alpha_e=m*delta_Q` | 电子压力/密度与历史修正系数 | 无量纲，空间导数按对应坐标 | M883–907,1065–1079 |
| marker 分支的温度敏感性 | `dQ/dTe=d²Q/dTe²=0`，故 `alpha_e_bis=alpha_e`、`alpha_e_tri=0`；Z_eff 的 T/rho/rhoimp 线性化以及 Q2 的 T 线性化保留初始化零值 | 原生矩阵线性化输入；不能改用完整非线性公式的自动微分冒充原生线性化 | 按原生场变量导数 | M910–934,1032–1069,1087–1132 |
| 修正流体密度 rc、ric | `ne_raw=rc+alpha_e*ric`；JU 下若 `<2e-4`，`ne=1e-4+1e-4*exp((ne_raw−2e-4)/1e-4)`；SI 下独立 `max(ne_raw*n_norm,1e16)` | Z_eff 分母；辐射与碰撞密度 | JU 数密度/n_norm；SI m^-3 | M1081–1085；`corr_neg_include.f90:309–331` |
| Q2、N、rc、ric、ne_JU | `Zeff=max((Q2*N+max(rc−ric,0))/ne_JU,1)` | 混合物有效电荷；没有普通非 marker 分支的上限 | 无量纲 | M1127–1132 |
| 原始 rho、rho_imp、Te | `Pe=(rho+alpha_e*rho_imp)*Te` | 电子压力值；梯度消费者另需原始流体导数和 alpha 导数 | JOREK 压力单位 | M1384–1389 |
| c1/c2、当前流体 dt | `energy_rate=−(c1+max(c2,0))`；直接弱 RHS 为 `v*R*J*dt_JU*energy_rate*w` | 方程9直接能量项；SI 率为 `energy_rate*E_norm/t_norm` | 率：JU / 归一化时间；SI W/m³；乘 dt 后为原生时间行单位 | M1727–1728,1854 |
| delta_alpha_e、原始 Te/rho_imp | `history=−delta_alpha_e*Te*rho_imp`；弱 RHS 为 `v*R*J*history*w` | 方程9投影历史项，**不另乘 dt、theta 或 zeta** | 原生电子压力时间行修正 | M1787–1790,1854 |

重要的源码细节：非负截断和 `.5` 下限只作用于值；上表商导数与历史项继续使用原始投影导数/增量。电荷触顶清空间导数但不清已算 delta_Q；低温覆盖也不清这两类量。这里复现的是代码顺序，不是光滑截断函数的数学导数。FFT 路径下限 `1e-2` 及其他公式不纳入实现。`delta_aux_s/t` 虽被求值并缩放，但在此普通装配函数没有后续消费者；`delta_aux_p` 初始化为零而未参与上述求值。组件只要求实际消费的原始 Gauss deltas 值，不伪造不存在的增量导数用途。

投影历史与流体 `node%deltas` 是不同对象。后者仍走已补齐的 ReferenceRollout 历史链；方程9中 `zeta*(rho+rho_imp*alpha_e_bis)*delta_Te` 等项（M1784–1786）属于**流体**历史，尚须流体装配器处理。本组件不把它们与投影历史混在一起。

## 各方程的直接依赖与后续消费者

“直接依赖”包括当前装配公式中通过闭合系数的依赖；只有通道1/2是独立加性功率源。联立求解后其他场的改变不等于有 marker 直接源项。
通道1/2在当前装配中显式固定，没有相应的直接 LHS 源导数；不能把其他闭合项产生的 LHS 依赖归给它们。

| 方程 | marker 五通道在普通装配中的直接作用 | RHS / LHS 源码依据 |
|---|---|---|
| 1 ψ | Q/Q2/N → alpha_e/Zeff → 库仑对数、电阻率；部分矩阵磁压相关项使用总压力 | RHS M1474；LHS M1938–1939,1946–1970 |
| 2 u | Q/N → alpha_e 和空间导数 → Pe、总压力梯度 | RHS M1494；LHS M2101–2113 |
| 3 j | 无五通道直接引用；电流定义由 ψ 给出 | RHS M1530；对应 amat_31/33 定义块 |
| 4 ω | 无五通道直接引用；涡量定义由 u 给出 | RHS M1536；对应 amat_42/44 定义块 |
| 5 rho | 无五通道直接引用；独立 shard `source_bg/source_imp` 不能用 c5 替代 | RHS M1542–1571；对应方程5矩阵块 |
| 6 Ti | Q/Q2/N → ne、杂质 Q2 → 离子电子能量交换 dTi_e；非独立 marker 功率注入 | RHS M1632；LHS M2305,2341,2422–2424 |
| 7 Vpar | alpha_e/导数 → 总压力及梯度 | RHS M1638–1639；LHS M2576–2598 |
| 8 rho_imp | 无五通道直接引用；独立 shard impurity MASS 密度源 | RHS M1700–1720（源 M1717）；对应方程8矩阵块 |
| 9 Te | c1/2 直接功率、D4/5 历史；alpha_e/导数进入压力/平流/压缩/时间系数；Zeff/ne 进入欧姆加热、背景辐射、能量交换 | RHS M1727–1830；LHS M2764,2784,2822,2868–3027 |

边界之外的消费者必须显式接入，当前没有返回零值代替它们：

- `models/mod_plasma_functions.f90:62–127` 库仑对数需要原始/修正密度、原始/修正温度、中心温度、alpha_e 及修正导数；它从流体密度重算电子密度并独立使用 SI 下限。
- 同文件 `135–233` 电阻率需要 eta/eta_ohmic、温度阈值/中心温度、开关、lnA 及其导数、Zeff 和原生敏感性。M1160–1173 是实际调用；M1134–1158 的旧电阻率写法是注释，不作为实现依据。
- M1254–1275 的背景杂质辐射需要其他 ADAS 表及背景杂质密度；主杂质日冕 Lrad/dLrad_dT 在 marker 分支明确为零（M1195–1200），不可再加一遍 marker 辐射。
- M1282–1367 能量交换需要 Ti/Te 的修正值、修正导数、质量、碰撞常数等；Q、Q2、ne、alpha_e 是本组件给出的输入，完整交换与 Jacobian 尚未实现。
- M1384–1416 的压力梯度/高阶压力项与方程矩阵块仍需完整流体导数、几何和测试/试探函数。组件给出 Pe 值和 alpha_e 的一阶导数，不声称完成这些块。

## 可运行接口和约束

实现：[marker_feedback.py](../src/models/marker_feedback.py)；独立测试：[test_marker_feedback.py](../tests/test_marker_feedback.py)。

`consume_marker_feedback(GaussMarkerFeedback, GaussFluid, GaussMetric, MarkerFeedbackParameters, FluidTimeCoefficients, dt_si=...)` 返回 `ConsumedMarkerFeedback`。五通道及其 s/t/phi 导数、原始 deltas 都是 `[q,5]`，流体/几何为 `[q]`，允许将元素/极向高斯点/环向平面按一致顺序展平。`location='gauss'` 是调用者对求值的明确声明，不是将节点量转换为高斯量的功能。`delta_convention='raw_projection_increment'` 禁止把上次消费输出的 scaled 数组回灌为原始状态。

显式单位记录包含 n_norm、t_norm、E_norm，按参数一致性校验（相对容差 1e-9，仅容纳原生元数据舍入）；流体必须声明 `model502_native`。时间步用 SI 秒，直接能量项仅此处换为 `dt_si/t_norm`；变步长缩放仅使用 `FluidTimeCoefficients.delta_scale` 一次。支持 Euler/Crank-Nicholson/Gears 的系数，并检查 zeta 与 ratio 的一致性。所有求值不改写输入，持久原始 deltas 不被覆盖。
本组件使用 NumPy float64，属于独立可核查参考路径；未提供 Torch 自动微分或设备驻留后端。

缺失导数/deltas/单位、布局不匹配、非有限值、奇异几何、错误归一化、FFT、未知杂质或不匹配的核电荷、禁用 marker/TiTe/杂质的配置明确失败。支持本源码明确列出的 D2/C/Ne/Ar/Fe/W，质量比含同位素修正，不采用源码未知物种回退为 D2 的行为。原始流体量可为负；已经修正的密度/温度要求正数。组件没有内置未知的上游流体修正默认值。

`electron_energy_weak_terms` 只乘给定的测试函数、R、带符号 J 和高斯权重，返回直接能量与投影历史**两个分立贡献**；未做基函数/环向模求和、边界条件、全局打包或九场矩阵求解。`NativeMarkerLinearization` 中的零值是核查过的原生分支语义，绝不是用零值占位未实现的电阻率或交换项。

## 验证记录与容差

全部使用仓库 `.torch-cuda-env/bin/python`，无训练。小型 fixture 使用已知高斯点量；oracle 是手算标量商规则、非正交坐标变换与物理单位换算，不是把实现调用两次互相比较。例如 m=.1、M1=6、N=2，Q=3；s/t/phi 商导数为 1/1/1.25，J=5 下 x/y 为 .4/.2；ratio=2 时 delta_Q=.2、投影历史项为 −.16。

覆盖零/负值、N 下限、Q 上限及相等边界、低温顺序、非零所有一阶导数、非零 deltas、12 种时间方案/步长比组合、点对点批处理、两种电子密度修正、能量符号/SI功率/弱形式权重、原始数据不变和错误配置失败。公式 float64 断言 `rtol=2e-14, atol=2e-15`（少量代数与除法舍入），原始数组不变用逐位一致。

| 命令 | 结果 |
|---|---|
| `.torch-cuda-env/bin/python -m pytest tests/test_fluid_history.py tests/test_source_module.py tests/test_field_propagator.py -q`（改动前） | 64 passed / 9.32 s |
| `.torch-cuda-env/bin/python -m pytest tests/test_marker_feedback.py -q`（首轮） | 3 failed, 38 passed：两个手算 fixture 用近1密度相减产生消去误差；缺单位测试在辅助函数中提前访问 None。改用小正密度及修正测试辅助函数后 41 passed / .05 s；未放宽公式容差 |
| `.torch-cuda-env/bin/python -m pytest tests/test_marker_feedback.py tests/test_fluid_history.py tests/test_marker_state.py tests/test_source_module.py tests/test_field_propagator.py -q`（补批处理/配置验证后） | 122 passed / 9.00 s |
| `.torch-cuda-env/bin/python -m pytest tests/test_marker_feedback.py tests/test_fluid_history.py tests/test_marker_state.py tests/test_source_module.py tests/test_field_propagator.py tests/test_step_dataset.py tests/test_output_contract.py -q` | **183 passed, 1 skipped / 13.47 s**；跳过既有数据组件测试，公式测试全通过 |
| `.torch-cuda-env/bin/python -m pytest tests/test_marker_feedback.py -q`（最终公式测试集） | 49 passed / .05 s |
| `.torch-cuda-env/bin/python -m compileall -q src/models/marker_feedback.py src/models/source_contract.py src/models/field_propagator.py tests/test_marker_feedback.py`；`git diff --check` | exit0 |

本次原生 phaseA 元数据读取命令使用 `.torch-cuda-env/bin/python -` 配合 h5py，逐 run 读取上述六个字段，exit0；它验证生产分支选择的元数据，**不是**真实 FEM 高斯点评估的数值验证。

## FieldPropagator 的最小接口改动与结构缺口

下一步物理接入的最小接口应是：显式增加 `gauss_feedback`、`gauss_fluid`、`gauss_metric`、`feedback_parameters`、`time_coefficients` 及带单位的 dt，将 `ConsumedMarkerFeedback` 交给方程装配后端；该后端分别消费电子能量贡献、闭合系数/导数和投影历史。已有 ReferenceRollout 的 `solve_fluid(..., fluid_deltas=..., time_coefficients=...) -> FluidSolveResult` 可承接该后端，但必须继续独立传流体历史。

现有 `FieldPropagator` 仅接 node-dof0 `[nodes,9]`、`[nodes,5]`，先经 encoder/FiLM/LTC 再进按通道 stencil；`source_proj` 是自由学习 `5→hidden` forcing。**它尚未物理对齐**，本轮没有把此组件偷偷接成另一个加性 forcing。还缺：完整 Hermite degree/harmonic 的高斯点评估与导数、同位置几何/测试试探基函数、原生场的归一化输入（不能把 latent 或 symlog 值当物理量）、电阻率/碰撞等闭合后端、非对角九场块装配与完整时间行、约束/轴/边界及 nodal 增量映射。当前 bilinear/lumped 降阶 sampler 不能无损填补这些接口。应先接这些显式依赖，再评估九场求解；本交付仅完成上述公式消费层。
