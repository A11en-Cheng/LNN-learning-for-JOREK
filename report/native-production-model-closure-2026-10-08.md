# 真实 phaseA 模型闭环完善：辐射、n=0 投影与 base 续跑

以下为切片一的历史记录；本报告末尾追加切片二：n=0 稀疏投影和连续 base 事件。切片一**仅完成目标辐射 PLT+PRB 到 marker 投影的模型路径**。未训练、未调用优化器或搜索超参，未修改数据读取、manifest、资格、公共状态类型、JOREK、roadmap 或 ADR。数据 agent 的产物未被覆盖。源→含源九场联合学习→OLD 场 shard→统一提交主流程未变；冻结容量未成为必经步骤。

## 所检查源码事实

只读 JOREK HEAD `54c675bcc39c33b082c0c35046e28122dc53b3aa`。

| 源码 | 事实与实现对应 |
| --- | --- |
| `models/mod_impurity.f90:38–49` | Ne 选择 `96_ne`；两个目标 namelist 的 480–485 行均为 Ne，并声明生产 ADAS 目录。此处测试使用本机对应 `prb96_ne.dat`，不据此证明生产文件逐字节相同。 |
| `tools/mod_openadas.f90:87–94,112–130,240` | PLT 是非 recombining，行对应 q=0…9；PRB 是 recombining，行对应 q=1…10。系数由 W cm³ 转为 W m³；在 log 轴及 log 系数上插值后指数恢复。 |
| `particles/projection_functions/mod_radiation.f90:52–76` | `get_Lz=(PRB+PLT)*ne`，可选中性密度不进入这条公式。不增加未经确认的 CX 项。 |
| `particles/examples/marker_spi_source.f90:543–548` | 非平衡路径使用更新前 P_imp 加权辐射，T_eV>1 才计算辐射，然后更新电荷分布。保留已有源步阈值/顺序。 |

生产记录标识为另一 dirty RCS；结论限定为**对照所检查源码版本**，不声称已复现生产构建或认证两个生产锚点。

## 实现与单位

`NativeNeonAtomicRates(tables, radiation_mode='plt_prb', prb=...)` 显式消费完整 PRB ADF11 三元组 `(log10 n_cm^-3, log10 T_eV, log10 W_cm3 coefficients[10,T,n])`。缺 PRB、错误模式、不完整电荷行、非有限表、非单调轴失败。provider 深拷贝输入表，调用者后续修改不污染其状态。

输入为温度 eV 和电子密度 m^-3。SCD/ACD 仍以 m³/s 返回；辐射以 W/atom 返回：

- q=0：PLT[0] × ne_cm^-3。
- q=1…9：(PLT[q] + PRB[q−1]) × ne_cm^-3。
- q=10：PRB[9] × ne_cm^-3；完全剥离离子仍可辐射，不能沿用 PLT-only 的末态零值。

这里不乘 dt。既有 marker 事件按原始 P_imp、权重与粒子子步累积 J，随后用实际 gather 归一化和投影。未给已积分载荷再次乘 dt，未将辐射加到场系数上。

原 `plt_reference` 默认模式及其 `native-ne-ADAS-SCD-ACD-PLT-v1` 指纹保留，防止旧实验被静默改变。新模式为 `native-ne-ADAS-SCD-ACD-PLT-PRB-v2`，指纹包含原表指纹、全部 PRB 数值、电荷行映射及单位。源构造新增 provider 自带 version 与声明 atomic_version 必须一致的检查，防止把新辐射配置伪装成旧检查点的精确续跑配置。无 version 的已登记解析 fixture callable 保留现有接口。

使用方式（不涉及数据接口变更）：

```python
from src.models.native_coupling import NativeNeonAtomicRates
from src.models.source_module import load_neon_tables
from src.physics.adas_closure import parse_adf11, ADAS_DIR
rates = NativeNeonAtomicRates(load_neon_tables(), radiation_mode='plt_prb',
                             prb=parse_adf11(ADAS_DIR / 'prb96_ne.dat'))
# NativeMarkerSource(..., atomic=rates, atomic_version=rates.version, ...)
```

无自动生产配置升级；调用者必须显式选择新模式并采用匹配的源/模型配置。通用插值仍沿已有表边缘截断策略；未证明所有越界插值行为与生产构建一致。

## TDD 与独立证据

`tests/test_native_production_radiation.py` 最初两项因缺 radiation_mode 接口失败；实现后通过。随后新增旧 atomic_version 伪装测试，先因未拒绝失败，再补配置守卫。预期值来自常数表手算、log-affine 解析场和真实 PRB 表原始 knot，不调用实现插值生成预期。

- 全 11 电荷态独立数值 oracle，检查 PLT/PRB 行偏移、SI/cm 单位和末态非零。
- 完全剥离 marker、1e12 原子、ne=4e19 m^-3、PRB=20e-20 W cm³、1e-9 s：独立预期 0.008 J。实际源推进得到此值，辐射进入完整投影，raw projection deltas 为本次投影减旧投影；输入云和反馈不修改。
- 实读 `prb96_ne.dat` 最后电荷态 knot 对照；表复制、解析 log 双线性中点、缺项和配置指纹测试。
- 已有非零空间响应、完整 degree、含源梯度、跨步记忆、两事件预测场反馈、OLD 场 shard、恢复和阶段失败回滚回归保留。

命令与结果：

```bash
.torch-cuda-env/bin/python -m pytest -q tests/test_native_production_radiation.py tests/test_native_coupling.py tests/test_native_joint_source.py tests/test_native_sparse_source.py tests/test_native_context_guard.py tests/test_native_temporal.py tests/test_native_field_step.py tests/test_marker_state.py tests/test_fluid_history.py tests/test_native_state.py --junitxml=/tmp/native-production-radiation.xml
# 122 passed in 20.08s；新增 atomic_version 守卫之前。
.torch-cuda-env/bin/python -m pytest -q tests/test_native_production_radiation.py tests/test_native_coupling.py tests/test_native_joint_source.py --junitxml=/tmp/native-production-radiation-final.xml
# 最终配置守卫加入后：32 passed in 3.02s。
```

没有运行课程 optimizer 测试；没有百万粒子推进。首次真实表及机制测试为 0.88–2.25 秒级小测试；未单独测量 provider 峰值内存，不用测试总耗时宣称物理加速。

## 支持矩阵与剩余切片

| 项目 | 本次状态 |
| --- | --- |
| Ne 非平衡 PLT+PRB 辐射 | 已实现并独立验证，需显式选模式 |
| n=0 滤波、投影惩罚与 if/elseif 优先级 | 尚未实现；现有 sparse 仍是未滤波无约束参考，不改称生产投影 |
| 投影条件数/残差/系数误差的生产验证 | 尚待滤波/边界实现；本次未新增真实网格生产投影结果 |
| 九场物理边界/高阶更新约束 | 尚未实现；共享平均不能替代 BC，未开放生产资格 |
| rocket6、nonlocal OLD 场诊断 | 尚未实现；不认定 nonlocal 永久不可重建 |
| 生产随机出生及密度核 | 尚未实现；未引入随机出生或未保存 RNG |
| ns_radius_min | 两目标分别 0.05/0.01；本次未合并或硬编码配置 |
| 时间原点/尺度版本化、完整编码饱和诊断 | 尚未实现；先前绝对时钟单列风险仍有效，未删因果时钟或伪造合法历史 |
| 真实配置闭环认证 | 等待数据 agent 合格锚点及上述生产模型支持，不运行真值重置闭环 |

下一切片应在现有源推进 seam 接入 n=0 滤波和投影边界；矩阵依赖 OLD 磁场的平行项必须显式传入并纳入配置/状态校验。如需新增共享状态或数据输入，先与数据 agent 协调，不修改其文件。随后处理流体 BC、时间尺度及 rocket/nonlocal/出生核，逐个红→绿验证，不以占位回调声明支持。

结论仅为目标辐射分支机制补齐。无生产闭环、物理预测精度、长期稳定、JOREK 加速或 C0 完成结论。

## 切片二：目标 H=1 投影与连续 base 事件（2026-10-08）

本次优先实现投影，不代表其余源/shard/边界任务完成。保持联合九场学习主路径；不训练、不改数据 agent 文件，不修改公共事件/状态类型、schema、ADR 或 JOREK。已向数据 agent 确认 `advance_coupled_reference`/ReferenceRollout 可作外层连续序列 Adapter，指标由实际预测状态提取；其训练/指标文件由对方维护，不在本切片覆盖。

### 源码核查与接口

仍对照 HEAD `54c675bcc39c33b082c0c35046e28122dc53b3aa`。仅实现非 stellarator、非 zonal、H=1 完整 cubic n=0 路径，不混用 FFT 或非零 toroidal 模式。

| 位置 | 事实与对应处理 |
| --- | --- |
| `particles/examples/marker_spi_source.f90:146–150`；`particles/diagnostics/mod_project_particles.f90:188–196` | new_projection 创建时装配，使用初始化节点 values 中的 ψ；后续 project_only 没有每窗刷新磁场矩阵。显式传入初始化 ψ 全轴快照、SI 时刻与来源，冻结在该矩阵生命周期中。不是随窗口任意清零或偷读未来场。 |
| `particles/diagnostics/mod_uncoupled_projection.f90:38–48,875–894,938–985,1010–1018` | n=0 用专属 perp/hyper/parallel 系数；物理一阶、二阶导数包含曲边几何；parallel 在 H=1 下仍可非零。实现完整 cubic 二阶链式法则，无有限差分。 |
| 同文件 `:922,955,987` | literal `BB2=(F0²+psi_R²+psi_Z²)/R`；`Bgrad v=(v_R psi_Z-v_Z psi_R)/R`。严格保留这个 n=0 定义，未替换成另一模型常见的 `/R²`。 |
| 同文件 `:1074–1077` | 局部条目写入全局前，fix_axis_nodes 且轴上 degree3/4、全局行列索引相等时替换为绝对 1e12。共享索引可能导致不同局部行列条目也相等，不能只覆盖局部对角。 |
| 同文件 `:1098–1130` | 第一分支 boundary=2/3/5/9 惩罚 degree1/3；只有 elseif 的 1/3/4/9 惩罚 degree1/2。3/9 命中第一分支，不取集合并集。逐节点追加惩罚，共享 index 的重复贡献按源码累加。 |
| 同文件 `:136–155,258–280,1077`；marker 源 `:670–675` | marker RHS 为沉积/gather，无额外 TWOPI；矩阵体积分乘 TWOPI，边界惩罚仍绝对 1e12。raw projection delta 在写入前等于新解−旧投影，不能重置云或前投影。 |
| `models/preset_parameters.f90:662` | 所检查源码默认 fix_axis_nodes=false；两个目标 namelist 未覆盖该项。本次真实网格诊断显式传 false，仍以生产构建参数认证为后续要求。 |

模型构造新增 `projection_n0` **模型层字典**，必需字段：`filter_perp/filter_hyper/filter_parallel/F0/psi/field_time_si/field_provenance/boundary/axis_nodes/fix_axis_nodes/apply_dirichlet`。ψ 布局 `[4,1,N]`，沿用完整原生 JOREK 系数；几何仍 `[2,4,1,N]`、signed size 和共享 index。单位来自已验证 NativeUnits；滤波使用 namelist 的原生数值及源码度量，不另拟合尺度或默认参数。

未知字段/不支持模式、缺 ψ/来源/标志、非法边界码/轴掩码、负滤波、singular parallel metric 明确拒绝。支持目标中存在的 0/1/2/5/9/11/15/19/20/21 以及上述源码分支中的 3/4；11/15/19/20/21 在**该投影函数**无惩罚，不意味着九场流体边界也无约束。只允许 sparse 后端启用这条装配；dense 上限不变，formula-only 几何继续不需要全局矩阵。

矩阵为：

`A_ij = 2π Σ_q R J w [B_i B_j + a grad(B_i)·grad(B_j) + b Lap_cyl(B_i)Lap_cyl(B_j) + c Bgrad(B_i)Bgrad(B_j)/BB2] + penalties`。

其中 `Lap_cyl=∂RR+∂R/R+∂ZZ`。曲边二阶导数用 Hessian 链式法则扣除映射 Hessian，非独立预测量。通过 local 16×16 块和共享 index COO/CSC 装配，不创建全局稠密 basis；保持对角均衡稀疏 LU。

新 mesh 指纹 `native-source-mesh-v3_n0`、投影 `jorek_inspected_n0_filtered`，含全部滤波/F0、ψ、边界/轴快照、初始化时刻/来源。**完整构造输入同时进入模型检查点配置**：只存摘要不足以重建旧冻结矩阵；恢复从保存的构造输入重建因子，不拿恢复时的预测 ψ 替换初始化 ψ。旧 v2 无滤波参考及旧检查点未修改、未静默升级。新模式完整 2π 因子尤其不能被当作旧模式的精确续跑。

独立密度源不应通过 marker 滤波矩阵产生。`SmallShardModel` 现在拒绝复用带 projection_n0 的 mesh 作其密度投影；本切片小事件仍使用独立的既有 Gaussian 密度参考 mesh。它不是目标生产密度核；不能因 marker 投影升级就宣称双源全生产对齐。

### 独立测试及连续性

`tests/test_native_projection_n0.py`：先出现 3 个缺构造接口失败；最小实现后解析能量/边界测试通过。随后独立密度错用测试先失败，再加入口拒绝。恢复测试发现没有保存 ψ 构造输入（KeyError），补入后用检查点输入重建新因子并通过。测试 fixture 的 NumPy 初始化显式 detach，仅属于不可微参考事件测试，不改 Torch 训练路径。

- 解析矩形 `[2,3]×[0,1]`、试函数 R、ψ=Z、F0=−2：独立能量为 `2π[65/4+a×2.5+b×log(1.5)+c/5]`。验证质量、perp、cylindrical hyper、parallel 与 2π；数值积分误差受四点 Gauss 控制。
- signed size 曲边解析坐标试函数 R，以及可精确表示的 R²；后者 `grad_R=2R`、`Lap_cyl=4`，确实检验非零物理二阶项。能量使用独立解析几何与 Legendre 点计算，容差 2–3e−13 相对误差。
- boundary=3/9 的 elseif 优先级及共享 index 轴条目替换；不把高阶系数作正标量截断。
- 零滤波、无约束的新矩阵得到旧参考投影除 2π，直接通过完整源推进核对；使用非零前投影和前 rawDP，更新后 deltas 仍减旧投影，输入不污染。
- 小规模 **3 个 base 步=3 次源/场/shard调用**，调用顺序逐次为 source→field→shard；每次源读取上一步预测场，shard 比对 OLD 场。不是一个放大 dt 的事件，也不是冻结场的 marker 子步循环。
- 第一步后保存，在新模型、新稀疏因子下执行余下两步，与连续路径逐位一致：当前/前场、rawD、学习记忆、完整云所有数组、P/rawDP、clock、shard 位置/速度/半径/消融率。指标/保存不自动用真值重置场。
- 既有 source/field/shard 失败回滚、模型恢复/restart/zero-probe 区别和 Torch 跨步反传回归保留。新增投影为 NumPy 参考，不能称整条链端到端可微。

### 真实网格有界数值诊断

仅 v400 4110 与 inner 2703 两个有限快照。读取实际 HDF 几何、index、boundary、axis_node、F0/values 和各自 namelist 的 n=0 三滤波；两者均为 1e−4 / 1e−8 / 1e−2，未外推其它参数相同。这里**在指定快照创建诊断矩阵**，不证明它等于生产更早创建并持续保留的矩阵；认证仍需要生产初始化 ψ 或完整同构建矩阵来源。

| 量 | v400 | inner |
| --- | --- | --- |
| 唯一 DOF / CSC 非零 | 16419 / 578583 | 16419 / 578583 |
| LU 非零合计 | 8181565 | 8180577 |
| 构造/因子秒 | 2.467 | 2.674 |
| 一 RHS 求解秒 | 0.00501 | 0.00972 |
| 制造解最大系数绝对误差 | 1.522e−8 | 1.198e−8 |
| infinity 相对残差 | 2.606e−18 | 2.606e−18 |
| componentwise backward error | 4.857e−13 | 5.648e−13 |
| 原矩阵一范数条件估计 | 3.509e24 | 3.509e24 |
| 对角均衡后一范数条件估计 | 1.853e8 | 1.854e8 |
| 进程累计峰值 RSS KiB | 813996 | 946508 |

制造解是所有轴上 `sin(index*.123)*1e−3`，RHS 为矩阵作用于它，核验全轴求解精度；**它不是独立装配 oracle**，装配公式由前述解析能量/边界例子独立核验。条件数使用 t=1 的一范数估计器，为估计而非精确谱条件数。原系数量纲/几何尺度与 1e12 惩罚使条件很差，不能用极小全局残差掩盖约 1e−8 系数误差，所以同时报告 componentwise backward error。系数误差验收 1e−7，未声称机器精度恢复或物理预测准确。

内存是完整 pytest/Python/Torch 进程峰值，非稀疏矩阵增量或粒子云成本。构造上限仍 max_elements=5000；稠密 max_dofs=128 未提高。没有推进 11168/52604 活粒子的两个真实候选云，原 10000 活 marker 防线仍在，未裁剪或直接删除上限。百万槽位与活粒子数分开；真实云推进资源支持仍缺有界实测。

### 命令及实际结果

```bash
.torch-cuda-env/bin/python -m pytest -q -s tests/test_native_projection_n0.py -k two_real
# 1 passed, 9 deselected in 6.25s；输出上表 JSON 诊断，无优化器。
.torch-cuda-env/bin/python -m pytest -q tests/test_native_projection_n0.py tests/test_native_production_radiation.py tests/test_native_sparse_source.py tests/test_native_joint_source.py tests/test_native_coupling.py tests/test_native_context_guard.py tests/test_native_checkpoint_control.py tests/test_native_field_step.py tests/test_native_temporal.py tests/test_native_features.py tests/test_marker_state.py tests/test_fluid_history.py tests/test_native_state.py tests/test_native_gauss.py tests/test_marker_feedback.py tests/test_native_windows.py tests/test_native_conditional_windows.py --junitxml=/tmp/native-n0-base-final.xml
# 242 passed in 36.63s。
.torch-cuda-env/bin/python -m pytest -q tests/test_native_projection_n0.py -k 'not two_real'
# 最后补强非零前投影/rawDP样例后：9 passed, 1 deselected in 1.02s。
```

另一次先前相关回归 224 passed in 31.90s；没有运行包含 optimizer 更新的课程测试或训练入口，没有拟合真实尺度。

### 剩余支持域与协作

- 已完成：对照所检查版本的 H1 n=0 三滤波、投影惩罚和优先级、完整 2π、冻结初始化磁场生命周期、全轴 rawDP 顺序；小参考三 base 与恢复。
- 仍缺：目标 rocket6、nonlocal OLD 场诊断、生产出生/权重/RNG及密度核、九场流体 BC、时间编码原点/尺度版本化、两个真实云资源与闭环验证。现有 PLT+PRB 必须显式选择，不自动升级。
- 原始流体 rawD 与投影 rawDP 分开，当前 field 路径保留原生历史；源的端点采样与 JOREK `U_now−a×rawD` 在输出舍入/residual 分支上的完整对应仍需进一步核验，不能把端点时间差当作历史恢复。
- 数据/训练 agent 的统一指标接口需要实际九场/rawD、同场 Gauss 值/梯度/volume、五通道及自身事件/gather、独立密度源、按身份位置；已协调从完整事件输出读取，缺量报 missing，不补零。RZphi 与 Cartesian 位置转换须显式，不能把 slot 猜成永久身份。
- 给定源 Torch 场 BPTT 与 NumPy 参考事件分别声明。对方若实现 stop-gradient 源重算 Adapter，下一 solver 必须复用图上的 U/rawD/memory，不能从 detached driver 状态重建并称跨步反传；本切片未代写其实现或资格。

验收仅限**生产投影源码机制切片及连续 base 参考事件续跑**。尚未完成目标 phaseA 全生产源—场—shard闭环，不宣称训练资格、准确性、长期稳定、加速或 C0 完成。

## 切片三：保存历史的源采样与局部 rocket6 update（2026-10-09）

本节追加而不覆盖前两次验收记录。按依赖先修 A 时间机制，再交付 B 的局部纯 Ne update；**目标 nonlocal 配置、生产出生与独立密度核尚未贯通**。没有训练、optimizer 更新、超参搜索或 JOREK 重跑。JOREK 和原始数据只读，已有修改保留。

### 一手源码与实际分支

检查 HEAD 仍为 `54c675bcc39c33b082c0c35046e28122dc53b3aa`；只读状态显示 `communication/eqdsk2jorek.f90` 修改及未跟踪 `poincare`。生产 RCS 为 `v2.25.12.0-1042-gb4165893f-dirty`。当前检查不能解释生产 dirty 改动；HEAD 不同本身也不证明这些子系统不同。完整逐条证据及出生/nonlocal 依赖见 [一手研究](native-source-time-primary-evidence-2026-10-09.md)。research skill 要求的后台研究已完成，研究者仅写该证据文件。

| 源码/记录 | 本轮核查结论 |
| --- | --- |
| `particles/mod_fields_linear.f90:68–89`；`tools/mod_linear.f90:19–49` | 一阶分支在 t_jorek>0、abs(time_now−time_prev)>1e−10 s 且非static时取 U_now−rawD×(time_now−t)/interval；否则场值保持U_now。正t_jorek时仍计算rawD/interval或rawD/t_jorek时间导数；非正时导数0。二阶重载:164的跨度条件没有abs，不在本次实现范围。 |
| `particles/examples/marker_spi_source.f90:350–355,486` | 采样起点sim−residual，可早于field起点；源码没有clamp。出生初速度:435→`mod_initialise_particles.f90:537`用sim_time，不是residual起点。 |
| marker主程序:76；linear reader:405；`core/mod_jorek_timestepping.f90:568` | live reader i=-1初始化static=true；检查版本未找到live路径清false赋值，stepper只改时端点。static仅控制子步时间插值，不冻结base之间的场推进。 |
| 两个`logfile_spitest`，v400:1811、injInner:1781 | 均记录启动static=T。日志证明启动flag；生产dirty构建其后是否额外改变仍需相应源码/构建证据。不能默认生产使用非static插值。 |
| `models/pellet_module.f90:270–315,684–733,864–887` | 旧注入口速度移动→旧消融侵蚀→新位置OLD场诊断→新消融→rocket6新速度。rocket质量用max(r,1e−5)^3，面积仍实际r²，方向用移动前delta_phi。use_marker的局部Parks3用Z_imp=0占位；nonlocal诊断电子密度的电荷分支另算。 |

### 实现选择、Interface 与兼容性

`src/models/native_source_time.py::NativeSourceTimeSampler` 显式接收完整九场U/rawD、time_prev/time_now、已完成流体tstep秒数及static。只提供`fields_at`和`derivative_at`，复制历史，不从输出差重建，不增加时间clamp。固定几何上先组合完整系数再用既有FEM求值，场值和一阶兼容梯度来自同一场。

集成 agent 独占 `native_coupling.py` 与执行 Adapter：新模式`jorek_raw_deltas_v1`显式绑定sampler、current和marker时端点，必须显式选择field_static（不能默认false）；旧`endpoint_reference`保留为已有参考，不静默升级检查点。完整事件从state.fluid_deltas和已完成流体步长构造sampler，不误用下一control dt。模型/source配置保留原投影初始化ψ生命周期、2π、滤波/边界优先级和显式PLT+PRB绑定。本轮未改公共物理状态类型；NumPy源/shard停止梯度，Torch九场/记忆图不等于整链端到端可微。

`src/models/native_shard_dynamics.py::InspectedNeonShardDynamics` 仅是 **update Interface**，不是完整生产shard Adapter：

- 构造显式绑定材料原子密度、F0/R_geo、磁减弱开关、nonlocal开关及ns_radius_min/ratio。保留两run的0.05/0.01差异，不用同一默认值代替。
- 状态必需固定ids、RZphi位置、注入口固定R/Z/RxZ速度、injector_phi、纯Ne species=1、radius和旧消融率。位置/速度坐标语义不匹配即失败；不把角速度当米/秒。
- 已实现局部Parks3和rocket6；无drift、无toroidal rotation、无混合材料。radius侵蚀截零及质量下限来自源码；没有新增速度/加速度clamp。
- `nonlocal_abl=True`明确失败，列出完整空间诊断依赖。没有通过把该开关设false宣称目标run支持；目标两run实际都是true。
- 不提供生产births/density_source；未组合成完整事件的shard替代。ns_radius参数目前只保留绑定，出生/密度核尚不消费它们。该模块的局部测试不能给目标phaseA加合并资格。

### 近似与未支持项

nonlocal缺的是可复核输入/诊断实现，而不是原则不可恢复：完整OLD场与几何/体积权重、ψ轴/边/X-point及Z依赖规则、诊断坐标分支、n_flux=40/n_nonlocal_array=1000、构建WITH_TiTe/Impurities/电荷分支、表最近诊断时刻和缓存历史。`mod_integrals3D.f90:1740–1775,2530–2559`按磁通壳算∫neTe/∫ne、∫ne/∫1，随后重采样；`pellet_module:464`按最近ψ点及离轴方向选表。core:414只在表首值为0时初始化，不能默认每base重建表或强行把缓存表时刻等同当前场。

出生仍为旧SmallShardModel的一中心marker参考：生产需要旧位置/消融率、5倍max(radius×ratio,min)出生框、圆柱均匀体积proposal、源码拒绝概率、int数量及>1e14至少1规则、float32权重、sim_time初速度与边界/空槽生命周期。未实现这些，未新增生产RNG流；既有回滚/检查点测试不证明新生产随机机制。即便后续实现同分布，也需生产seed、MPI/线程拆流及消费位置才能宣称逐粒子一致。

独立密度核仍缺 `mod_injection_source.f90:360` 的完整体积/宽度/toroidal核实现。已有小Gaussian只属参考；禁止复用marker五通道滤波矩阵，既有入口拒绝守卫保留。source出生参考的端点插值/出生采样时刻尚未同步修成生产机制，不能把A的source.advance修复扩大为整个出生链已对齐。

### 实际运行证据、容差与成本

新增时间模块10项、shard update 5项：先红后绿建立机制入口，后补解析兼容梯度、移动前方向、实际小半径面积与新位置OLD场验证。预期来自手算/独立Decimal SI数值，不调用被测实现生成oracle。时间例U_now=10/rawD=4，中点8、residual越前取4；static/短时保持10。导数2e9的浮点容差rtol=2e−16；解析值/梯度绝对容差2–3e−15。

rocket例旧r=.003m、dt=1e−6s、旧abl=4.863185427757e27 atoms/s得到新r=.002m；ne=1e20、Te=2000eV使Parks各幂为1，新abl=8.081150161106476e23 atoms/s。pressure=1e5Pa、实际面积及20u材料质量独立算得acceleration=26259.335139022505m/s²，速度增量.026259335139022505m/s。消融相对容差3e−15；从1e5速度相减的增量容差3e−10，明确考虑相减舍入，不提高物理机制误差容忍。

真实有界验证仅v4004112/前4111、injInner2711/前2710的完整 `[9,4,1,4125]` U/rawD。8个static/非static情景采样每记录均有限；非static列是反事实分支核验，不是生产模式认证。前文件只供内部时间及差异审计，不用于构造rawD。相邻差−rawD最大绝对差分别2.274e−13/2.842e−14，仅这些记录近一致，不推广为历史恢复规则。

最终流式日志工具产物 `output/native-source-mechanisms-2026-10-09/time-sampling-streamed.json`，0.189秒，进程峰值RSS67904KiB。初版整读183/427MB日志曾4.15秒、1385536KiB，保留原产物不覆盖；已改为流读至启动证据即停，复现用下面最终工具。没有全云推进、没有裁剪11168活marker或百万槽，旧资源上限未删除。这不是真实闭环或性能预测。

```bash
PYTHONPATH=. .torch-cuda-env/bin/python tools/audit_native_source_time.py \
  --case data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400:4112 \
  --case data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner:2711 \
  --output /tmp/native-source-time-review.json
.torch-cuda-env/bin/python -m pytest -q tests/test_native_source_time.py tests/test_native_shard_dynamics.py tests/test_native_source_time_integration.py tests/test_native_production_radiation.py tests/test_native_projection_n0.py tests/test_native_coupling.py tests/test_native_joint_source.py tests/test_native_sparse_source.py tests/test_marker_state.py tests/test_fluid_history.py tests/test_native_checkpoint_control.py tests/test_base_coupled_training.py
# 最终 96 passed in 13.89s；JUnit: output/native-source-mechanisms-2026-10-09/regression-final.xml
```

该回归覆盖已有K次完整事件、预测场消费/OLD场shard、完整历史、失败回滚、新模型恢复；它们仍是小参考/fixture事件，不是完整真实云生产机制验证。未运行包含optimizer更新的课程测试。物理BC、时间编码、噪声/指标及真实云验证由集成agent独立负责，其后续结果不能由本节提前代称通过。

最终范围包含新增时间10项、shard5项及集成agent时间接口4项（含真实日志对应的static=True源消费和新模式必须显式static的守卫）。早先94项通过保留在regression.xml。补guard时一次并发进程已加载旧constructor，出现95通过/1失败；固定相关文件版本重跑后96项全部通过。`git diff --check`通过。

最终结论仅为A源采样修复及B局部rocket6 update可验证切片。生产nonlocal、出生/权重/RNG、独立密度核及dirty构建证明仍待补齐；不宣称真实闭环、训练资格或C0自动通过。
