# 小规模原生源—场集成参考链（2026-10-08）

> **后续主路径修订（ADR-0022）**：本报告的冻结容量混合方案及历史结果保留。当前主路径改为双源/闭合条件下的九场联合学习，不再必须先解冻结容量；旧响应仅为显式互斥参考模式。新接口与机制证据见 [含源联合学习报告](native-joint-source-learning-2026-10-08.md)。

## 范围与结论

实际运行链为 **完整 cubic 旧场 → 持久 marker 推进/投影 → 两种独立源消费 → 完整原生学习场步 → OLD 场 shard 更新 → 统一提交 → 下一事件**。新增 `src/models/native_coupling.py`；事件仍由现有 `ReferenceRollout` 执行。12 项集成测试通过，相关回归 250 项通过；两组 phaseA 的有限片段经过元数据审计和真实局部载荷/Torch 反向验证。未训练、未读未来标签、未执行百万粒子回放。

验收仅为“小规模源演化—反馈消费—原生场推进—完整状态续跑的框架接通，覆盖范围有证据”。**尚未完成五通道全部物理消费者、完整 FEM 装配或 C0。** 旧 field-only 实验和产物保持原状；旧 `FieldPropagator.source_proj` 仍未物理对齐。

## 源码确认事实

只读源码 HEAD：`54c675bcc39c33b082c0c35046e28122dc53b3aa`。以下均为当前本地文件实际行号，普通 model502 装配分支；没有混用 FFT 装配。

| 文件与位置 | 确认事实 | 本轮对应实现 |
|---|---|---|
| `core/mod_jorek_timestepping.f90:327–328,392,418,424–429` | 启动时间来自粒子事件；流体求解后用 OLD 场更新 shard，再更新值、deltas、时间与前 dt | 复用 `reference_particle_then_fluid` / `reference_fluid_event` |
| `particles/examples/marker_spi_source.f90:311–355,426–442` | 粒子真实时间为 sim time 减 residual；实际 gather 为整数子步之和；先出生再推进所有存活粒子 | 复用 `advance_window` 的完整时钟/槽位规则 |
| 同文件 `496–529,535–601` | 粒子消费上一投影电荷闭合；电荷概率显式推进；矩由旧概率产生；辐射/电离能量分别计算 | full-native sample + 显式 atomic provider；本轮限定 Ne/non-CE |
| 同文件 `614–663,670–676` | OLD 单元/位置的完整 test basis 沉积，随后 AB 速度推进和定位；累计 RHS 除真实 gather | `NativeSourceMesh.deposit` 与一致投影，无额外 `2π` |
| `particles/mod_fields.f90:235–294,335–418` | 完整基函数场/导数；B 与流速的 R、t_norm 转换；Ne/Te 下限 | `NativeMarkerSource.advance` 的完整 cubic 点求值 |
| `particles/diagnostics/mod_uncoupled_projection.f90:146–155,258–294,499,527–530,563–590` | 共享 DOF index 汇总；投影 deltas=new−old；带 signed size 的 cubic/HZ、R J 积分 | H=1、无滤波、小网格完整一致质量矩阵；保留完整 projection deltas |
| `models/model502/mod_elt_matrix.f90:989–1047` | c2/3/4 截断、c5 下限0.5；商的空间导数与历史；delta_scale=dt/dt_prev；电荷上限与低温覆盖 | Torch 确定性消费者，原始 deltas 不改写 |
| 同文件 `1065–1085,1127–1132` | alpha_e 与电子密度；电荷平方矩进入 Zeff | 输出闭合量；alpha_e 进入本轮电子容量 |
| 同文件 `1163–1170,1290–1308,1384–1389` | 电荷闭合影响电阻率、电子离子交换、压力 | 仅压力公式/梯度输出；这些方程算子尚未执行 |
| 同文件 `1542,1717,1727–1728,1784–1789` | shard bg/imp 源进入质量方程；电子两项负号；流体 Gears 历史与投影历史是不同项 | 两项电子弱式载荷及独立密度源近似响应 |
| 同文件 `2767,2868,2941,3100–3103` | 电子时间块99与95/98联立；99含 `(rho+alpha_e_bis*rho_imp)*(1+zeta)` | 用户确认后采用冻结小网格一致99近似；省略95/98 |
| `models/pellet_module.f90:270,307–308,325` | 先按已有速度移动，半径用已有烧蚀率侵蚀，再由 OLD 场求新烧蚀率 | `SmallShardModel.update`，支持显式 model3；不支持 production model6 rocket |
| `models/mod_injection_source.f90:297–456`，尤其 `404–405` | `source_background`/`source_impurity` 独立提供；数密度转各物种质量密度 | `NativeDensitySource` 独立对象；不使用 marker c5 替代 |

文件 SHA256（复核本地源码内容，而非仅依赖 HEAD）：

```text
mod_elt_matrix.f90             5a47f3d314cf7251543d999e13109c5e781b31491eb94469fb826777432f814d
marker_spi_source.f90          4f6e832b107569d7a8a0155ce5995d04d59cc50771d256cdf99c1e692318991c
mod_jorek_timestepping.f90      f45b62fff5165f2c0f728c7a24d14b919628ff455ed4d1497daafaecceecb5fe
pellet_module.f90              6b9a9c78ccca3d504fcf2aae70792ea12d4b116738b474a36ffa7da3ef7c0e7a
mod_uncoupled_projection.f90   a79b3dc940d23de1c0cd10831d3bea58e84e2703e50d699cc8863f13e06ea25e
mod_fields.f90                 55f5b60aca9036d4c9c807f488e5c70e8a4887d3c356911333a20071a9fee232
mod_injection_source.f90       ebd1e97a2e384b5ed38d1ea15f13f77bee48bd2a377a1ad18f2cb0b5545e5eaf
```

### JOREK 怎么把电子载荷变成场更新？

JOREK 在九场联立线性系统内同时求场增量。电子时间容量不是逐节点除法：完整测试/试函数积分构成99块，且95/98包含密度变化的电子能量耦合；还有输运、传导等矩阵项。其原始 fluid deltas 进入独立的 Gears 历史项。这里只移植两项已核验电子载荷，无法称作该联立求解器。

用户答复允许在无法实现对应完整求解时先采用明确记录的近似。因此本轮采用：

- `A99=(1+zeta) Σ_e Bᵀ diag(w*J*R*(rho+alpha_e*rho_imp)) B`；容量在本步当前场冻结；非正容量明确拒绝。
- `ΔTe=solve(A99, L_direct+L_projection_history)`。
- 两个质量行使用一致单位容量，积分独立 bg/imp 质量率；`Δrho=Δbg+Δimp`，`Δrho_imp=Δimp`。
- 省略95/98、完整输运、闭合方程算子；**不另外实现物理 Gears fluid-history RHS**。raw fluid deltas 继续完整进入已存在学习核心，不能据此声称已执行那些物理历史项。
- 物理增量和学习原生增量在同一量纲下合成，再执行既有 native update mapping；没有把载荷直接加场，也没有任意5→9映射。

冻结容量、忽略密度联立块会产生能量交叉项误差，尤其强源密度快速变化时；本轮无精度或稳定性结论。该近似并未成为新的 ADR 决策。

## Interface、单位与状态

### 源推进

`NativeMarkerSource.advance(runtime,target_si,U_previous,U_current,births=...)` 返回新的 runtime 与诊断。场 `[9,4,1,N]`、反馈及原始 deltas `[5,4,1,N]`；采用完整 cubic 点求值、R/Z 导数、完整一致投影和共享 DOF 累加。几何绑定坐标全部系数、signed size、vertex、indices、单位和版本；最多128个唯一 DOF 的小型稠密参考，超过明确失败。

保留 x、element、st、weight、全部11电荷概率、V_prev、生命周期/槽位身份、投影值/原始历史、粒子真实时间、sim time、residual 和 gather。停止出生后原有活云继续推进。缺投影历史、错误单位、错误场轴、无法使用的端点或 startup 未执行均拒绝；没有伪造空云。由真实已知空云推进得到的零投影允许存在。

复用现有 charge/AB/生命周期实现，不走旧 node-dof0 SourceModule 的采样/投影降阶路径。`NativeNeonAtomicRates` 使用真实现有 ADAS SCD/ACD/PLT 表，表内容哈希入配置；cm³/s→m³/s，PLT×ne(cm⁻³)→W/ion。此 atomic Adapter 明确为 **PLT-only**，缺 PRB/CX，与生产辐射全路径不等价；不允许 surrogate fallback。解析测试另用显式已知 SCD provider，不能称作 ADAS 实测云。

### 两源与含源场步

`CoupledNativeField(field,response,source_configuration=...,shard_configuration=...)` 包装现有完整原生场步，保留 G/P、参数分组、rawD、官方 CfC、显式记忆和时间一致性外层。Torch `forward` 必须显式提供完整 marker feedback、projection deltas 和 `NativeDensitySource`。

| 对象 | 单位/时间因子所有权 |
|---|---|
| marker c1/c2 | 原生投影能量率；`t_norm/E_norm` 已在源沉积，消费者仅乘本步 dtJU 一次 |
| marker c3/c4/c5 | 电荷平方/电荷/数密度矩，按 n_norm；除 c5 的是消费层，非源端重复归一化 |
| 原始 projection deltas | 保存未经步长修正的新旧投影差；消费时仅乘 dt/dt_prev 一次 |
| 电子 direct/history load | 原生测试空间向量，已经时间积分/历史修正；容量求解不再乘 dt |
| 独立 shard bg/imp | `[2,4,1,N]`，`model502_mass_density_per_JU`；密度响应只积分一次 |
| 学习分支 | 时间适配后的原生本步增量；保持原时间尺度和恢复版本 |
| raw fluid deltas | 使用 update mapping 输出的原始增量，不能从舍入后的端点重建 |
| field CfC memory | 独立 `[N,hidden]`，不替代粒子物理状态或流体历史 |

`SmallShardModel` 的独立密度源是显式归一化轴对称 Gaussian，强度为 `ablation_atoms_s*t_norm/n_norm*(20/central_mass)/(2π)`；按物理环向体积预算构造，**不是生产 inj_source 核的逐点移植**。每个存活 shard 每个源窗产生一个中心 marker，weight=`ablation_atoms_s*gather`；没有恢复生产随机出生分布。小规模 model3 Parks/无 rocket/无 nonlocal；不把此路径假作两组 phaseA 的完整生产 shard 演化。

### 消费覆盖清单

| 消费者 | 当前级别 | 真正进入场响应？ |
|---|---|---|
| c1 电离能量 | 公式、局部载荷、冻结99近似响应 | 是，Te 行；可正负电离能量，原符号保留 |
| c2 辐射 | 公式、非负截断、局部载荷、冻结99近似响应 | 是，Te 行负源 |
| c4/c5 商历史 | 原始历史、比值修正、局部载荷、冻结99响应 | 是；不与 fluid deltas 合并 |
| c4/c5 电荷闭合 | alpha_e、ne、平均电荷 | alpha_e 用作本步电子容量系数；亦反馈下一轮源采样 |
| c3 电荷平方闭合 | 平均平方电荷/Zeff 公式输出 | **未进入本轮物理场响应**；非随意新增直接源 |
| 电荷对压力 | 压力及一阶梯度公式输出 | **未执行压力力/输运算子** |
| 电阻率、电子离子交换 | 消费依赖已定位 | **尚未实现** |
| 电子时间容量 | 使用当前闭合的99系数 | 近似响应；95/98省略 |
| shard 独立密度源 | full-native 一致投影/积分响应 | 是，rho/rho_imp；与 c5 独立 |

仅输出闭合量不等于执行消费者。本轮未将它们隐式拼进网络后声称完整物理耦合。

### 事件、恢复与失败

`advance_coupled_reference(state,model=...,source=...,shard_model=...)` 是现有事件调度的具体 Adapter：用旧 shard 计算独立密度率；正常事件在允许的上一/当前场端点内推进源；随后现有 fluid event 消费当前反馈、用 OLD 场更新 shard、统一提交所有字段。第二次源真实读取第一次预测的场，不读监督帧。没有新建平行调度器。

模型恢复沿 `save_model_rollout`/`load_model_rollout`，完整 field/rawD/memory、粒子/反馈/clock/shard、事件计划、配置均保留；新 coupled/source/shard/response 版本进入恢复一致性检查，field-only 权重不能静默恢复为 coupled 配置。注册 JOREK restart 单独清零投影/其历史并执行启动 fluid event，保留恢复粒子；模型恢复不触发它。

CPU NumPy 事件路径采用候选副本，任何源/场/shard 异常不污染原状态。model buffer/parameter 意外改变也拒绝并恢复；失败恢复 Python/NumPy/Torch CPU RNG。此小型事件 Adapter 显式拒绝 GPU model，避免承诺未覆盖的 CUDA 事务语义。已有 checkpoint 原子替换/损坏恢复回滚回归保留。

## 可微范围与学习近似

Torch `CoupledNativeField.forward` 的完整 U/rawD/G/P/参数/CfC/记忆、容量与显式源输入均可微；反馈给定条件下连续两步仅第二步 loss 可回传到第一步场/记忆和初态/核心参数。显式 detach 截断这些跨步梯度。

源电荷推进/定位/投影/shard 事件当前为 NumPy 参考，`NativeReferenceSolver` 仍为 no_grad 推理 Adapter；**没有宣称源—场端到端可微**。含源 Torch 场步与自生成源事件测试分别标记。共享 native stencil 到 degree 轴、Gauss→节点特征池化近似、物理 BC 未验收继续保留。H=1，无 n>0 环向学习验证。

## 实现与测试证据

按源推进→Torch 载荷→双源响应→完整事件逐片红绿推进；初始缺模块/类的红测试，以及几何不匹配/反馈事件时间不匹配未拒绝的真实红测试，随后最小实现通过。测试在用户已确认的三个公开 seam 运行，没有新审批等待。

`tests/test_native_coupling.py`：

- 独立中性→一级电离比例0.04、20eV能量及数密度积分预算；完整高阶投影，旧/新槽位分离，停止出生继续演化。
- 解析 `f=R+2Z+RZ`：值4.02、物理梯度(1.4,4.3)、R加权积分12；未用自实现生成预期系数值。
- ADAS 表格精确结点及 cm³/s 转换；拒绝 surrogate。
- 独立均匀场能量/商历史手算、Gears比值，c3只改变Zeff；三种输入 Torch 梯度。
- 零学习解码下 `ΔTe=-0.8`、`Δrho=0.06`、`Δrho_imp=0.04`；投影历史与 rawD 独立，source→field 确实响应。
- 移动 marker 的 R 位置改变空间反馈及 Te 响应；质量行不因移动 marker 改变；几何错配明确拒绝。
- 两次真实小云出生/电离/投影/场/烧蚀事件，无占位物理回调；第二事件检查实际场输入；新模型恢复与连续轨迹逐位一致。
- 三阶段异常回滚；field-only 配置恢复拒绝且模型不变；restart 与模型恢复规则不同。
- 含源 Torch 两步 BPTT及显式截断；完整场/历史/G/P/参数/CfC路径保持。
- 两组真实 phaseA 的局部完整 native 载荷与已有 NumPy 参考对照及反向；不是完整真实云闭环。

小型解析误差容差约1e-12（double稠密求解/求积舍入）；真实局部载荷 rtol3e-11、atol1e-18，延续已有独立 Fortran 基函数/载荷验证精度。保存恢复比较逐位相同。没有精度学习或长期稳定验收。

### 真实数据有限覆盖与资格

审计产物：`output/native-source-coupling-2026-10-08/phasea-audit.json`。

| run | 显式读取片段 | 注册restart | restart后普通步长 | 当前结论 |
|---|---|---|---|---|
| v400 | 4109–4112 | 4111 | 0.05 JU | 完整轴/时间审计、4111局部载荷与反向；完整云耦合未验收 |
| v400_injInner | 2702–2705 | 2704 | 5 JU | 完整轴/时间审计、2704局部载荷与反向；完整云耦合未验收 |

两组普通步长相差100倍；restart流体重复时间不代表粒子零 gather。所测片段普通流体输出逐步，不能泛化成跨多流体步。part内部时间按事件审计，不能按同编号流体时间强配。

真实局部历史只在两个已保存完整上一零投影的特例中计算 `current−past_zero`，没有推论任意邻帧能恢复历史。普通 HDF5 缺原始 projection deltas 时读入失败。已有真实 registered checkpoint 读取回归覆盖 v400 marker 快照，但本轮未推进其完整云：网格3986元素/4125节点超出小稠密参考，云规模也超小云预算；缺普通续跑 residual/反馈历史等不能补零。injInner 未因局部载荷通过而获得完整耦合资格。

后续真实耦合需显式匹配 shard/marker 的事件快照、前投影及原始 deltas、粒子 residual/dt、场前端点/rawD/前dt和真实 schedule；或实现已验证因果回放。还需稀疏投影/容量、生产出生核/原子辐射补齐、BC与物理消费者覆盖。不得使用未来 marker/反馈或百万粒子默认回放绕开缺口。

## 可复现命令与结果

```bash
.torch-cuda-env/bin/python -m pytest tests/test_native_coupling.py -q
# 12 passed, 2.84 s（包含因果输入、配置拒绝及反馈事件时间校验）
.torch-cuda-env/bin/python -m pytest tests/test_native_coupling.py tests/test_marker_state.py tests/test_fluid_history.py tests/test_native_state.py tests/test_marker_feedback.py tests/test_native_gauss.py tests/test_native_features.py tests/test_native_field_step.py tests/test_native_temporal.py tests/test_native_windows.py tests/test_native_curriculum.py tests/test_source_module.py tests/test_field_propagator.py -q
# 250 passed, 38.02 s；最后的反馈时间校验另由上列12项复验通过
PYTHONPATH=. .torch-cuda-env/bin/python tools/audit_native_coupling.py --output output/native-source-coupling-2026-10-08/phasea-audit.json
```

compileall 对新增/修改的 Python 文件通过；report/README.md 的 diff whitespace 检查通过。没有训练命令。所有运行属于小云机制、局部公式或结构验证；无 JOREK 相同工作单元计时，不宣称加速。未解决的消费者和生产源近似需要下一轮逐项验证，不能由本轮状态/协议通过推导物理完整性。
