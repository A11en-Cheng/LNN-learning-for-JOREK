# 含源九场联合学习主路径与机制验证（2026-10-08）

## 结论与架构变更

**源依赖已显式进入九场联合学习推进，小规模事件链和状态续跑保持正确。** 新默认模式 `joint_learning` 不计算或叠加冻结容量场增量；旧混合模式必须显式选 `frozen99_reference`，解析容量测试保留。本轮无训练，随机权重的响应和梯度只证明依赖接通，不证明已学会联立物理、准确预测、长期稳定或 C0。

决策：[ADR-0022](../docs/adr/0022-source-conditioned-joint-native-advancement.md)；定向补充0021，保留0020双源/marker/事件/restart契约。roadmap 的 B2、领域词汇、源 Interface 文档同步；旧实验报告仅加后续修订链接，没有改写原结果。

## 实际代码核查与新旧调用链

改动前直接读取 `native_field_step.py` 与 `native_coupling.py` 确认：学习场步的 encoder 只有完整U/rawD/五个流体时间特征；G/P和参数走已有特征分支。双源没有进入该学习分支。`CoupledNativeField.forward` 先调用学习场步，再由 `SmallNativeResponse.increment` 解冻结99与密度容量并叠加。c4/c5参与容量/历史响应；c3只生成Zeff而没有场响应消费者。这与“网络已含源联合推进”并不相符。

新路径：

```text
已有两端场 + 持久marker/shard
  → NativeMarkerSource.advance / SmallShardModel.density_source
  → 完整双源、原始projection deltas、源事件/gather信息
  → JointSourceEncoding：完整原生系数 + native Gauss值/梯度 + 确定性闭合
  → NativeFieldStep：U/rawD/流体时间 + G/P/分组参数 + 源编码
  → 残差stencil → 官方CfC → 显式记忆时间门控 → 九场学习速率
  → 现有时间适配积分一次 → native update mapping → U/rawD/记忆候选
  → 用OLD场更新shard → ReferenceRollout统一提交
```

没有新事件调度器；`advance_coupled_reference` 复用原事件路径。源投影仍需求解其独立 FEM 投影问题；这是源空间表示，不是冻结电子容量场响应。默认模式不调用 `SmallNativeResponse.increment`。formula-only消费者仍可计算已验证弱式载荷供参考，但默认网络没有消费或叠加这些载荷。

| 模式 | 场响应所有权 | 输入与恢复 |
|---|---|---|
| `joint_learning`（默认） | 全部九场响应由网络联合预测；完整U恒等支路 | 必须有formula-only消费者、完整G/P、显式源尺度、双源/历史及源时钟 |
| `frozen99_reference`（显式） | 旧学习增量 + 冻结99/质量容量近似 | 必须为SmallNativeResponse；拒绝joint源编码尺度/时间条件；保留旧物理测试 |

把SmallNativeResponse传给默认模式、未知mode、缺尺度、混用两个模式明确失败，不按对象类型静默选择。

## 源码事实（只读复核）

JOREK HEAD `54c675bcc39c33b082c0c35046e28122dc53b3aa`。本轮复核普通 model502 装配分支：

| 位置 | 事实及本轮含义 |
|---|---|
| `models/model502/mod_elt_matrix.f90:989–1028,1043–1047` | c2/3/4截断、c5下限0.5；电荷矩/平方矩除c5；商空间导数及dt/dt_prev历史；电荷上限和低温覆盖。复用现有Torch公式并将结果编码 |
| 同文件 `1065–1085,1127–1132` | alpha_e、电子密度和Zeff闭合；c3不是平均电荷平方的现成均值，也不是功率源 |
| 同文件 `1163–1170,1290–1308,1384–1389` | 电阻率、交换和压力依赖电荷闭合。本轮输入闭合特征，不移植完整消费者 |
| 同文件 `1542,1717,1727–1728,1784–1789` | 独立shard质量源、电子功率负号、独立流体/投影历史项；不把c5当密度注入 |
| 同文件 `2767,2868,2941,3100–3103` | JOREK电子容量95/98/99联立块；网络学习其整步映射不要求显式执行其中一个近似块 |
| `core/mod_jorek_timestepping.f90:392,418,424–429` | solve → OLD场update_spi → update_values → update_deltas → 时间/前dt；学习候选替换solve位置，顺序不变 |

SHA256：M=`5a47f3d314cf7251543d999e13109c5e781b31491eb94469fb826777432f814d`；timestepping=`f45b62fff5165f2c0f728c7a24d14b919628ff455ed4d1497daafaecceecb5fe`。其余源/投影/shard源码证据沿用[上一报告](native-source-field-coupling-2026-10-08.md)中记录的相同源码版本；未修改 JOREK。

电子功率直接作用于方程9；shard质量率直接作用于5/8；电荷闭合影响1/2/6/7/9的系数/右端。方程3/4/5/8没有marker五通道直接引用，但联立解可以间接受影响。因此本网络九场联合输出没有人为限制“电子源只影响Te”；这仍不是一个有物理含义的固定5→9直接RHS矩阵。

## Interface、逐层shape与信息损失

实现：`src/models/native_source_features.py`、`src/models/native_coupling.py`；NativeFieldStep只新增进入stencil之前的显式latent conditioning Interface，不改官方CfC或时间适配公式。

```python
model = CoupledNativeField(
    full_native_field_step, formula_only_TorchMarkerLoads,
    source_configuration=source.configuration,
    shard_configuration=shard_model.configuration,
    mode='joint_learning', source_scales=SourceFeatureScales.physical_reference())
result = model(U, rawD, dt_si=dt, previous_dt_si=prev_dt, scheme=scheme,
    memory=memory, feature_parameters=confirmed_parameters,
    marker_feedback=P, projection_deltas=rawDP,
    shard_source=NativeDensitySource(Sbg_imp),
    source_time=SourceEventTime(event_time_si, actual_gather_si, units=units))
```

| 层 | shape/顺序 | 意义 |
|---|---|---|
| U/rawD | 各`[9,4,1,N]` | 原生状态和独立原始流体历史；旧77→hidden输入分支保持 |
| marker P/rawDP | 各`[5,4,1,N]` | 完整五通道与原始投影历史；不取dof0 |
| shard | `[2,4,1,N]` | 独立background/impurity质量率 |
| 完整源系数支路 | `[N,48]` | `[P(5),rawDP(5),shard(2)]×4 degree`；线性缩放，无有损投影 |
| full cubic Gauss求值 | 值`[E,16,12]`、R/Z梯度`[E,16,12,2]` | 在物理原生单位中求值/求导，再分别缩放 |
| 闭合/解释量 | `[E,16,14]` | 固定CLOSURE_SCHEMA，见下表 |
| Gauss拼接/聚合 | `[E,16,50] → [N,50]` | 12值+24梯度+14解释量 |
| 源时钟 | `[N,3]` | gatherJU、eventJU、注册restart零投影phase标志 |
| 源编码 | `[N,101] → Linear(101,hidden) → tanh` | latent特征，加到已有场嵌入后进入stencil/CfC；不是场增量或方程forcing |
| 学习输出 | `[9,4,1,N]` | 完整九场本步原生增量；更新映射提供values/rawD；记忆`[N,hidden]` |

聚合复用G/P的正权重公式：
`f_node = Σ_(e,q containing node) w_q*|J|*R*hat_v(s,t)*f_eq / Σ w_q*|J|*R*hat_v(s,t)`。
`hat`是双线性正分割权重，**只用于特征池化，不替代cubic场求值/投影或FEM弱式装配**。聚合损失单元内部高斯分布；完整48维源系数与完整U/rawD支路继续保留，但有限宽度编码本身有压缩。节点latent stencil近似和物理BC未验收均未消失。

## 实际输入/依赖与消费者级别

固定源schema为P1..P5、rawDP1..rawDP5、shard_bg_rate、shard_imp_rate。固定14解释量依次是：Q、Q2、alpha_e、ne、Zeff、Pe、电子容量、Q_R/Q_Z、Pe_R/Pe_Z、delta_Q、解释功率率、投影历史压力增量。

| 输入 | 已验证解释/依赖 | 学习路径 | 显式物理消费者 |
|---|---|---|---|
| c1/c2 | `energy_rate=−(c1+max(c2,0))`；c1有符号 | 完整系数、Gauss值/梯度及解释功率率 | 主模式不执行电子容量响应 |
| c3 | Q2=max(c3,0)/N，Zeff | 完整空间输入+Q2/Zeff特征 | 电阻率/交换算子未执行 |
| c4/c5 | N=max(c5,.5)，Q=max(c4,0)/N，cap/低温顺序，alpha/ne | 完整空间输入+Q/alpha/ne/容量/Pe/兼容商梯度 | 压力力、输运及电子容量求解未执行 |
| rawDP | 全五通道原始历史；`delta_Q=(D4−Q*D5)*(dt/prev_dt)/N` | 独立系数、Gauss值/梯度、delta_Q及`−mass_ratio*delta_Q*Te*rho_imp` | 历史压力量仅作特征，不加场；不与rawD合并 |
| shard bg/imp | 独立质量源率，非c5 | 完整系数、Gauss值/梯度 | 主模式无显式质量容量求解 |
| U/rawD | U定义值/梯度/Pe等，rawD提供真实历史 | 原有完整系数支路、G/P及闭合特征 | 不宣称物理Gears RHS已装配 |
| 参数 | 原已确认schema：n_spi/pellet_density场景编码，eta×兼容psi梯度物理特征 | 保留原分组编码/FiLM/磁特征 | eta特征不等于已执行JOREK电阻率算子 |

所有通道/源/有效历史都通过预测响应和梯度测试，非仅被接收。下限/上限未激活的有效测试状态取c3=8、c4=4、c5=2，Q=2、Q2=4；不以截断掩盖通道作用。c3/4/5变化不改变独立功率或shard率的特征槽。它们仍可经联合学习改变多个场，这属于待学习的间接响应，不是新增直接源。

## 单位、缩放与时间因子

归一化由NativeUnits明确给出，SourceEventTime必须带相同单位；与消费者/场不一致或缺失时拒绝。路径/来源字符串只验证、保存，不作为数值特征。

| 组 | 原生物理意义/单位 | 缩放与时间所有权 |
|---|---|---|
| c1/c2 | SI功率密度×t_norm/E_norm，E_norm=1/[μ0(γ−1)] | 作为率特征；主模式不预积分为物理增量 |
| c3/c4/c5 | q²/q/数密度矩除n_norm | 消费层求均值；不再次当已归一化均值 |
| rawDP | 与对应投影通道同单位的原始增量 | 本体不改写；仅派生delta_Q消费dt/prev_dt一次 |
| shard rates | model502质量密度/JU | 网络输入率；不额外执行密度响应积分 |
| 源系数 | 各原生通道/degree系数单位 | 独立`[12,4]`正尺度 |
| Gauss值/梯度 | 各原生通道单位、相应单位/m | 独立`[12]`和`[12,2]`尺度；没有把系数scale当梯度scale |
| Q/Q2/alpha/ne/Zeff/容量 | 无量纲JU归一化；Q_R/Z为每米 | 独立closure尺度 |
| Pe及梯度 | `(rho+alpha*rho_imp)*Te`原生压力及每米梯度 | 独立closure尺度 |
| 投影历史压力量 | `−delta_alpha*Te*rho_imp`，已历史修正的压力增量 | 仅作特征，不能解释成源率、再直接积分入场 |
| gather/event | SI秒转换为JU；不是当前/前流体dt | 独立两个JU尺度；phase独立注册标志，不以缺gather默认普通零窗 |
| 学习输出 | decoder无量纲速率，scale/τ为原生系数/JU | 现有时间适配唯一产生本步增量，`ΔU=increment_scale*rate*(dtJU/τ)` |

新增SourceFeatureScales为显式正物理参考尺度（本诊断所有参考数值为1，含JU及每米单位参考），**未新增数据拟合统计**。只有明确的physical_reference来源和版本受支持；验证/测试统计、来源不明或缺组均拒绝。未来如新增拟合，需从训练split唯一支撑拟合并保存版本；本轮未开放该能力。已有U/rawD/G/P/参数统计契约保留。本诊断小fixture尺度不是实际生产训练尺度。

官方 ncps CfC不变，外层记忆门控仍为`m+[-expm1(-dtJU/τ)]*(candidate−m)`。正dt缩小时场增量、记忆变化趋零；精确0是独立完整no-op，保持U/rawD/记忆，不能当restart。正步新rawD直接保存计算增量；精确0保留旧rawD。Gauss解释量用于no-op的输入校验不推进任何事件；普通ReferenceRollout事件仍要求正dt。

## 状态、可微范围、教师与版本

- `native-source-field-v2`绑定coupling_mode、`native-joint-source-encoding-v1`、固定schema、`native-source-linear-scales-v1`、每组尺度/来源、几何/单位、源/shard/闭合及原有速率输出语义；样例配置`config/native-joint-source-small.json`及schema位于`docs/schemas/native-joint-source-v1.schema.json`。
- 新旧模式精确恢复互斥；field-only和旧冻结容量检查点不静默升级。构造器可以接收调用者显式提供的已有field模型，复用其U/rawD/G/P/参数/stencil/CfC/decoder权重，新源投影层随机初始化；这是新模式初始化，不是等价续跑，没有自动迁移加载器。
- 完整ReferenceRollout检查点保存原有全部物理状态、rawD、学习记忆、反馈及投影历史、gather/粒子residual、shard、时钟/步长计划。恢复不触发restart，普通窗口不重置云。注册restart保留粒子、清零反馈及历史，先执行正dt启动场事件。
- 加载器先在私有模型probe加载，并重新核对其配置摘要。发现源尺度张量与声明元数据不符时失败；修复了原加载器对这种损坏未拒绝的问题。模型/状态/RNG不部分提交。
- 给定显式源的Torch场步：U、rawD、P/rawDP、双密度率、记忆和Tensor参数值均可微；两步第二步损失可以回到第一步场/记忆、初态及核心参数。截断明确detach，不清空物理状态。第二步闭合/兼容梯度由实际预测场重新求值。
- NumPy源/定位/投影/shard与no_grad ReferenceRollout继续为非可微参考范围；自生成源两事件运行与给定源Torch多步训练分别验收，不称为端到端可微闭环。
- 源教师策略与场教师策略独立：源必须由调用者显式提供或自生成；场TF仍必须U/rawD成套替换、记忆保留并重新求梯度。新模式没有修改旧field-only训练器，没有实现含源课程优化器或新的窗口教师调度。

## 测试与运行证据

逐片红→绿：缺新源编码模块；formula-only消费者缺共享index；源单位缺显式校验；解释功率/历史特征尚不存在；损坏尺度恢复未拒绝。各片最小实现后复验。另一次解析fixture断言把原生cubic导数系数错当普通Hermite导数：修正预期为±1/6（端点导数/3后再/scale2），没有改实现或放宽容差。

新增15项测试（含参数化），旧12项耦合/容量测试保留；最终相关回归 **265 passed / 38.27s**，无跳过真实案例。

- 禁用torch.linalg.solve时主场步仍执行；零decoder后场保持、rawD=0，无隐藏物理skip。
- 九场全部degree更新/梯度；两个seed逐通道、逐历史/独立源高阶扰动改变预测。
- 解析c1=R−2：物理R梯度1，独立梯度scale3得到1/3；正权池化解析左5/14、右11/16，独立value scale4；原生导数系数/3和方向符号保留。
- 电荷Q/Q2手算；负c2截断后的功率率−.02；ratio=.5、D4=.1、D5=.02、N=2、Q=2时delta_Q=.015，与投影历史压力量的独立手算相符。
- 移动实际marker位置改变其完整空间反馈、新特征及学习响应。
- 非零记忆/偏置的完整no-op、正dt极限与有限局部变化率；非均匀两步BPTT与显式截断；参数Tensor梯度与分组响应。
- 两次真实小云参考源事件；顺序记录source→field→shard，shard确实拿OLD场，第二源拿第一次预测场；new-model恢复逐位一致；普通窗口、模型恢复、restart不混淆。
- 源/场/shard阶段故障原状态不变；单位/输入/时钟/尺度/模式不兼容、旧模式恢复及损坏尺度明确拒绝。

解析求值/聚合float64容差1e-14；恢复用逐位相等。随机响应比较容差只用于识别路径是否不同，不作为物理误差标准。

独立小诊断产物：`output/native-joint-source-2026-10-08/small-seed23/{joint.pt,diagnostic.json}`。未训练2691参数，其中源编码816参数；4节点/16唯一DOF，2次source/field/shard事件约0.0304s（一次CPU开发诊断，无JOREK工作单元对照，不宣称加速）。活marker从1到2，所有field/degree增量范数及gather/时钟写入JSON，新模型恢复相同。使用显式解析SCD provider与中心出生近似，不称为生产ADAS完整云运行。

## 真实数据资格与未解决项

复跑有限phaseA审计：v400 4109–4112、injInner 2702–2705，注册restart4111/2704，普通后续步长0.05/5 JU，相差100倍。完整轴读取、真实局部载荷/Torch反向及真实registered初始化回归继续通过。`phasea-audit.json`单独保存；未扫描全run、未使用这些局部案例进行训练。

完整真实云运行仍不具资格：源一致投影采用小稠密128唯一DOF上限，真实3986元素/4125节点超限；小云上限10000活marker；普通projection缺原始deltas/residual/gather来源，不能任意邻帧差分、补零或读未来。完整生产出生分布、PRB/CX、rocket/nonlocal shard model、滤波和BC也未覆盖。真实局部载荷的past-zero特例不推广为普通历史恢复。

后续主模式不要求稀疏电子容量求解或完整九场矩阵作为前置；需要的是可扩展源投影、可恢复初始化/历史资格、生产源近似验证和适当训练窗口。压力/电阻率/交换显式算子仍未实现；其相关闭合特征已进入学习路径。这两种覆盖级别不能互相替代。

## 可复现命令

```bash
.torch-cuda-env/bin/python -m pytest tests/test_native_joint_source.py tests/test_native_coupling.py tests/test_marker_state.py tests/test_fluid_history.py tests/test_native_state.py tests/test_marker_feedback.py tests/test_native_gauss.py tests/test_native_features.py tests/test_native_field_step.py tests/test_native_temporal.py tests/test_native_windows.py tests/test_native_curriculum.py tests/test_source_module.py tests/test_field_propagator.py -q --junitxml=output/native-joint-source-2026-10-08/regression-complete.xml
# 265 passed / 38.27s
PYTHONPATH=.:tests .torch-cuda-env/bin/python tools/diagnose_native_joint_source.py --output output/native-joint-source-2026-10-08/small-seed23
# 已存在输出会明确拒绝；重跑请指定新目录，保留旧产物
PYTHONPATH=. .torch-cuda-env/bin/python tools/audit_native_coupling.py --output output/native-joint-source-2026-10-08/phasea-audit.json
```

最后检查点加强单独回归73 passed /20.19s，之后已纳入265项最终回归。compileall通过；本轮文档diff whitespace检查通过。schema由系统python3的jsonschema验证样例通过；Torch环境没有jsonschema，未安装新依赖。首次诊断仅`PYTHONPATH=.`缺tests fixture导入，改用上列显式路径后成功；该入口明确是测试fixture诊断，不是生产训练入口。
