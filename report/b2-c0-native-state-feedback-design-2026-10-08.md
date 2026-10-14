# B2→C0 完整原生状态与反馈 Interface 设计（2026-10-08）

> **后续主路径修订（ADR-0022）**：本报告的冻结容量混合方案及历史结果保留。当前主路径改为双源/闭合条件下的九场联合学习，不再必须先解冻结容量；旧响应仅为显式互斥参考模式。新接口与机制证据见 [含源联合学习报告](native-joint-source-learning-2026-10-08.md)。

> **同日后续决策**：[ADR-0021](../docs/adr/0021-native-state-cfc-window-curriculum.md)及
> [窗口/残差CfC课程设计](native-window-cfc-curriculum-design-2026-10-08.md)确定当前执行方向：
> 原生状态与联合场步结构已交付，下一步窗口、兼容值/梯度、参数路由与可替换CfC核心。
> 本文冻结容量响应仍是候选近似，不是训练前置；§7/末段A→B→C的旧建议不再是强制顺序。
> 参数仅浅层/响应形状不变的限制被ADR-0021定向修订。物理顺序与双源对象不变。

状态：**设计建议，尚未实现或训练**。推荐保留并预测九场全部原生自由度，以 stencil-conv 学习联合推进；不把完整移植 JOREK 九场矩阵设为前置。本文没有改变 roadmap 或已接受 ADR；ADR-0018/0019/0020 为依据，尤其 0020 决定3/4/5/9。冻结的 `[nodes,5]` 共享 S 保留为兼容观察，新增完整原生反馈 Interface，不偷偷改写其布局。

使用 codebase-design 明确 Module、Interface、Seam 和 Adapter；使用 domain-modeling 区分原生状态、弱式载荷、历史增量与已提交预测状态（词汇补充至 CONTEXT）。本报告不将设计选择登记成已接受 ADR。若以后改变完整自由度目标、以云摘要代替可推进物理状态或改变冻结 S，须另提建议。

## 1. 核查依据与当前能力

已阅读 roadmap_v2、ADR-0018/0019/0020、CONTEXT，以及最新 [native-gauss 实现报告](native-gauss-local-marker-loads-2026-10-08.md)、[一手源码报告](native-fem-gauss-primary-evidence-2026-10-08.md)、[marker-feedback 报告](marker-feedback-consumption-2026-10-07.md)、[persistent-state 报告](marker-persistent-state-implementation-2026-10-07.md)。旧 ADR-0001..0017 不作 gate。

本轮重新只读核对 JOREK HEAD `54c675bcc39c33b082c0c35046e28122dc53b3aa`；下面 M 表示 `models/model502/mod_elt_matrix.f90`。基函数、生产普通路径和单位的证据以以上源码报告为准，本轮额外核对 M1784–1790、2784及2941–2955的时间行；没有混用 FFT。

| 分类 | 核查结果 | 位置 |
|---|---|---|
| 源码事实 | 全场及原始增量按全部 degree/harmonic 求值，乘带符号 element size；原生 cubic 不是标准导数系数直接拼接 | M263–326；基函数/几何详见一手报告 |
| 源码事实 | 电子时间行不仅有 Te 自身容量，还有密度/杂质密度交叉块；marker 分支 alpha_e_bis=alpha_e | M2784（95）、相邻98块、2941（99）；M1784–1788 |
| 源码事实 | 直接电子源载荷含一次 dt；投影历史载荷不另含 dt/theta/zeta；流体历史是另一组带 zeta 的项 | M1728、1784–1788 |
| 实现事实 | FieldPropagator 只接受 `[N,9]`，decoder 只输出相同视图；自由学习 source_proj 把 `[N,5]` 映到 hidden forcing | `src/models/field_propagator.py:653–690`；source_proj定义602 |
| 实现事实 | 原生训练配对读入截取 degree0/harmonic0；不能据此监督完整原生状态 | `src/data/native_run.py:682–699,1707–1708`；StepDataset沿用该路径 |
| 实现事实 | ReferenceRollout携带完整场、原始 fluid_deltas、计划游标和前步长，但场shape对驱动仍是不透明的；可传shape相同的非原生数组 | `src/models/marker_state.py:321–356` |
| 实现事实 | solve_fluid显式收流体历史和时间系数，返回values与post-update deltas，旧场更新shard后提交；外层粒子→流体在副本上执行 | 同文件 `389–456` |
| 实现事实 | 原生求值/两载荷只到局部向量；rhs_indices是元素内行号，不是全局受约束自由度编号 | `src/models/native_gauss.py:162–202` |
| 实现事实 | AugmentedState只是降阶源窗口便利对象，缺完整流体事件历史，不适合作为C0顶层状态 | `src/models/source_module.py:1025–1087` |
| 实现事实 | 模型保存已有权重、模式、rollout，但_module_signature主要面向marker配置；不足以证明未来FieldPropagator网格/单位/BC/模式完全相同 | `src/models/marker_state.py:280–316` |

上轮209 passed/1 skipped证明已登记组件，不是本文新架构的测试结果。本轮仅设计和文档检查，没有运行训练或声称新路径可运行。

## 2. 推荐状态：保留完整系数，节点视图只作观察

**架构选择**：唯一权威物理场为 `U[9,4,H,N]`（原生JU、明确dtype），原始 `D_u` 同shape。持有前后两个完整场端点及各自时间；几何 `x[2,4,1,N]`、signed size、vertex、模式计划、约束映射为不可变网格上下文。首个实现支持已验证生产轴对称 `H=1`、cubic/Gauss4；Interface保留H轴，其他配置未经实现则失败，不能默默丢模。H>1不是本轮主目标，更不是开展谱网络分支。

内存重排为 `[N,4,H,9]` 是可逆Adapter；encoder不得把原生状态本身压成 `[N,9]` 后仅靠latent回填。每个节点保留degree/harmonic标记，decoder显式输出 **全部** `[N,4,H,9]` 候选增量。原生U作为显式skip携带；latent只是计算表示，任何latent瓶颈都不能成为检查点唯一物理状态。九场联合处理，不拆7+2。

节点dof0丢失：两个方向的切向系数及混合系数、由它们决定的单元内部曲率/梯度/连续性和积分；若截harmonic还丢失环向结构及导数。系数须与signed size、节点切向和源基函数共同解释，不能称作三个普通物理梯度值。两个U即便dof0完全相同，仍可有不同Gauss温度/压力、marker路径及载荷。零补/四角插值无法无损恢复。完整U也不能无损代替D_u，原始增量独立保存。

现有stencil稀疏matmul的空间机制可复用；算子需扩展为带degree块的FEM邻接matvec，保留size、试探/测试基和共享节点关系。将旧节点算子简单广播4次只能作标注的临时学习近似，不能称作原生空间算子已对齐。既有边界节点整行复位也不等价于全部degree的JOREK约束：需要逐场/degree/harmonic的约束、轴及冻结模式映射；未知配置失败，不能仅acknowledge代码就声称物理验证。

完整状态规模本身不是百万粒子回放：phaseA H=1,N=4125时一份九场float64约1.19MB。两个端点加D_u约3.56MB；marker槽池、图中间量和矩阵另计，不据此推断训练成本。

**不推荐替换目标为降阶重建**。若以后提出dof0→高阶系数重建：新增非唯一逆问题/平滑先验、边界与曲率偏差、不能恢复截断模式或D_u、projection历史误差；必须用相同dof0不同高阶系数反例、独立Gauss值/导数、积分/局部载荷、长步滚动和边界连续性测试量化，并提交用户确认。本文没有采用该替代。

## 3. 载荷怎样进入物理更新

必须分开三类对象：`b_e`是测试空间对偶载荷，`ΔU`是原生试探系数增量，`z`是无物理单位的latent特征。不得做 `U+b_e`、`z+b_e`，也不得把任意线性5→9映射叫作物理响应。

**推荐最小可解释响应**：新增 `ElectronTimeResponse` Module，先提供完整自由度的标量电子时间块Adapter，再由联合stencil学习其余耦合响应。它不是完整九场矩阵，也不是把16个局部载荷分别求逆后平均。

令 `φ_a` 为带size/HZ的原生测试/试探基；沿源码局部约定组装：

```
b_e = scatter_sum_elements(b_direct + b_projection_history)
C_e[a,b] = Σ_elements,q φ_a φ_b R J w (rho_raw + alpha_e_bis*rho_imp_raw)
(1+zeta) C_e ΔTe_feedback = b_e      # 施加支持的齐次增量约束后
```

依据是M2941的99时间块；marker分支dE_ion_dT为零。这里只冻结密度/电荷，省略95/98交叉、对流、传导、交换和其他九场块。因此这是**来源可解释的冻结容量响应近似/预条件响应**，不是JOREK完整更新。即使C_e可逆，也不能声称电子能量完整守恒或九场耦合完成。

只装配共享自由度的scalar容量/投影矩阵和小网格参考求解，不装配全九场Jacobian；这是允许的最小确定性支撑。可使用稀疏matvec/迭代，禁止degree行和lumping默认替代一致容量（导数基可能有符号/零行和）。支持的网格需核对方向和约束：不得偷偷把signed J取abs，负容量、奇异/轴退化、未支持BC必须明确失败并登记；不能用epsilon补逆。若最终需正向Riesz度量而非源码时间块，须另行登记其用途与差异。

时间因子所有权：native消费层已给直接项乘dt_JU、projection deltas乘dt/dt_prev；局部载荷已乘测试函数/size/HZ/R/J/w；global scatter不再乘这些因子。响应只应用本身LHS的 `(1+zeta)`。不将投影历史再乘dt，不乘theta/zeta，不覆盖原始D_S。D_u的独立历史项由流体更新Module消费一次，不能重用D_S代替。

**联合学习保持B2目标**：`NativeFieldStep`以完整U、D_u、时间系数、几何及确定性闭合量为输入，联合stencil输出九场原生残余增量；`ΔTe_feedback`作为有单位的参考增量经显式缩放后编码为局部特征，并作为电子分量的物理skip。最终在同一原生系数空间组合 `ΔU_candidate = inject_Te(ΔTe_feedback) + ΔU_learned`，经统一约束映射更新。相加的是同单位场增量，不是载荷或latent。参考增量不得再次按dt积分；学习残余的dt定义单独约定，decoder输出本步增量而非未声明的rate。

这一分解把冻结容量漏掉的联立响应留给**联合**stencil-conv学习，仍然近似；不保证网络不会抵消物理skip，不在未训练时宣称准确或稳定。需要源响应方向/缩放、反馈消融及物理dt刚性准稳态测试。继续遵守ADR-0018：不引入物理残差训练loss，不要求完整九场矩阵作为训练前置；训练日后仅用符合契约的完整原生端点/增量监督。无反馈时学习背景演化仍可非零。

## 4. 五通道闭合：不止两个载荷

从同一完整U/S/D_S求值获得 `ConsumedMarkerFeedback`，明确保留原始/修正流体、Q/Q²/N、alpha_e及其梯度/增量、ne_JU与ne_SI、Zeff和原生marker线性化。通道3/4是矩，先按源码除以带下限通道5，不能先验当均值；通道5不是rho_imp或注入源。

| 物理消费者 | 输入/直接依赖 | 拟议接入与尚缺工作 |
|---|---|---|
| 电子直接能量/投影历史 | c1/2、D4/5；方程9 | 已验证局部载荷，新增scatter与容量响应；非全电子方程 |
| 电子/总压力与梯度 | Q/N→alpha_e及梯度，原始rho/rhoimp/Te；方程2/7/9及部分矩阵压力依赖 | 独立实现Pe及一阶梯度消费者，作为有单位的局部物理特征/空间系数；高阶项仍缺 |
| 电阻率/欧姆项 | Q²/N、ne、Zeff、lnΛ、温度/eta参数；方程1/9 | 先实现并独立验证lnΛ/eta消费者，再提供局部输运系数，不能将Zeff直接等同eta；`mod_plasma_functions.f90:62–233`，M1160–1173 |
| 离子电子交换 | Q²、Q、ne、Ti/Te、质量/碰撞参数；方程6/9 | 实现成对能量交换及需要的线性化，作为明确局部物理特征/响应；M1282–1367，不能零占位 |
| 电子时间容量 | alpha_e_bis、原始rho/rhoimp；方程9及交叉块 | 最小slice只用99冻结容量；95/98联立由学习近似承担且显式登记 |
| marker下一窗口 | 旧反馈Q/N、电子密度/扩散，持久云 | 完整原生采样Adapter；不将全局摘要当充分状态 |
| 独立shard密度源 | source_bg/source_imp；方程5/8 | 独立对象和单位转换；禁止用c5充当密度源 |

各方程直接依赖沿已核查普通路径：1/2/6/7/9通过上述闭合或直接项；3/4/5/8没有marker五通道直接项，仍可经联合更新间接受影响。学习九场输出可以有这些间接响应，不能把decoder的非零通道解释成新直接源。

确定性消费者的输出可以经有版本的缩放编码进stencil局部特征，**特征影响**不是已经执行对应物理方程。需要在输出coverage中区分 `formula_verified / used_as_feature / used_as_coefficient / local_load / approximate_response / unimplemented`。只把c3–5拼进网络不能验收完整五通道物理耦合；需要消费者公式验证、参数开关、对相关输运/交换/压力响应的独立测试和后续rollout证据。深层允许局部物理系数，不把它们混同为禁止的全局condition FiLM；全局场景condition仍限最浅层。

## 5. 一个可续跑事件的状态与提交

对外使用一个深Module `CoupledStep`：`advance(committed_state, request) -> committed_state, diagnostics`。调用者不自行排序粒子/流体/shard或缩放历史。现有reference事件驱动与拟议native学习流体Adapter构成真实Seam；不新增一套仅改名的通用调度框架。

状态至少包括：完整U_previous/U_current及端点时间、原始D_u；shard位置/速度/radius/abl/species等；marker完整可推进状态（或已登记降阶Adapter身份）、S.values与原始D_S和投影时间；sim/real_particle_time关系、step_rest_time、particle_dt、场插值端点时间、startup标志；流体时间/编号、scheme、前dt、计划/游标；可选学习记忆。网格/单位/物种/ADAS/BC配置和模型版本以不可变manifest指纹绑定。原始S为 `[5,4,H,N]`，dof0共享视图只在兼容处派生。

普通事件按现有reference driver：

1. 从已提交状态创建独立candidate；marker只在已有U_previous→U_current间推进（包括源码余留允许的外推），得到新云、S和D_S，字段端点与真实粒子时间分别校验。不读下一记录帧。
2. 在当前U上求值并消费新反馈；fluid Adapter接收原始D_u、dt/dt_prev与scheme，求候选values、原始post-update D_u及可选学习记忆。源窗口跨度、实际gather time和下一流体dt不是同一概念。
3. 在**更新前场**更新shard，再验证所有候选量、约束及时间一致性；统一提交新的场端点、D_u、shard、云/反馈、时钟、前dt、游标和记忆。不因停止烧蚀删除既有marker。
4. 任一环节失败，输入状态及模型运行缓冲区不变；禁止保存部分candidate。原子落盘用临时文件+同文件系统rename，schema/完整性校验后才能加载。自定义回调不得偷改模块缓冲区/RNG；若使用随机扩散则保存生成器状态，不能只保存seed。

这里“原子”是模型事件一致性，不改变JOREK内部先后顺序。现有deepcopy在reference路径提供部分基础，但不覆盖外部回调副作用、Torch缓冲区和未来checkpoint协议；必须补测试。ReferenceRollout需加强原生layout、模式、单位和全部端点时间校验，不能只验证shape相等。

**原始D_u更新**：更新Adapter接收求得的原生增量和明确约束/冻结规则，返回values及准确保留的节点历史；不能用舍入后的values-old_values代替。自由行可保存实际求解增量，约束/冻结行遵守核查过的映射；未核查规则失败。约束修正后的历史也不可笼统设为零。学习输出和物理skip的组合仅是candidate，最终两数组来自同一个更新Adapter。

三种操作：连续窗口保持全部历史；模型checkpoint恢复完整模型/物理状态、模式、配置指纹及必要RNG，不触发startup；JOREK程序restart按已登记规则保留导入流体D_u/marker槽、重建零S/D_S、余留归零并先执行startup流体/shard事件，首步dt_prev=current dt，然后才采样粒子。不能从重复输出时间推断粒子窗口长度或dt_prev。

## 6. 最小 Interface 草案（名称尚未实现）

```
NativeFieldState(values[9,4,H,N], raw_deltas[same], layout, units, time_si)
NativeFeedbackState(values[5,4,H,N], raw_deltas[same], event_time, gather_time, provenance)
NativeStepContext(mesh_id, basis/modes, constraints, normalization,
                  resolved_parameters, closure_config, time_coefficients, dt_si)

NativeFieldStep.predict(current, feedback, shards, context, memory)
  -> NativeFieldCandidate(values, raw_deltas, memory, coverage, diagnostics)
CoupledStep.advance(committed_state, request) -> next_state, diagnostics
```

`NativeFieldStep`内部复用完整求值/五通道消费、scalar时间响应、联合stencil和更新映射，调用者只学习以上输入不变量。ReferenceRollout的solve_fluid Adapter显式构造context；字段layout/units放在类型及manifest内，而非再次增加十几个位置参数。NativeElement读取器只初始化网格，运行中不读磁盘。NumPy float64参考Adapter先保留；Torch可微Adapter若后续实现须逐项对照，不能把detach悄悄置于训练接口。首轮不承诺GPU或梯度路径完成。

失败条件：缺原始D_u/D_S或完整degree、单位/模式/时间不匹配、未知约束/闭合配置、容量不可逆、状态含非有限量、计划耗尽、配置指纹不符；不默认补零或后读未来帧。允许的零反馈仅显式注入前/登记restart事件。

## 7. 下一轮可直接实施的切片与测试

各切片先公共Interface红→绿；不训练。测试不是新gate，也不要求完整九场矩阵。

| 顺序 | 可实施任务 | 独立测试及完成口径 |
|---|---|---|
| A（下一轮首选） | 原生状态类型/版本；完整U/D_u/S/D_S读入与可逆Torch重排；ReferenceRollout adapter与原子checkpoint；冻结旧dof0接口 | 相同dof0不同高阶系数产生不同Gauss值/输入；全部轴roundtrip逐位一致；缺历史失败；同平台完整模型续跑逐位一致；网格/单位/BC/mode改动拒绝；失败不改状态；真实phaseA读取 |
| B | 小型共享节点scalar容量组装与ElectronTimeResponse；接两项载荷，不写九场矩阵 | 独立解析2元素一致容量、共享节点scatter和受约束解；独立常场ΔTe=−dt*(c1+c2)/[(1+zeta)*capacity]；非零D4/5不二次乘dt；变dt、非均匀capacity及负/奇异失败；零投影startup；不是16局部独立逆平均 |
| C | 完整degree联合stencil encoder/decoder和明确更新映射；接B作为同单位skip及特征；D_u显式历史输入 | 扰动任一degree会到可训练输出；固定小型权重验证高阶更新；九场全shape/梯度检查（Torch路径可用后）；相同U不同D_u产生不同结果；dt0只作纯恒等探针，不能执行粒子或正步事件；degree级边界/冻结历史独立oracle；不拿随机权重稳定性冒充训练能力 |
| D | 一阶压力消费者→lnΛ/eta消费者→成对交换消费者分别接入；闭合coverage registry | 单独改变c3会改变Zeff/eta/交换而非独立功率载荷；改变c4/5按独立商规则改变capacity/压力梯度；非零梯度、cap/低温分支；c1/2不伪造LHS导数；交换能量配对及各方程直接依赖表；未实现消费者显式标记 |
| E | 完整原生marker场采样与投影Adapter，运行小型持久云；统一CoupledStep | 独立native场采样oracle；不是旧双线性定位/采样；沉积→一致投影保留全部degree及rawD_S；连续2步vs中间保存恢复；停止新生旧云继续；source/solver/shard失败回滚；禁止未来文件读取；注册restart零反馈事件顺序 |

A可单独交付状态保真，不宣称接通反馈；B+C可验收“完整原生学习推进结构+两载荷响应近似”。D的已覆盖消费者才逐项验收，E打通小规模C0事件链。完整五通道耦合与自生成源闭环必须在D/E及相关实际推进验证后另行报告。真实百万槽只读取验证，训练不默认回放；现有marker降阶近似仍登记，不能因native场完成而抹去。

## 8. 未解决问题与决策建议

- 全degree约束/轴/冻结模式映射、容量的可逆支持域和方向一致性尚需源码核查；缺它们不得承诺真实全网格响应求解通过。先小型支持网格，不以静默回退扩大能力。
- lnΛ/eta/交换完整参数与必要修正导数、部分压力二阶量尚未实现；这些不是用已给Q/ne自动完成。每项消费者需独立oracle。
- 自生成反馈目前还有双线性采样/lumped投影等降阶近似；完整native场不能自行修复其信息损失。至少E的参考Adapter先承载可核查小云，再讨论标定的低成本近似。
- **2026-10-08 切片A核查修正**：不能把旧路线的“记录帧跨多流体步”泛化到当前phaseA逐步数据。实读v400的4108–4114和injInner的2701–2707，流体内部步号逐一连续，普通输出间隔匹配当前步长；登记restart分别为4111和2704，流体时间重复。v400的4112开始步长从0.5改为0.05。marker快照稀疏且时间错步，restart前零投影也有滞后时间，不能因此否定流体逐步输出。具体数值与缺文件记录见[切片A实现报告](native-state-fidelity-resume-2026-10-08.md)。这里不外推未扫描片段；即便逐步输出，端点相减仍不能无损替代原始D_u。缺失片段单独登记，不伪造子步监督。
- 学习残余能否保持预算、刚性准稳态及约束尚无证据；冻结容量skip只是解释起点，不是稳定性证明。未来测试/训练按现行ADR，不新增完整求解器gate。
- 冻结dof0共享S与新native Interface并存是显式Adapter扩展，未改ADR决定；若希望废弃旧S或宣称其为完整状态，需要先提出ADR修订建议。当前推荐不需要变更已接受决定。

推荐执行顺序A→B→C，随后逐项D并接E。下一轮可直接下发：“TDD实现A：完整NativeFieldState/NativeFeedbackState、全轴读取与布局Adapter、ReferenceRollout原生校验及完整checkpoint续跑测试；保留现有node视图和source_proj未对齐标识，不训练，不改roadmap/ADR，不实现九场矩阵。”这提供下一步真实可用状态Seam，再把已验证载荷接进有单位解释的学习推进。

## 9. 本轮检查记录

只读核查上述仓库文件/报告与本地JOREK时间行，`git -C /home/ac_desktop/JOREK_code/jorek rev-parse HEAD`确认版本。修改仅本设计报告、报告索引及CONTEXT词汇增补；无实现变更、无新运行数值测试。最终 `git diff --check` 检查文档补丁格式；旧用户改动保持。
