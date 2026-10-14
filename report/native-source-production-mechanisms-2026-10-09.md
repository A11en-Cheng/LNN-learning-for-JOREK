# 生产源机制：一手追踪与实施（agent A，2026-10-09）

范围：nonlocal OLD 场诊断、生产分布出生/权重/RNG、独立 shard 密度核。遵循 AGENTS、现行 ADR18–22、roadmap_v2 与指定三份报告；v1 历史不作 gate。使用 codebase-design/tdd；测试 seam 为用户已确认的源推进、含源场步、完整事件及 checkpoint。本文件由 A 独占，先研究后实现。不训练、不调用 optimizer、不搜索参数；JOREK 和原始数据只读。

## 所检查版本与证据边界

只读 `/home/ac_desktop/JOREK_code/jorek` HEAD `54c675bcc39c33b082c0c35046e28122dc53b3aa`；工作树有 `communication/eqdsk2jorek.f90` 修改和未跟踪 `poincare`。生产日志声明 `v2.25.12.0-1042-gb4165893f-dirty`。以下为所检查源机制，不等同生产 dirty 构建认证。

## 实际输入与时序

|一手源码（上述版本）|事实及实现约束|
|---|---|
|`core/mod_jorek_timestepping.f90:393,411–426,446,510`|先解九场候选；nonlocal 表 Te 首值零才在 OLD 场初始化；随后 shard update、提交场、更新时间；再更新平衡诊断并 int3d_new。不能任意每个 shard update 重算缓存，也不能用场候选更新 shard。|
|`diagnostics/mod_integrals3D.f90:438–447,839–843,1740–1777,2531–2557`|目标 reduced-MHD 用当前 ψ 作磁通坐标；n_flux 个等 ψ 壳，末端不含边界；∫neTe/∫ne 与 ∫ne/∫1，非算术平均 Te。任意空壳或非正电子总数失败。n_nonlocal_array 点按最近粗点及离轴方向线性插值/末壳外推。|
|同文件`:1034–1036,1103–1108,1131–1146,1764–1765`|Ne 诊断质量比额外乘 `2/2.01455279245`；marker 投影第4/5通道得到 Z，c5 的 corr_neg_dens 系数 (.1,.1), floor .001，Z 上限10；壳电子密度用未修正的 r0/rimp0 与 alpha_e。不能替换成 pellet 局部 Z=0 分支。|
|`models/equil_info.f90:982–1024`|需要实际 ES ψ轴/边、LCFS_lost、X-point case及上下 X-point ψ/Z；带 Z 的 private-flux 修正逐个执行。轴丢失或 Δψ<1e−6 返回1.01。|
|`models/pellet_module.f90:270–315,464–480,699–734,864–887`|旧速度移动→旧烧蚀率侵蚀→新位置 OLD 场→局部/nonlocal Parks3 取 max→磁减弱→rocket6；非局域采样按最近表点向轴侧选整数索引，无线性插值。rocket6 仍用局部 ne/Te。|
|`particles/examples/marker_spi_source.f90:94–96,350–355,394–437,809–817`|宏包目标 weight=sum(spi_quantity)/n_particles；出生每窗口一次，在子步前；宽度=max(radius×ratio,min)，出生框 ±5宽度；总源率按species和MPI分配；数量 int(rate×actual_gather/weight)，仅 atoms>1e14 强制至少1；拒绝概率 float32(max(source×dv/Pnorm,0))，dv=.01² R_geo 2π/n_plane，不新增概率clamp。|
|`particles/mod_initialise_particles.f90:391–418,420–460,497–544`|出生框与域逐方向取交，无交则回退整个方向域；圆柱体积均匀proposal；域外拒绝；使用前空槽，slot lifetime加1、t_birth保持；V_prev在sim_time求值（不是residual起点）；最终weight float32(rate×gather/count)，P_imp初始中性。每次调用新seed广播、按rank/thread拆PCG32。|
|`tools/mod_sampling.f90:74–84`|proposal R=sqrt(u(Rmax²−Rmin²)+Rmin²), Z/phi线性。|
|`models/mod_source_shape.f90:27–42`；`models/mod_injection_source.f90:161–171,278–279,361–405`|poloidal exp(−distance²/width²)，环向取最短角差；体积>0用记录数值体积，否则解析πR ns_tor_norm width²；独立密度是atoms/s×t_norm/n_norm×shape/volume，再除central_mass/20；不是c5，不可经marker滤波矩阵。|
|`models/model502/initialise_parameters.f90:209`；`diagnostics/mod_integrals3D.f90:447,1494,2495–2503`|解析fallback的ns_tor_norm=width_phi sqrtπ erf(π/width_phi)；实际数值source_volume按离散phi平面累积shape×R J w delta_phi，不可把有限平面替换成连续环向积分。两实际H5均n_plane=n_tor=n_period=1。数值体积随诊断更新，有独立缓存生命周期。|

两目标 namelist `:153,487–488,501,516–517`：n_flux=40、n_nonlocal_array=1000、nonlocal=True、ns_deltaphi=1000、radius_ratio=1.4；v400 radius_min=.05，injInner=.01。未显式覆盖的 spi_num_vol=True 与 ns_delta_minor_rad=0 只能按所检查源码默认链说明，不能据此认证 dirty 构建。

## 最小协调需求（先报告，不编辑协调文件）

A仅编辑 native_shard_dynamics、新 native_* 模块和新增专项 tests。`native_coupling`/`marker_state`/`reduced_marker`由主线程协调。新模块必须显式接收完整 OLD 场及其同事件反馈/平衡，缓存诊断带来源时刻；缺输入失败。独立密度需要未滤波 projector。出生 rawD 模式需要传入 sim_time sampler；NumPy 全局 MT19937 是模型分布流，完整既有 RNG checkpoint/失败回滚可覆盖，但不是生产 PCG32 字节复现。实际事件接入状态、红绿测试和剩余项在实施后追加。

## 实现、输入与接入范围

`src/models/native_source_nonlocal.py` 的 `NativeFluxEquilibrium.bind` 要求显式 ES：psi_axis/bnd、LCFS_is_lost、xcase、上下 X-point psi/Z、SI时刻和来源，并绑定完整场digest。它不是平衡求解器或ES真实性认证器。`NativeNonlocalDiagnostic.build` 消费完整九场、已安装的完整五通道aux、同场ES、n_flux/n_table及feedback自身时刻，生成带场/反馈/mesh指纹的缓存。字段可先后错步，禁止未来反馈；不要求反馈时间等于新场时间。完整几何R/J/四点Gauss及2π进入壳积分，电子温度按电子数加权。目标限定model502、WITH_TiTe+Impurities、Ne marker、H1、n_period=1；不提供stellarator/fullMHD或非marker日冕分支。

`src/models/native_shard_dynamics.py` 保留既有局部路径版本；nonlocal模式升级v2，要求shard字典显式含 `nonlocal_diagnostic` 和 `old_equilibrium`。后者必须匹配本次OLD场及time，缓存不得来自未来或另一mesh。先移动、旧rate侵蚀、新位置OLD局部诊断、按缓存表查值并取max，再磁减弱、用局部温压rocket6。缺缓存/ES明确失败；不会把开关改成false或用local冒充nonlocal。表与当前ES可以来自不同诊断时刻，这对应源中缓存生命周期。

`src/models/native_source_density.py::NativeShardDensityKernel` 显式配置radius_min/ratio、deltaphi、spi_num_vol、t_ns和离散环向planes，保留.05/.01。当前只支持纯Ne、无drift、圆形poloidal云。`rates_at` 返回无dt因子的杂质质量密度rate/JU；`refresh_volumes` 是调用者显式选择的诊断事件。数值体积缺失失败，有来源的零体积按源码解析fallback处理。缓存带SI时刻和来源，不擅自按每个source调用刷新。`density_source` 用独立**未滤波**FEM投影形成现有 `[2,4,1,N]` 场输入（background/impurity顺序），不用c5或marker滤波矩阵。点核对照源码；其有限cubic系数表示仍是给学习场的投影表示，不能说重新求值后的每个Gauss点严格等于原指数核或已移植JOREK弱式密度装配。真实H1有限phi平面与解析fallback环向积分分开。

`src/models/native_source_birth.py::NativeProductionBirths` 实现所检查出生分布：窗口前出生、源强/MPI计数、严格>1e14阈值、±5宽度框及域交/逐方向回退、圆柱体积proposal、逐轮pending槽拒绝、float32权重和拒绝概率、sim_time初速度、完整中性P_imp。rawD模式必须显式传入匹配 `NativeSourceTimeSampler`，不以端点差恢复rawD、不取residual起点。返回云由既有 `MarkerCloud.insert` 按前空槽插入，slot lifetime加1且t_birth保持。不扩容，不覆盖活云，提案预算耗尽明确失败。

出生使用现有被完整保存的NumPy全局MT19937流；模型本身不另藏seed或Generator。函数自身拒绝/错误回滚全部NumPy RNG，完整事件下游失败由已有事件driver回滚。检查点保存624字状态、cursor及Gaussian cache字段，恢复不是仅重设seed。**生产随机分布机制与逐粒子复现明确分开**：原码每次调用重seed并按MPI/OpenMP拆PCG32；本轮MT19937重放只保证本模型相同检查点的复现，不能声称生产粒子逐个一致，概率>1仍按源码比较而不新增clamp。

`src/models/native_source_production.py::NativeProductionShardModel` 把上述机制组合成现有 `density_source/births/update` Interface，并提供纯候选 `refresh_diagnostics`。配置指纹包括子机制、物理参数、RNG语义、提案预算、壳数/表数和必需外部ES；无隐藏可变物理状态。更新后的shard时刻写入候选，nonlocal/volume缓存留在shard状态，进入既有模型检查点。

接入状态必须区分：

|范围|本轮结果|
|---|---|
|独立source诊断/出生/密度、非局域shard update|已实现，有独立oracle|
|现有NativeMarkerSource实际source.advance|专项测试用新出生云真实执行电荷/投影/输运；验证槽位/寿命/时刻|
|现有advance_coupled_reference + joint_learning九场|小解析fixture两次完整事件通过，使用新组合provider、独立密度及nonlocal update|
|postcommit诊断刷新|fixture显式调用refresh_diagnostics；**尚未放进现有协调driver的原子回滚范围**|
|rawD出生在完整事件中|模块正确消费显式sampler；现有driver先births后构造sampler，仍需主线程接线|
|目标两run完整生产闭环|0事件；没有改qualification或用fixture授予生产资格|

上述小joint事件的decoder显式归零，用来隔离确定性源/shard/RNG，场时间记忆仍按模型推进；提案R_geo=1e8是明确的解析fixture大dv，用来隔离检查点机制而限制拒绝成本，并非目标run配置。未进入冻结容量参考，不以随机网络预测准确度作为gate，也不宣称反馈物理已经学会。

## 红绿、独立oracle及检查点证据

每个机制逐个红绿：nonlocal模块缺失→加权温度/体积通过；shard仍拒绝nonlocal→加入缓存表max通过；密度模块缺失→独立核比值通过；缺离散plane接口→数值体积/独立投影预算通过；出生模块缺失→计数/float32/速度/生命周期通过；组合provider缺失→实际joint两事件/恢复通过。随后真实Fortran精度oracle先红，再修正real-kind运算。

- 解析矩形R∈[2,3],Z∈[0,1]，ne=1+Z，Te=3600Z eV；期望电子加权Te=2000eV、平均ne=1.2e20、体积=5π。不是把实现输出当预期值。
- 旧热缓存4000eV、本次OLD局部2000eV，r旧.003、dt1e−6、旧rate侵蚀后r=.002：独立Decimal Parks3值 `2.565605253370680852986595e24`；rocket仍局部，速度增量 `.026259335139022505` m/s。保留旧rate侵蚀顺序。
- 已安装的lagged aux平均电荷4，rho=2/rhoimp=1，诊断Ne质量比有2/2.01455279245修正；独立Decimal ne=`9.588441867591028689e19`。另验减向psi表的axisward离散索引、private区域排除、轴丢失、空壳和错误OLD场digest失败。
- 核空间偏移一个width比值e^-1、最短角差.25比值e^-.0625；.05→.01中心强度比25。数值体积归一化后独立投影总杂质质量rate/t_norm=2500，背景为0；不复用五通道反馈。
- 出生以外部随机数oracle验证柱坐标proposal，actual gather计数、MPI local source、严格>阈值、float32包权重、sim_time速度、完整中性P_imp及t_birth/slot lifetime。rawD速度例当前vpar=.001/rawD=.0004，在sim=.5ns取.0008，区别于residual=.3ns的.00072及端点差插值。
- `tests/native_birth_reference.f90` 独立摘录 `f_source_imp`。源码两个0.01字面量先以默认real相乘，再被real8 R_geo提升，最后dv赋real4；Python先全double再转real4会错一个ULP。R_geo=2.5,n_plane=1得到dv=`.00157079624477773905`，接受概率=`.0451355651021003723`。通过接受/拒绝行为（非读内部成员）检验边界。最初测试未拒绝而红，修正运算顺序后绿。
- 两actual joint base事件从空8槽云依次产生1/2活marker，plan_cursor=1/2；第一次后保存到新模型、新source及新shard provider恢复，所有场/rawD/记忆、完整cloud数组、feedback/rawDP、shard位置/速度/radius/rate/volume与时钟逐位一致。MT19937全部words/cursor/cache与连续路径一致。随机出生后atomic provider失败，既有事件driver恢复RNG且输入云不变。无未来标签、无真值重置、无训练。

## 真实有界独立密度诊断

只读v4004112和injInner2711，完整几何与原始数值source_volume。每run选择一个最大记录烧蚀rate的shard作有界独立核诊断（不是裁剪云进行rollout）。HDF直接验证n_plane=n_tor=n_period=1；几何mesh为formula-only，不建立全局矩阵，不推进marker。

|run/记录/0-based shard槽|radius_min|记录spi_vol m³|重算m³|相对差|秒|
|---|---:|---:|---:|---:|---:|
|v400 / 4112 / 478|.05|.07863742597064208|.0786374259706418|−3.552713678800501e−15|.8095|
|injInner / 2711 / 531|.01|.0008267800383469544|.0008267800383469538|−6.661338147750939e−16|.7271|

允许相对误差3e−11，实际远小于该值；记录spi_vol是独立JOREK产物。峰值RSS分别555804/556236KiB，包含pytest/Torch进程初始化，不是增量内存或完整云资源成本。匹配仅支撑这两个独立体积诊断，不证明dirty构建的其余机制、未来场或全云闭环。真实HDF有psi_axis/bnd/xpoint历史数组，但所查快照无LCFS_is_lost/xcase字段；namelist有xpoint=True/xcase=3。没有把缺失的实际ES状态默认成False，也没有把最后磁通表文件当合法当前缓存。

## 协调文件的具体最小接线（A未编辑）

1. `native_coupling.advance_coupled_reference` 及相应Torch事件Adapter：先调用既有source_time_sampler，再对新生产provider的births传time_sampler；source.advance继续消费同一sampler。旧SmallShard参考可保留独立分支。
2. OLD shard更新前，必须有匹配OLD完整U/time的实际ES，以及合法nonlocal缓存（首表为零时按源码初始化）；不存在实际ES provider时明确失败。不能从未来记录补ES或从新场候选偷读。
3. 场/shard候选提交后，在**同一个try/回滚范围、返回已提交状态之前**，用实际新U、新ES和本步已安装的（可能lagged）feedback调用 `refresh_diagnostics`；刷新失败整个事件失败。fixture当前是在driver返回后显式调用，所以未证明该新阶段的统一原子性。
4. 缓存已位于shards且全球NumPy RNG已有完整保存，无需新公共状态容器。恢复必须保留源/诊断构造配置指纹和缓存；不额外执行JOREK restart、不清云/反馈/余留时间。预算、源config与真实执行绑定由主线程统一办理。

这些是明确交付需求，不是请求扩大写范围。已与B直接沟通；A没有编辑其边界/场/资源代码或协调文件。

## 修改路径与未完成项

修改既有授权文件 `src/models/native_shard_dynamics.py`。新增 `src/models/native_source_nonlocal.py`、`native_source_density.py`、`native_source_birth.py`、`native_source_production.py`；新增对应 `tests/test_native_production_nonlocal.py`、`test_native_production_density.py`、`test_native_production_birth.py`、`test_native_production_events.py` 和 `tests/native_birth_reference.f90`。本报告是A唯一文档写入。

未完成：真实ES finder/有效ES缓存来源、上述rawD出生及诊断原子接线、非局域真实表独立生产对照、PCG32/MPI/线程逐粒子重现、全846 shard与两个实际云的有界执行成本/两事件、dirty生产构建及有效配置证明；Mach/natural边界及定位器由其他所有者处理。未新增完整物理求解器，未移植完整JOREK矩阵，没有冻结容量回归、训练或optimizer curriculum测试。生产资格和C0仍未授予。

## 实际命令与结果

环境 `.torch-cuda-env/bin/python`。最终定向回归 **56 passed / 4.40s**，JUnit `/tmp/native-A-production-20261009-yMJOsj/regression-final.xml`。前一版本55 passed/4.28s保留于同目录regression.xml；后加Fortran real-kind oracle并修正后重新跑最终56项。专项新测试17项（含两真实单shard体积诊断），其余为时间、局部shard、源/辐射、状态/检查点及joint场依赖回归。没有执行包含冻结容量分支的旧完整coupling/joint事件测试；只选择三个纯joint场测试入口。没有运行curriculum。

```bash
.torch-cuda-env/bin/python -m pytest -q \
  tests/test_native_production_nonlocal.py tests/test_native_production_density.py \
  tests/test_native_production_birth.py tests/test_native_production_events.py \
  tests/test_native_shard_dynamics.py tests/test_native_source_time.py \
  tests/test_native_source_time_integration.py tests/test_native_production_radiation.py \
  tests/test_native_checkpoint_control.py tests/test_marker_state.py \
  tests/test_native_joint_source.py::test_joint_main_path_needs_no_capacity_and_zero_decoder_has_no_physical_skip \
  tests/test_native_joint_source.py::test_all_sources_histories_and_high_degrees_reach_joint_prediction_and_gradients \
  tests/test_native_joint_source.py::test_joint_positive_time_limit_complete_noop_and_two_step_bptt \
  --junitxml=/tmp/native-A-production-20261009-yMJOsj/regression-final.xml
.torch-cuda-env/bin/python -m pytest -q -s tests/test_native_production_density.py -k real_snapshot
git diff --check
```

最后一条只检查Git已跟踪diff，新增/原已未跟踪文件另外逐文件 `git diff --no-index --check /dev/null <owned_path>` 检查。五个模型模块compileall通过。JOREK只读status仍为研究开始时的两项既有状态；A没有改协调文件、field/boundary/training/data、原始数据、其他agent报告或Git索引。
