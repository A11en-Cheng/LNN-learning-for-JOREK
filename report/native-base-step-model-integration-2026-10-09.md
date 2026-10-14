# Base-step 实际模型接线与有限事件诊断（2026-10-09）

本轮**部分交付，真实配置有限步闭环尚未完成**。已把已有 base-step runner 接到实际 Torch 场 Adapter，补齐观察支路、持久时间原点、Dirichlet 更新行子集及检查点失败回滚；小云两事件执行/跨步梯度/新模型恢复通过。真实 phaseA 只完成全槽位资源预检，完整事件 **0**。禁止训练已遵守：无 optimizer 更新、搜索或原始数据写入。

依据：[base-step 训练契约](native-base-step-coupled-training-design-2026-10-08.md)、[生产模型源切片](native-production-model-closure-2026-10-08.md)、[大步反馈时序](native-large-step-feedback-timing-2026-10-08.md)。现行 ADR-0018～0022 的定向覆盖不变；本轮未改变 roadmap/ADR。以前的 field-only 实验和检查点未改写。实施使用 codebase-design/tdd；领域状态仍沿用已有完整物理状态与显式学习记忆，没有新增调度器或公共状态容器。

## 1. 实际执行链与所有权

`run_sequence → TorchRecomputedSourceEvents.advance → reference_particle_then_fluid / reference_fluid_event → source.advance → CoupledNativeField → shard.update → unified commit`。

Torch `values/raw_deltas/memory` 从上一事件 Tensor 直接进入下一场步；NumPy 镜像只供确定性源/shard 与校验使用，不从镜像重新创建学习输入。源数值上读取预测场 detached copy，场图跨步延续；源/shard/投影本身停止梯度。显式 TBPTT 只 detach，不清物理值、时钟或记忆。K 个 control 仍是 K 次完整交换，监督节点不自动替换状态。restart startup 和普通事件复用已有分支。

与源模型线程协调：该线程所有 `native_source_time.py`、`native_shard_dynamics.py` 及生产源研究；本侧只对 `native_coupling.py` 做 sampler/观察接入，负责 field step、source encoding、训练 Adapter/噪声/指标。在完成所有权沟通后，仅给 `src/data/base_coupled.py` 的已存在身份标签分支补一行自身时刻；没有改公共类型或已有 manifest。

### 源场时间读取修正

实际源公开入口新增 `jorek_raw_deltas_v1` 模式，必须显式选择 `field_static`。ReferenceEvents 与 Torch Adapter 都传入保存的 raw fluid_deltas、marker 场端点时钟及**已完成**流体步长；不用下一控制 dt，也不以相邻端点差替代原始历史。

检查源码 HEAD `54c675bcc39c33b082c0c35046e28122dc53b3aa`：`particles/marker_spi_source.f90:76–80` 的 import `i=-1`；`particles/mod_fields_linear.f90:405` 将 `static=.true.`，一阶求值分支使用该标志；core 更新场时钟不等同于改写 static。两组启动日志也记录 T。非 static rawD 时间插值只是另一个明确测试分支，**不作为目标默认**。生产记录声明 dirty 构建，与所检查源码的后续运行等价性仍待证据。

独立测试：相同场端点、rho rawD=.25，非 static 初始采样使离化概率 .02；static 使用当前场得到 .04。缺 sampler、错误时钟及未显式选择 static 拒绝。

## 2. 观察支路与时间尺度

`NativeFieldStep(... observation_values=...)` 将观察系数送入 encoder 与同场 Gauss 值/梯度路径；`CoupledNativeField` 的闭合特征也从该观察重求。恒等支路仍是 clean U，原始 rawD 不独立加噪，源/shard 始终读取 clean committed U。解码归零时不会把噪声提交成物理场；零时间返回 clean U/rawD/memory。

`NativeObservationNoise` 使用显式低秩原生系数扰动、原生量纲的分组 scale、固定 amplitude 与来源；唯一 DOF 平均并传播每场约束到全部共享 alias。支持 `[4,1,N]` 共用约束和 `[9,4,1,N]` 每场约束。从同一扰动完整系数重新求值，不独立扰动梯度、投影历史或 marker 云。closure descriptor 仅声明由实际联合消费层重算，不冒充另一个已实现物理算子。

噪声 CPU Torch RNG 纳入 session；事件失败恢复 Python/NumPy/CPU Torch RNG 与模型权重/缓冲。本轮恢复证据是 CPU，**未认证 CUDA RNG 或外部 provider 自有 RNG**，生产随机出生尚未实现，不能据此声称其随机性已恢复。

`SourceFeatureScales.anchored_physical_reference` 提供显式 SI 时间原点、gather/event 参考尺度与 provenance，转 JU 一次。网络编码 `(eventSI−originSI)/eventScaleSI`；真实绝对时钟仍用于因果校验。原点绑定模型配置，不随窗口重置。fixture 使用 origin=.0035 s、gather scale=1 µs、event scale=1 ms；.003514 s 的特征为 .014。此为显式物理参考、**非训练拟合统计或两组真实尺度认证**。旧 v1 绝对时间模式仍可追踪；新 v2 anchored schema 与原点进入指纹，不能静默恢复旧语义。真实合法全源输入尚缺，本轮未声称真实完整编码饱和问题已验证解决。

field-step 配置升级为 `v3_rate_memory_observation`；观察策略进入检查点。精确恢复与部分权重初始化不同，旧配置不属于本轮精确恢复支持范围。学习速率的既有时间适配不变；不增加第二次 dt 积分。

## 3. 网络输出边界：仅已核验子集

新增 `NativeDirichletRows`，应用于联合网络增量、共享装配之后。依据 `models/model502/mod_boundary_conditions.f90:155–364`：按相邻局部 vertex 的边方向，固定 value 及对应切向 derivative；对 ψ、Phi、j、omega 执行所检查固定边界分支，code 2/3/12/20/21 或 no_mach1_bc 时加入 rho/Ti/vpar/Te。rho_imp 未擅自套用此分支。`vacuum/vacuum.f90:496–516` 说明 ψ/current 的 free-boundary 判定还依赖 STARWALL harmonic 状态，当前明确拒绝 freeboundary/RMP/simple-types 配置。

这采用理想零更新行，区别于 JOREK 有限 `1e12` penalty；属于登记近似。不是 marker 投影边界，不对全部 cubic 系数做正值裁剪。解析底边测试核对 value+s-derivative、正常/交叉高阶保留及 rho_imp 区别；完整场步测试证明掩码真正作用于学习输出且其余行仍有梯度。

配置明确标 `dirichlet_subset_open_unverified`。**Mach1、自然开放边界及目标完整物理 BC 未验收**，不能据此给真实配置授予模型资格。默认未指定此模块的旧场步继续标 learned_unverified。

## 4. 统一指标和检查点

实际 Adapter 输出完整九场/rawD、marker 五通道及自身时刻；身份/坐标有证据时输出 Cartesian 米 shard 位置，另带提交事件时刻 `shard_time_si`。独立密度源单列，不以第五通道替代。未保留在事件输出中的密度源目前仍 missing，不能当作零。

`coupled_node_metrics`/`recorded_supervision` 复用逐场完整 degree、Gauss 值、兼容梯度、逐通道 Pearson r/相对 L2/绝对误差、固定身份位置/位移/覆盖与逐场 loss。新增防线：缺场标签显式 missing；shard 标签必须有自己的显式时刻且与预测匹配，不因文件编号或场时刻相同而推定。常量相关度、零范数相对误差保持 undefined；绝对误差仍报告。经协调后，实际数据 reader 已在显式身份分支将同一场文件内部时刻写入 `node.object_times_si['shard_positions']`；新增 reader 测试以文件名999999、内部步号7与时间6 s 验证不按文件名配对，不猜身份或存活。指标/NumPy 源不可微项不冒充可优化 loss。

session 保存采用同目录临时文件 + 原子替换；加载期间的权重、完整状态及 RNG 校验在同一回滚范围。新增损坏 RNG 测试先红：原实现先改 Python RNG 再在 Torch RNG 失败，留下部分状态；修正后权重/RNG 全部恢复。只在更新边界保存已 detach 状态，不序列化未完成 autograd 图。

## 5. 小云实际执行证据（不是 phaseA）

产物：`output/native-base-step-model-integration-2026-10-09/mechanism/diagnostic.json` 和 `session.pt`。固定 seed 17，fixture 实际模型 2691 参数，2 次完整事件：1→2 ns、2→3 ns。源/shard 使用小网格参考机制，不是生产 nonlocal/birth/kernel。

只第二步 field loss backward：第一步 values 梯度 L1=12.0881126564；第一步 memory=2.3656399843e-5；初始 memory=2.3626989862e-5；CfC 参数=4.2009857859e-7。新模型中途恢复与连续执行 bitwise 相同。耗时 .2024 s，进程峰值 RSS 559596 KiB（含 Torch 初始化，不是增量内存）。未来标签不进入推进、显式截断、source/field/shard 失败回滚由回归覆盖。

此诊断没有真值标签，统一输出全部 missing，未从同一实现制造真值或报零误差。loss 算术与位置/undefined/time 防线另有独立 fixture oracle；未形成真实预测指标。

## 6. 真实云预检与拒绝原因

`cloud-preflight-final.json` 全槽位分块计数（65536 slots/chunk），预算 1024 MiB：

|锚点|槽位|活粒子|原始 payload|三份状态副本下限|
|---|---:|---:|---:|---:|
|v400 4110 restart|1000000|11168|172000035 B|516000105 B|
|inner 2710 ordinary|1000000|19|172000035 B|516000105 B|
|inner 5680 restart|1000000|52604|172000035 B|516000105 B|

全预检耗时 0.0732 s，峰值 RSS 43988 KiB。只是读取和计数预算，不是总推进内存估计或粒子成本认证。活粒子 count 与认证 manifest 一致，未裁剪、不取消 10000 活粒子参考防线。百万槽位不等于百万活粒子。

真正执行在第一完整事件**之前**拒绝，未发生随机网络支持域失败，也未用真值重置：

- 目标 nonlocal=True，而当前独立 `InspectedNeonShardDynamics` 只支持 pure-Ne/local abl3/rocket6 的 update 切片，nonlocal 明确拒绝；尚无完整生产 birth/density/RNG provider。需从规定 OLD 场构建并验证 nonlocal 诊断/消融，再补生产出生和密度核。
- 流体 Mach1/natural-open BC 尚未完成；Dirichlet 子集不能替代它。
- restart 4110/5680 的零投影属于注册初始化，不能替 ordinary 2710 补历史。后者 clock/residual/projection deltas 等因果证据仍 pending；需因果重放缓存或补采相应状态，不能任意相邻投影差分。
- production dirty 构建与有效配置绑定待证据；`actual_gather` 需从合法重放计算/认证。自主外部下一 dt unavailable，不妨碍**另行声明且有表的** recorded_schedule_replay；两者不混用。
- 两个大活云超参考 pusher 防线，需不改变身份/槽位/RNG语义的有界推进路径及成本证据，当前没有认证。

原 `phaseA-example`/record-interface 是记录契约，不是已绑定真实执行模型；本轮没有手改 eligible 或拿 fixture 给真实锚点授予资格。`cloud-preflight` 是资源/缺口证据，不是新动态资格计算器。完整四轴模型绑定与真实事件验收仍待以上实施，不能报告已完成。

## 7. 可复现命令

环境 `.torch-cuda-env/bin/python`。所有命令无训练、无 optimizer 更新。

```bash
PYTHONPATH=.:tests .torch-cuda-env/bin/python tools/diagnose_base_coupled_execution.py --output-dir output/native-base-step-model-integration-2026-10-09/mechanism
PYTHONPATH=. .torch-cuda-env/bin/python tools/audit_native_execution_resources.py --anchors output/native-closure-certification-2026-10-08/certified/anchors.json --output output/native-base-step-model-integration-2026-10-09/cloud-preflight-final.json --memory-budget-mib 1024
PYTHONPATH=.:tests .torch-cuda-env/bin/python -m pytest -q tests/test_native_state.py tests/test_native_gauss.py tests/test_native_field_step.py tests/test_native_features.py tests/test_native_temporal.py tests/test_native_joint_source.py tests/test_native_context_guard.py tests/test_native_coupling.py tests/test_native_sparse_source.py tests/test_native_projection_n0.py tests/test_native_production_radiation.py tests/test_native_source_time.py tests/test_native_source_time_integration.py tests/test_native_field_boundary.py tests/test_native_execution_features.py tests/test_native_windows.py tests/test_native_conditional_windows.py tests/test_native_checkpoint_control.py tests/test_base_coupled_training.py tests/test_base_coupled_noise.py tests/test_coupled_metrics.py --junitxml=output/native-base-step-model-integration-2026-10-09/regression-final.xml
```

完整相关回归 **211 passed / 35.75 s**（`regression-final.xml`）。其后仅补 shard reader 时刻一行及独立测试，相关训练接口/指标回归 **16 passed / 2.47 s**，见 `shard-clock-final.xml`；没有重跑包含 optimizer 的 curriculum 测试。

补充回归命令：

```bash
PYTHONPATH=.:tests .torch-cuda-env/bin/python -m pytest -q tests/test_base_coupled_training.py tests/test_coupled_metrics.py --junitxml=output/native-base-step-model-integration-2026-10-09/shard-clock-final.xml
```

诊断/预检拒绝覆盖已有产物；复现请换新输出目录/文件。上述机制产物为首次版本（fields 当时为 `{}`），最终代码已将缺失场改为 `status=missing`；第二次机制产物 `mechanism-final` 对应最终指标语义。

## 8. 下一最小动作与预算申请（不执行）

先完成 production provider、必要开放边界、有界 pusher、实际配置/初态/控制绑定与合法两事件，再认证训练资格。当前**不批准或执行任何真实耦合训练**。

达到资格后建议首次申请仅 1 seed、1 个固定预测起点和固定物理预测时长：最多 1 次单步 optimizer 更新作机制 smoke，先做无更新 baseline；burn-in 候选只比较事先指定的 0 与 1 base 步（支撑合法时），各从可追溯初态推进到同一预测起点，不选“最优” burn-in。总预算建议 10 分钟、峰值内存 2 GiB，源参考资源认证若超预算则先停止，不自行放宽。此为后续待确认预算，未执行。

验收仅为小云实际 Adapter 与训练接口的机制一致性部分交付；不是目标真实 phaseA 有限步闭环通过、物理方程精确耦合、预测有效、长期稳定或 C0 完成。
