# 真实耦合事件与消融率残差后续核查（2026-10-09）

> 后续补证：[云分量与投影恢复相对误差](native-cloud-replay-relative-errors-2026-10-09.md)已完成同一真实片段的成对采集。云差异限于P_imp末位，运动/身份/权重一致；发现并复现反馈布局引起的插值归约差异。本文保留当时精确摘要失败及尚未量化的历史记录，最新数值判断以后续报告为准。

本轮范围：继续真实 ψ 教师支路下的完整源—九场—shard 事件，并判断先前 injInner 消融率残差是否可由统计/舍入解释。不训练，不更新优化器，不修改 JOREK 或原始数据。未完成项与实际运行结果分别记录，不能把分项测试拼成真实整体闭环认证。

最终摘要：v400完整百万槽位的两个ordinary事件已实际完成；新模型续跑的完整状态逐位检查未通过，不能宣称恢复验收通过。injInner消融残差不是统计噪声；源码兼容缓存把最大相对差降至约1e-8，但历史进程拓扑证据仍不完整，默认模型未悄悄切换。详见文末实测结果及残差专项。

## 初始可复核失败

主线程实际运行：

```bash
.torch-cuda-env/bin/python -m pytest -q tests/test_native_psi_teacher_records.py -k registered_poststartup --runxfail
```

结果为 `1 failed, 1 passed, 8 deselected in 6.69s`。injInner5681→5682 在已纠正因果 aux 配对后，消融率最大相对差仍为 `0.0004898476160541286`，高于该精度检查的 `1e-6`。此阈值是逐值重建检查，不是用户已同意的统计接受界限。

判断原则：固定真实完整场、投影、ES 与 shard 状态的消融率重算是确定性对照。随机出生云的方差不能自动解释它；不以全云总粒子数的 `1/sqrt(N)` 作为局部消融率误差容限。需分别报告最大/分位相对差、受影响碎片数、总释放量有符号差与绝对加权差，并区分局部/nonlocal 分支。

## 完整源体积诊断成本修复

在完整3986元素、846碎片的真实v400几何上，旧 `NativeShardDensityKernel.refresh_volumes` 用 Python 双层元素/碎片循环：4110只读成本测量32.312秒；独立新增4112验收测试32.514秒，先因15秒的宽松本地资源上限失败。该测试先验证全部846项与 JOREK 原始 `spi_vol_arr` 一致，再断言运行成本。

实现将独立碎片按最多256个批量计算，保留每个碎片的高斯点求和、元素累积顺序和原核公式。没有裁剪、改变核宽度或省略环向因子；每块暂存仅为256×16个高斯点量级。

修改后同4112公开入口实测1.154秒；对原始846项体积最大绝对差 `4.025e-16 m³`，最大非零相对差 `5.064e-15`。这是单个诊断组件的本机成本，不能外推为完整 JOREK 加速比。

```bash
.torch-cuda-env/bin/python -m pytest -q tests/test_native_production_density.py tests/test_native_production_events.py tests/test_native_psi_teacher_chain.py
```

结果 `14 passed in 6.46s`。完整真实846项测试使用 `rtol=3e-11, atol=1e-20` 的已有体积对照口径；未把容差放宽。

主线程随后完成21文件跨模块回归：`132 passed in 20.22s`，JUnit 位于 `output/native-real-chain-followup-2026-10-09/preliminary-regression.xml`。这是体积批量化后的阶段结果；不包含尚在研究的消融率残差最终修复或真实整体事件验收。

### 出生提案的完整密度查询

另一公开入口 `rates_at` 每个随机提案都遍历846个碎片。新增512次完整查询的资源/叠加测试先红：3.358秒超过3秒宽松本地预算。实现改为最多256碎片×32查询点的批量核；按原碎片顺序做包含上一块初值的逐项累加，不使用改变加法顺序的树求和。零半径/零率规则和环向平均不变，出生算法、提案数、随机数流及接受阈值均未修改。

修改后512次846碎片单点查询实测0.0668秒。额外差分检查使用随机分布的846个碎片、65个查询点（跨两类块边界），非平均及离散环向平均两模式与旧标量实现均逐位一致，最大绝对差0。密度、出生、事件、教师支路四文件回归 `22 passed in 6.60s`；原独立Fortran出生接受阈值测试也通过。这是局部核优化，不改变生产RNG并未逐样本复现的声明。

## 实际源输运配置核查

此前冻结场资源探针中的 `diffusion_ju=0` 不能直接沿用为目标生产配置。两目标 namelist 第435–440行指定 `D_perp(1)=5e-6, D_perp(2)=0, D_perp(6)=0`。只读源码 `models/diffusivities.f90:44–70` 的所选解析 profile 因而为常数5e-6；`preset_parameters.f90:582` 默认 `d_perp_file='none'`，`read_num_profiles.f90:126` 决定是否用数值profile，`:830` 默认 `diff_diffusive_flux=.false.`。

`particles/examples/marker_spi_source.f90:509–519` 使用该扩散系数乘密度梯度尺度构造额外速度。真实事件入口必须解析/核验这些条件并显式传入5e-6，不能默默使用资源探针的零扩散。教师 ψ 仍只介入已声明的诊断/shard支路，不覆盖提交的九场、marker场向速度或其原生梯度。

## 方法

使用 diagnosing-bugs 先建立已运行的确定性红反馈，再检验原因；使用 tdd 在已有公开源诊断/完整事件接口先做失败检查，再实施最小修改。残差专项和真实事件最终运行证据在完成后追加。

## 消融率残差：不是统计噪声，已找到源码兼容解释

详见 [专项报告](native-inner-nonlocal-residual-2026-10-09.md)。主线程独立复跑新增 CLI/Fortran 对照测试：`3 passed in 23.09s`，JUnit `output/native-real-chain-followup-2026-10-09/independent-nonlocal-oracle.xml`。固定真实输入的两次重复逐位一致，没有随机调用。

|injInner对照|原最大相对差|原总率有符号相对差|源码兼容缓存后最大相对差|
|---|---:|---:|---:|
|5681→5682|4.898e−4|+1.422e−5|9.723e−9|
|5682→5683|8.646e−3|−1.300e−3|4.817e−9|

后一步708/846碎片受影响，局部消融主导的16个碎片原本就约为舍入级差异。不是由极小分母的几个点造成，也不能用出生方差解释。

主线程直接核对 `core/mod_jorek_timestepping.f90:353,444–469`：求解前所有rank刷新OLD ES，提交后只有rank0刷新新ES；后续所有rank参与int3d。另直接核对 `diagnostics/mod_integrals3D.f90:671–686`：`use_marker` 与 `present(aux_node_list)` 两分支均累加同一个aux。由于密度下限是非线性的，电荷/密度均乘2在小密度区域并不相消。独立Fortran缓存实现这些顺序后匹配真实rate，未调整 Parks 常数或时间容差。

重要证据界限：兼容诊断采用10-rank分片；同目录现存10-rank日志来自后续重启，不是5680的精确生产构建证明。结论是“一手源码机制与原始输出共同支持的兼容重建”，不能宣称已认证当时二进制及进程拓扑。

没有默默将 rank滞后ES/重复aux写入默认模型。现有真实链仍使用统一当前ES、单aux重建，builder将这一差异写入 `shard.configuration['historical_nonlocal_compatibility']` 和近似清单。原模型精度xfail保留；兼容oracle通过不等于默认模型已修复。是否将历史实现行为作为显式兼容模式，需与统一诊断的模型策略区分后决定。

## 真实初态构建

新增 `tools/native_real_chain_builder.py::build_real` 复用既有组件。主线程已实际构建v400 poststartup初态：U4111/rawD/shards、part4110全百万槽位/11168活粒子及原始零P4110；没有重新执行startup或把它计作ordinary事件。初始完整场只导入一次，后续4112/4113仅读取允许的ψ/ES教师及记录步长，其它八场不作为推进输入重置。

投影几何使用初始导入U4110的ψ构造冻结滤波算子；密度使用独立未滤波投影。初始原始字段时间、逻辑startup区间及文件SHA均保留。约8秒构建完成，峰值约1.14GB，首个实际gather为40粒子子步。九场/CfC联合网络3147参数，固定非零权重、无优化器；n_spi、pellet_density、eta分组输入以及真实边界参数参与现有学习接口。缩放为显式参考量，不冒称拟合训练统计。

单rank全局出生分布不是生产MPI槽位/逐rank取整重放，MT19937不是PCG逐粒子复现；开放边界响应仍是学习输入而非硬方程求解。此构建通过只证明完整组件绑定，实际两ordinary事件结果另列。

## 首个真实窗口的资源失败及等价优化

v400初始窗口的实际源率 `1.0107751477983678e24 atoms/s`、gather `2.0428876129759165e-7 s`，对应29个新marker。代理第一次20万提案运行耗时229.95秒后拒绝预算耗尽，0事件提交；主线程复现于 `output/native-real-chain-followup-2026-10-09/v400-run1/diagnostic.json`，228.77秒、峰值1.72GB、同样0提交。此前5ns出生探针不能替代这一真实窗口。

`NativeProductionBirths.births` 原来先对所有提案做单元定位，再判密度拒绝。两条件是独立的接受谓词。现仅把密度判据移到定位之前：每次仍抽同一7个uniform，提案/拒绝轮次/数目和域内接受集合保持不变，不改密度、R_geo、权重或采样窗口。

公开出生入口的固定旧流回归先红：8次耗时5.476秒超过3秒宽松预算。优化后位置/权重逐位匹配旧结果，完整MT19937状态digest仍为 `d8e03e59afdf37d00aab68cfcc3ea8b8dcd95a680664c25c1ef2ca16ba188e9a`；密度/出生/事件/教师24项通过，9.75秒。独立Fortran接受阈值测试继续通过。

重跑路径 `v400-run2` 使用同一seed20261009、完整百万槽、原始40/4子步，900秒/8GiB外部上限；只将提案上限提高为100万。不以截断云、缩dt或提高接受概率换取通过。工具还加入公开调用阶段日志，能区分出生、marker推进/投影、九场、OLD shard及诊断刷新。

## 本轮综合回归

最终26文件无训练回归 `161 passed, 1 xfailed in 102.18s`，JUnit：`output/native-real-chain-followup-2026-10-09/final-regression.xml`。包含上轮24文件集合及 `test_native_nonlocal_residual.py`、`test_native_real_chain.py`。没有curriculum/optimizer测试；真实构建专项只验构建阶段，不将有界中止计为两个完整事件。

保留的strict xfail针对默认统一ES/单aux路径的历史rate逐值检查；新增独立Fortran兼容模式测试通过不能替代默认行为的这个检查。真实runner的解析fixture验证了完整状态/RNG恢复及跨步反传，真实执行本轮不请求BPTT。模块编译及 `git diff --check` 通过。

## 真实两个事件已执行，完整恢复验收仍失败

实际命令（输出目录必须尚不存在）：

```bash
.torch-cuda-env/bin/python -m tools.diagnose_native_real_chain \
  --run data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400 --anchor 4110 \
  --output-dir output/native-real-chain-followup-2026-10-09/v400-run2 \
  --wall-seconds 900 --memory-mib 8192 --birth-max-proposals 1000000 --chunk-size 256
```

`v400-run2/diagnostic.json` 保留 `success:false`、`full_coupled_events:2`，失败阶段是 `checkpoint_exact_comparison`，而非源出生或事件推进。总耗时611.716秒，峰值RSS 2,781,052,928字节。没有云裁剪、缩短步长、优化器更新或训练。

|普通事件|新marker|活marker|粒子子步|实际粒子更新次数|
|---|---:|---:|---:|---:|
|4112|29|11197|40|447880|
|4113|2|11199|4|44796|

每步实际执行独立密度核、出生、输运与完整滤波投影、含源九场联合网络、OLD场rocket6/shard更新及非局部/体积诊断刷新。九场增量全部非零；shard位置、半径和消融率均改变。这证明真实尺寸下的机制接线可执行，不证明未训练网络预测准确、统计出生与生产逐样本相同，或自主ψ闭环成立。injInner本轮完成真实场残差审计，没有执行其完整云的两个整体事件。

首事件检查点245,024,810字节。保存前与独立读回的全状态摘要一致；新模型载入权重与原模型一致；第二事件后Python/NumPy/Torch CPU RNG均逐位一致。但完整状态摘要不同，保持失败，未放宽门槛。

新增 `tools/replay_native_real_checkpoint.py` 只从原首事件检查点重放第二个4子步窗口，避免再次运行40子步。独立重放得到与原runner恢复支路相同的全状态摘要 `087eb87b3f34b99e361837419788203c86bab8f88ceae78723f904121ce8f366`，并非不稳定的随机重放。实测命令：

```bash
timeout 180 .torch-cuda-env/bin/python -m tools.replay_native_real_checkpoint \
  --directory output/native-real-chain-followup-2026-10-09/v400-run2 \
  --output output/native-real-chain-followup-2026-10-09/v400-second-replay-differences.json
```

相对原连续第二事件：

|比较项|逐位一致|最大绝对差|
|---|---|---:|
|九场值|是|0|
|shard位置/半径/消融率|是|0|
|原始历史增量|否，30项|3.309e-24|
|CfC记忆|否，1849项|8.674e-18|
|五通道投影|否，47667项|4.541e-10|
|投影历史增量|否，47663项|4.541e-10|

完整云及shard含缓存摘要也不同。投影数组最大幅值约9689.27，最大绝对差/全局最大幅值约4.69e-14；这不是逐点相对误差，也不能替代各通道及云分量的误差核查。当前证据只限定了差异范围，尚不能断定为稀疏重建/内存布局舍入，更不能据此宣称恢复通过。诊断脚本成功退出表示完成差异报告，**不是恢复验收成功**。

下一最小工作：从同一首事件状态对照原实例与新实例，在源采样、云输运和投影三个边界记录逐分量差异，定位首个分歧；保持精确恢复检查不变。另须区分“历史JOREK兼容诊断”与“统一当前ES诊断”的模型策略，取得历史进程拓扑证据后再决定是否增加显式兼容模式。暂不训练，也不以v400局部主导的结果替代injInner非局部闭环验证。

最终补查：runner fixture、出生与完整密度测试 `17 passed, 2 deselected in 7.60s`；新增/修改模块编译通过，`git diff --check` 通过。该17项与上述161项有重叠，不相加。
