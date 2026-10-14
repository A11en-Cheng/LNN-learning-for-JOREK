# 非零九场完整生产机制链执行（agent A，2026-10-09）

## 结论与边界

已执行实际 `TorchRecomputedSourceEvents` 的两/三完整 base 事件：自生成 marker 源 → 含双源九场联合网络 → 新位置 OLD 场 shard 更新 → 新场诊断刷新 → 原子返回。九场提交每步全部实际非零，不是 zero decoder 充当预测；未训练、未调用 optimizer、未搜索参数，也未进入冻结容量参考。

已新增独立的解析 psi-teacher 模式：使用主线程的 `NativePsiTeacherShardModel` 与 `RecordedPsiFeedback`，仅诊断/shard 分支替换 psi；提交九场、rawD、marker 推进及出生速度仍消费预测九场。每个事件保存 branch intervention、预测/诊断场 digest 和各自时刻。这个模式明确 **不是自主 AR**。

本报告和三个执行产物均为 **解析小网格/8 槽 fixture**，不是两个实际 phaseA run 的完整生产闭环。没有使用真实数据、没有人为授予 qualification，也没有等待或实现自主 ES finder。真实记录读取接口已由 B 提供；真实数据完整事件接入仍由主线程协调。

## 写集及设计

仅新增：

- `tools/diagnose_native_production_chain.py`：不导入 `tests`。独立构造完整 native geometry、nine-field/CfC/stencil、source、生产出生/密度/nonlocal/shard provider。`execute_chain(output_dir, events=2|3, builder=..., scope=...)` 是完整事件/checkpoint seam 的小接口；builder 可被主线程的显式真实记录初始化替换，不扩张成物理求解器。
- `tests/test_native_chain_execution.py`：五个专项测试，沿已确认完整事件/checkpoint seam 逐个红绿。
- 本报告。

`native_coupling.py`、`base_coupled.py`、psi teacher 模块与其他人的测试均未编辑。codebase-design 使 fixture、事件推进及证据生成集中在工具接口，TDD 每次只新增一个机制切片，没有重测/重写原出生和密度公式。

## 一手时序依据

只读 JOREK HEAD `54c675bcc39c33b082c0c35046e28122dc53b3aa`（本轮重新核对）；路径根 `/home/ac_desktop/JOREK_code/jorek`。

- `core/mod_jorek_timestepping.f90:393`：九场候选求解；`:414` 首 nonlocal 表缺失时用 OLD 场初始化；`:418` OLD 场 shard 更新；`:424–426` 提交场、rawD并推进时间；`:446` 新场平衡诊断；`:510` 安装 aux 后 nonlocal/volume 诊断刷新。
- `models/pellet_module.f90:270–315`：旧速度移动、旧 rate 侵蚀，后续才用新位置 OLD 场计算新 rate；原顺序沿用现有已验证机制。
- Python 对应现有公共 helpers `src/models/native_coupling.py::advance_source_event` 与 `refresh_coupled_diagnostics`；Torch 在返回之前的现有 rollback try 内执行刷新，未另起 driver 外 refresh。

所检查源码不等同生产日志的 `v2.25.12.0-1042-gb4165893f-dirty` 构建；本轮没有声称生产构建一致性。

## 确定性有界 fixture，不是学得准确度

矩形 R=[2,3]、Z=[0,1]、完整 cubic/native `[9,4,1,4]`、4 节点、8 marker 槽、一个纯 Ne shard。初始 psi=R−2、Te=2000 eV、rho=1.25；保存 rawD 的 vpar=.0002，而当前 vpar=.001。source 使用非 static 的真实 saved-history sampler，不从端点差重建 rawD。

所有随机初始化参数均以固定解析数值替换：一般参数为 `.02*sin(.17*i+.31*ordinal)`；九场 decoder weight 全为 `1e-4`、36 项 bias 为 `[2e-4,4e-4]` 的固定线性序列。该单一规定没有参数搜索。隐记忆从 .1 起，官方 CfC 与时间一致外层积分照常运行。

特征缩放为显式 fixture 单位参考，不拟合任何训练数据。现有 schema 的 `training_support` 字段明确登记 `analytic_fixture_unit_reference_NOT_training`；scenario/physical-eta 参数组在此 fixture 明确关闭，不伪造真实 namelist 输入。

与原机制 fixture 相同的成本设置：source width min=1m、birth proposal R_geo=1e8、固定 SCD=1e−12 的解析 atomic provider、n_flux=2/n_table=10。它们不是生产 ADAS 或两个 run 的物理参数；真实 `.05/.01` radius-min 差异未被改写。磁减弱/rocket 不在此 fixture 启用，其他机制覆盖仍以既有独立 oracle 为准。

工具逐事件检查九个实际输出增量均有限且非零，zero decoder 明确失败且不写成功 diagnostic.json。专项测试还检查九场每步变化小于1e−4（规定 fixture 的执行边界，不是预测准确度 gate）。没有真实标签、真值 reset 或完整 JOREK 矩阵移植。

## 实际机器产物

根目录 `output/native-nine-field-chain-2026-10-09/`，每个子目录均含 `diagnostic.json`、`trajectory.npz`、`bptt.npz`、`session.pt`；输出目录必须新建，禁止覆盖旧证据。

|子目录|完整事件|各事件九场 max-abs 变化范围|活 marker|新模型完整恢复|
|---|---:|---|---|---|
|`analytic-two`|2|3.63395e−7…5.87286e−7|1→2|逐位一致|
|`analytic-three`|3|3.63277e−7…5.87286e−7|1→2→3|逐位一致|
|`teacher-three`|3|3.63277e−7…5.87286e−7|1→2→3|逐位一致|

三事件 shard radius：`.0029999997943734587 → .002999999509047876 → .0029999992236962856` m；新 rate：`1.387591026033248e24 → 1.3877172422096496e24 → 1.3878434369855891e24` atoms/s。给定初速度 .2m/s，shard R 也逐步增加。每次出生权重、slot lifetime、完整 P_imp、V_prev、feedback/rawDP、field/rawD/memory、shard 状态均有轨迹产物。

teacher fixture 的 psi 记录就是固定初始解析 psi，各时刻显式登记，不读取真实文件或未来邻点。预测 psi 随九场网络变化；每步预测场 digest != 诊断场 digest，诊断 psi 来自同刻 record，其余八场保持预测。由于变化有界且源 fixture 空间温密近均匀，两模式上述聚合数值相同不表示 teacher 无接通；不同 branch binding、intervention 及完整状态 digest 提供了分支证据。

## 恢复与 BPTT

保存第一完整事件及 **该事件自己的 RNG**，不是窗口结束后的 RNG。独立调用 builder 新建 model/source/shard/provider，并故意改变新模型 decoder bias，证明加载前权重不同；`load_session` 覆盖为 checkpoint 权重后运行余下事件。比较包括完整 dataclass 状态、全部 marker 槽历史、反馈/rawDP、场/rawD/CfC memory、诊断缓存、provider binding、intervention、时钟和控制。Python/NumPy/Torch CPU RNG 全状态逐位一致。

`analytic-three` 直接/恢复完整状态 digest 同为 `3ae6f2777f992ffe120a375d3a0e56e41ec9b052ad9ff221eb27f6f7df24f106`；teacher 模式同为 `5cd01e8e56724bb2754e7b8cd0c9cce50045df05a9889d698bfa3b49b0cbfccc`。

BPTT 仅对最后已提交九场平方和做 backward，不读标签，也不附加 memory loss 来掩盖场—记忆断链。三事件梯度 L1：初始 rawD `2.858258766365202e−12`；初始 memory `4.17974523599417e−5`；第一事件 memory `4.184862569475562e−5`；CfC 参数 `1.237736246004019e−7`；decoder 参数 `.09417359085137085`。初始/第一事件 values、rawD、memory 均有有限非零梯度。梯度大小不是性能门槛；模型参数/buffer digest 在执行和 backward 后不变。

NumPy 源/shard 继续 stop-gradient；保存/加载断开既有历史图是 checkpoint 语义，精确恢复检验的是数值与状态，BPTT 在未截断的连续路径另行验证。未验证 CUDA RNG，也不把 MT19937 生产分布流等价于生产 PCG32/MPI 逐粒子重现。

## 红绿、实际测试与复跑

逐切片：工具缺失红→三非零九场事件绿；checkpoint 字段缺失红→新模型完整恢复绿；BPTT 字段缺失红→仅最后九场 loss 的跨事件梯度绿；teacher builder 缺失红→同刻分支干预绿；zero decoder 未显式拒绝红→九场实际输出 fail-closed 绿。

专项 5 passed。关联六文件最终回归 **39 passed / 6.38s**：

```sh
.torch-cuda-env/bin/python -m pytest -q \
  tests/test_native_chain_execution.py \
  tests/test_native_production_torch_integration.py \
  tests/test_native_psi_teacher_chain.py \
  tests/test_native_production_events.py \
  tests/test_base_coupled_training.py \
  tests/test_native_source_time_integration.py \
  --junitxml=output/native-nine-field-chain-2026-10-09/regression.xml
```

只运行无 optimizer 的测试，没有运行 curriculum 训练测试。工具独立运行（目录须不存在）：

```sh
.torch-cuda-env/bin/python -m tools.diagnose_native_production_chain \
  --events 3 --output-dir /tmp/native-chain-new-artifact
.torch-cuda-env/bin/python -m tools.diagnose_native_production_chain \
  --events 3 --psi-feedback teacher-fixture --output-dir /tmp/native-teacher-chain-new-artifact
```

## 未完成项与接力

主线程可用 `builder(events)->(model,source,shard,runtime,equilibrium_provider)` 注入真实同刻记录初始化；同时显式传递真实/conditional scope。teacher wrapper 的 configuration 与 model/native_context 必须同步，初态缓存须由相同 teacher 分支初始化。当前 CLI 两模式只有解析 fixture，不允许把该产物改名成真实验证。

尚未由本写集验证：真实 .05/.01 两 run 的完整九场/marker/shard teacher 事件、生产 ADAS 输入与全部有效配置、真实 rollout 内部所需每个 base 时刻的 psi record 齐备性、实际 nonlocal 表对照、实际全云完整耦合事件及其资源边界。缺同刻输入应明确失败，不用邻点/未来记录补齐。自主 ES finder 不再作为本轮接线的前置，但 teacher 模式不能据此冒称自主闭环或预测物理已学会。
