# 九场工作链与真实 ψ 回授：两条验证路径（2026-10-09）

用户本轮要求先执行既有九场预测工作链，再暂用真实 ψ 回授诊断依赖，以真实记录验证 marker 生成及 shard 状态变化。本轮不训练、不更新 optimizer、不修改 JOREK/原始数据。使用 codebase-design/tdd；验收仍在源推进、完整事件、检查点这些已确认的 seam。

## 两项任务不混用结论

1. **完整执行机制**：非零九场网络输出 → 提交 → 下一事件源/非局域/shard 消费 → 续跑；小解析域使用明确的解析诊断依赖，验证执行、图与状态，不宣称预测精度或真实网格资格。
2. **真实 ψ 教师回授**：只对诊断/shard 分支介入同刻完整 cubic ψ，配用该记录的同刻轴/X 点/边界诊断，临时绕过自主 ES 搜索；验证真实源机制，不称为自主闭环。

真实 ψ 不等于真实九场整体替换。网络输出仍保存在 `fields_current`，rawD/学习记忆不被教师改写。教师分支的其它八场保持该次预测值；marker 输运及出生初速度仍消费已提交预测场。这一分支范围为明确选择，不声称已经测试“所有磁场消费者均用真值 ψ”。初始真实九场用于单步源/shard 机制对照不等于后续反复重置八场。

## 公共 Interface

- `src/models/native_psi_teacher.py::RecordedPsiFeedback`：读取已显式加载的同刻 record，校验完整 ψ、单位、几何、ES 时间与来源。时间匹配容差最多 1e−14 s，仅处理浮点对时；没有最近帧、未来帧或插值回退。歧义重复记录拒绝。
- `NativePsiTeacherShardModel`：沿现有生产 shard Interface，仅为 `update` 和 `refresh_diagnostics` 构造 ψ 替换支路。nonlocal 的 ψ 壳、局部 ψ 查表与 ES 绑定在同一支路，不能把真实 ES 直接绑定到不同预测 ψ。
- `NativeProductionShardModel.diagnostic_fields`：默认返回预测场副本，教师 Adapter 才覆盖这一策略。
- 完整事件先计算/提交私有网络候选，然后按指定策略刷新诊断，最终原子返回；缺教师记录等后段失败仍恢复 CPU RNG 与输入状态。
- `psi_teacher_event`、`diagnostic_field_binding` 分别记录介入来源、网络场与诊断场 digest。源/shard配置、ES provider配置及全部教师 record fingerprint 进入检查点兼容校验。预测输出显式带 `autonomous_prediction=false`。

`RecordedPsiFeedback` 使用真实 ψ 对应的**记录 ES**，不是新实现了磁轴/X 点搜索。自主路径仍须后续实现 ES finder；本轮不再将其作为教师回授机制实验的前置条件。

## 测试范围

专项已覆盖：九场均发生非零更新而未被教师覆盖；诊断 ψ 与网络 ψ 的 digest 区别；另外八场不变；同刻记录缺失时拒绝未来替代并回滚；两事件新模型恢复；Torch 跨步 BPTT；网格/单位和教师 provider 配置检查。预期来自独立解析 fixture；没有 optimizer 更新。

数据读取、真实 shard 对照与出生有界结果见下文及同轮代理报告。本文件不以少量接口测试代替真实物理数据对照。

## 完整非零九场工作链：已实际执行

新增独立工具 `tools/diagnose_native_production_chain.py`，不从 tests 导入 fixture。明确的小解析域、处方原子率、固定非零网络权重，只隔离机制，不当作 phaseA 参数。九场每一步均产生非零原生增量，完整执行出生、云推进、投影、joint_learning 场更新、OLD 场 shard 更新和诊断刷新。

`output/native-nine-field-chain-2026-10-09/analytic-two`、`analytic-three` 分别执行 2/3 次完整事件；`teacher-three` 另执行 3 次显式 ψ 教师事件（这里教师记录也是解析 fixture，不冒称真实数据）。3 步云由 0→1→2→3 活 marker，半径持续侵蚀，位置与烧蚀率实际更新。第一步后在新构造且故意不同初权重的模型加载检查点，最终完整状态与 Python/NumPy/Torch CPU RNG 逐位相同。

末步值平方和反传到初态与第一步的 fields/rawD/memory；源/shard/诊断仍停止梯度。所有运行 `model_weights_unchanged=true`、`optimizer_updates=0`。输出包含 `trajectory.npz`、`session.pt`、`bptt.npz` 和逐事件 `diagnostic.json`；不要求未训练网络准确，不用零 decoder 冒充九场推进。

教师支路专项还只读两组真实 4110/5680 完整 ψ，验证它们在真实网格上仅替换诊断场的第0场，其它八场与网络状态保持原值，单位/几何和混合场 ES digest 匹配。该测试只是实际数据接线，不是完整真实事件。

## 真实数据源机制：已执行及尚需区分的范围

详见 [原始 ψ/shard 核查](native-psi-teacher-records-2026-10-09.md)。两组全 **846 shard** 的重启边机制对照均实际执行：R/Z/phi 各坐标最大误差均为零；半径最大绝对误差 0 / 1.06e−22 m；速度最大绝对误差约 2.36e−10 / 8.80e−11 m/s；消融率对非零真值最大相对差约 1.60e−8 / 3.31e−9。这是有显式下一控制的 shard 机制对照，不把重复输出时间当零 dt 推进，也不冒充普通连续完整事件。

真实原始 cloud 保留全部百万槽位，两组出生探针实际各生成并插入一个 marker：**11168→11169、52604→52605**。使用真实全846 shard率/位置/半径/数值体积和生产 R_geo、注入量、包权重，不为接受率调整这些参数。新生为11态中性Ne、初速度有限；计数/权重遵循取整及最低包阈值。此为显式 **5 ns 出生机制探针**，非原始 imported gather/base 回放，且 MT19937 分布复现不等于生产 PCG32 逐粒子复现。

原审计另保留了 inner5681→5682 消融率偏差和 v4004111 的 projection 超前1ULP拒绝。主线程检查到旧审计以同编号投影重建 nonlocal 缓存，与 int3d 当时已安装的 aux 可能错配。实读原始 `projections004110`/`005680` 五通道全零，而 `004111`/`005681` 非零；因此不先改公式或放宽时间检查，改用源码事件顺序支持的旧反馈重新验证。后续结果单列，不覆盖旧失败证据。

原始 JSON 已归档到 `output/native-psi-feedback-integration-2026-10-09/real-records/`；v400 成功出生使用 `native-B-psi-v400-4110-v3.json`，最终单位修订的短重跑 `native-B-psi-v400-units-final.json` 故意只有1次提案预算并明确失败，不能拿后者声称出生成功。

## 主线程回归与命令

本轮主线程独立回归 **147 passed / 19.02s**，JUnit：`output/native-psi-feedback-integration-2026-10-09/regression.xml`。它包含教师接线、真实record reader、非零九场链及已有源/场/时间/边界/检查点/噪声/指标共24个测试文件；无 curriculum/optimizer 测试。

缓存重配审计和拒绝用例补齐后的最终同范围回归：**150 passed，1 strict xfailed / 28.89s**，JUnit：`output/native-psi-feedback-integration-2026-10-09/regression-final.xml`。该预期失败明确保留 inner 真实消融率精度未达 1e−6 的缺口，不计为通过。新增模块/工具编译检查与 `git diff --check` 通过。

```bash
.torch-cuda-env/bin/python -m pytest -q tests/test_native_psi_teacher_chain.py tests/test_native_psi_teacher_records.py tests/test_native_chain_execution.py
.torch-cuda-env/bin/python -m tools.diagnose_native_production_chain --events 3 --output-dir /tmp/new-nine-field-chain
.torch-cuda-env/bin/python -m tools.diagnose_native_production_chain --events 3 --psi-feedback teacher-fixture --output-dir /tmp/new-psi-teacher-chain
```

完整回归文件集合等于 `native-coupling-coordination-2026-10-09.md` 的21文件集合，加上本节第一条命令的3文件。输出路径必须是新路径，不覆盖证据。

## 最终因果缓存核查

只读源码与原始投影支持：在这些事件中，`int3d(U_k)` 的缓存用当时已安装的 `P_(k−1)`，不是后续源事件才生成的同编号 `P_k`。具体行号及四次命令见配套记录报告的“因果aux重配补充”。这是事件语义配对，不把缺失投影补零；startup 使用登记过且实际读取为零的原始文件。

|真实846 shard普通步|正确缓存投影|消融率最大相对差（非零真值）|
|---|---|---:|
|v400 4111→4112|原始 P4110，零|1.466e−8|
|v400 4112→4113|原始 P4111，非零|1.479e−8|
|injInner 5681→5682|原始 P5680，零|4.898e−4|
|injInner 5682→5683|原始 P5681，非零|8.646e−3|

四次位置均实际变化，R/Z/phi 各自对真实下一记录的误差为零。先前 v400 的1ULP时间拒绝随正确对象配对自然消失，未放宽容差；inner 主要误差缩小，但剩余消融率偏差尚未定位，不宣称逐值复现。以上四次使用 `--shards-only`，不冒充重复执行出生。新 JSON 与旧失败证据一并保存于本报告的 `real-records/` 目录。

## 交付界限与下一步

本轮已完成：非零九场学习器完整小域链、显式真实ψ诊断/shard教师接口、真实两组出生机制探针及全846 shard真实步对照。**未完成真实网格、完整云、真实base/gather时长下连续两次整体事件**；真实机制核查与解析完整链不能拼接成这一验收。生产随机数逐样本复现也未完成。

下一步仍不训练：将已验证的真实ψ教师接口、完整云有界推进和真实事件控制合并为两次真实整体事件，记录每段输入/输出时间与投影版本；同时定向定位inner消融率残差。训练资格与自主ψ诊断不由本轮结果自动获得。

本轮使用 codebase-design 固定诊断场与提交场之间的窄接口，使用 tdd 先验证缺失记录/配置误配失败，再接通完整事件、续跑及跨步梯度。
