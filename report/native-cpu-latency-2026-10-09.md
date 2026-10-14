# 完整耦合事件 CPU 延迟优化（2026-10-09）

## 范围

用户授权降低 CPU 延迟、减少 GPU 空载，并委托子代理实施，主代理负责证据复核与任务协调。
本轮不训练、不更新优化器、不裁剪云、不修改物理步长、JOREK 或原始数据。
固定两个已用 train 初态（v400 4110、injInner 5680），每个案例执行两个完整自主事件。
预测段不读取真实场回授或监督标签；记录步长回放仍不是自主控制预测。
真实完整云、rawD、投影历史、OLD 场 shard 更新及状态提交顺序不变。

使用 diagnosing-bugs skill：先真实计时，再写操作次数红测试，优化后复验完整状态和时间。
真实全云必须按分钟计时，因此快速红绿回路用小型确定性 fixture；全云用于最终验收。

## 分工与实验约束

- Ramanujan：独立计时入口、真实基线/复测、出生密度核事件内预处理。
- Franklin：nonlocal 查询的重复绑定/摘要校验。
- Plato：Newton 定位的纯几何求值，去掉未消费的虚拟场求值。
- 主代理：审阅热点与数值契约、放行修改、统一比较结果、报告。

三个生产修改范围不重叠，真实计时串行执行，防止 CPU/内存争用污染比较。
每次运行使用独立输出目录；有 profiler 的数据用于定位热点，**不与无 profiler 数据计算加速比**。
运行中包含 CUDA 场前向，但不做反向传播或参数更新。

## 一手热点证据

目录 `output/native-cpu-latency-2026-10-09/baseline-{v400,inner}/chain/`，
逐事件 `.prof` 及 `event-*-profile.json` 可复核。下面为首事件的累计时间，父子项重叠，不能相加：

| 路径 | v400 | injInner |
|---|---:|---:|
| 完整事件 | 86.60 s | 98.46 s |
| births | 45.61 s | 非主要热点 |
| density `rates_at` | 45.90 s / 258790 调用 | 非主要热点 |
| `locate_batch` | 8.06 s | 45.31 s |
| `nonlocal.sample` | 16.54 s / 844 调用 | 14.16 s / 724 调用 |
| 全链 `configuration_digest` | 19.77 s / 1733 调用 | 17.41 s / 1493 调用 |

三项待验证的解释是：事件内不变数据反复准备；定位/查询执行了未使用的工作；
扣除重复工作后真实粒子计算才是 GPU 化候选。不按单个 run 的热点泛化两个案例。

## 对照中的模型复现问题

最初 v400 profiled 与 no-profile 构建的初始物理组件一致、RNG 后态一致，
但模型指纹和后续 field memory/rawD 不一致，不能当作严格数值一致性对照。
独立构建检查将初始化漂移定位到 stencil 的 `term_lambda` 缓冲区：谱估计 `eigsh`
没有指定 `v0`。本轮不改变生产谱估计算法，而是保存/恢复该缓冲区在内的完整模型。
保留这些产物，不放松误差判据。
计时工具增加 `initial-model.pt`，保存完整参数、缓冲区、配置、RNG，并绑定初态几何/场/历史/控制。
复测显式载入对应案例的该快照；仅设同一 seed 不足以替代此绑定。
完整组件比较同时检查初态、每事件云数组、场/历史/记忆、反馈、shard 数组、时钟和 RNG。

## 工具

- `tools/profile_native_coupled_latency.py`：两事件有界计时，`--no-profile` 用于墙钟比较，
  `--model-state` 用于严格重放实际模型与 RNG。CUDA 不可见时明确失败，不回退 CPU。
- `tools/report_native_cpu_latency.py`：拒绝不同 seed、控制、硬件环境、初态、模型指纹或 profiler 模式的比较；
  加速与组件逐位一致分别报告。不把一次成对测量解释成统计置信区间。

## 已实施的优化

1. `NativeShardDensityKernel.prepare_rates`：按本次事件准备 live shard 块、宽度平方、
   体积/源强系数和环向平均，复制为只读数组，不保留跨事件缓存。
   births 在拒绝采样循环外只准备一次；独立密度投影也复用本次准备结果。
   逐 shard 累加顺序、七随机数抽样协议和 float32 接受/权重舍入不变。
   公共 `rates_at` 仍校验输入，注入前零源与缺失/未来体积诊断的原语义不变。
2. `NativeNonlocalDiagnostic`：在一次同步 shard 更新内惰性绑定 OLD 场与网格，
   只做一次两个完整摘要；不改变早退、域外、死亡 shard 的行为。
   独立公共 `sample` 每次仍验证，绑定不跨事件、不接受新的未校验字段。
3. `NativeSourceMesh`：Newton 查询仅计算坐标/Jacobian，不再创建并插值零的虚拟场。
   几何浮点算式、候选顺序、迭代上限、阻尼试探、边界判断和容差不变。

本轮没有增加 CPU worker、跨窗口流水或 GPU 粒子后端；先消除有证据的重复开销。
真实复测完成前，不以操作次数下降宣称完整事件已加速。

## 同快照、无 profiler 的完整事件计时

| 案例 | 事件1 前→后 | 事件2 前→后 | 两事件 前→后 | 加速比 | 延迟减少 |
|---|---:|---:|---:|---:|---:|
| v400 | 56.085→30.654 s | 14.688→6.360 s | 70.773→37.014 s | 1.912× | 47.70% |
| injInner | 77.652→66.290 s | 23.970→16.616 s | 101.622→82.906 s | 1.226× | 18.42% |

以上为各一次成对测量，包含完整 CPU 物理事件和 CUDA 场步，不含构建/初态读取，
不是统计置信区间，也不是完整训练吞吐；未证明 GPU 利用率已提高到某个水平。
包含构建等开销的 probe 墙钟分别为 v400 80.475→46.858 s、injInner 111.638→92.809 s。
参考目录分别为 `baseline-v400-bound-no-profile`、`baseline-inner-no-profile`，
复测为 `after-v400-no-profile`、`after-inner-no-profile`。

机器比较：[v400](../output/native-cpu-latency-2026-10-09/comparison-v400.json)、
[injInner](../output/native-cpu-latency-2026-10-09/comparison-inner.json)。
两组初态、模型指纹匹配；两事件的云、反馈投影、shard 数组、时钟与 RNG 逐位一致。
**场/rawD/记忆摘要不逐位一致，完整状态精确比较仍返回失败，不能隐藏此结果。**
独立密度源是场前临时输入，不包含在提交后的 shard 数组摘要中；
另用真实 Gauss/投影对照和固定场输入 CUDA 重复前向检查此差异。

## 数值隔离检查

### 独立密度输入

[density-input-check-v2.json](../output/native-cpu-latency-2026-10-09/density-input-check-v2.json)
覆盖两个真实初态，每组 846 shards、3986 元素、63776 Gauss 点。
独立旧逐 shard 公式与 prepared 路径的 rates、weak load、投影系数、公开 density 输出
全部逐位一致，relative L2 为 0；两次独立构建的网格矩阵、分解及投影输出也一致。
此检查不推进云或事件，不使用 GPU，不代表任意未来演化态已逐位验收。
初版探针因缺少 `spi_num_vol` 键失败，保留原 JSON；v2 遵循真实 builder 已声明的配置取值。

### CUDA 场归约重复性

冻结相同原生场、rawD、记忆、密度输入与零初始反馈，分别对同模型和加载同一完整
state_dict 的新模型各做 4 次前向；不推进物理云、不训练。

| 相对 L2 最大差异 | v400 | injInner |
|---|---:|---:|
| 默认模式九场值（逐场） | 3.49e-26 | 0 |
| 默认模式 rawD（逐场） | 3.97e-17 | 4.37e-17 |
| 默认模式记忆 | 2.80e-17 | 3.41e-17 |
| 确定性模式 | 8/8 次逐位一致 | 8/8 次逐位一致 |

输入及模型均未改变。特征池化、stencil、共享 DOF 处存在重复目标的 CUDA `index_add`；
默认模式的重复前向已直接复现末位数值差异，开启确定性算法后消失。
证据：[v400](../output/native-cpu-latency-2026-10-09/frozen-field-repeat-v400/field-repeat/result.json)、
[injInner](../output/native-cpu-latency-2026-10-09/frozen-field-repeat-inner/field-repeat/result.json)。
该探针是**固定零反馈条件场前向**，不是原完整事件输入的逐值重放。
原 A/B 完整事件只保存了摘要，因此不能将这里的 1e-17 数值当作原 A/B 误差上界，
不能改写其 `all_event_components_bitwise_equal=false`。
CPU 物理优化、真实密度输入等价与 CUDA 默认归约非逐位重复性分开报告。

## 剩余热点与下一步

优化后的 injInner profiler 在 `after-inner-profile/chain/`：首事件 77.93 s、第二事件 21.79 s。
这些带 profiler 数字不用于上面的加速比。
首事件 `locate_batch` 累计 40.90 s，`sample` 累计 20.89 s（其中 ADAS 约 8.36 s），
`deposit_batch` 约 4.32 s。父子累计项不能相加。
完整 `points` 调用由 50103 次降至 6318 次，虚拟场计算已去掉；
剩下的坐标/Jacobian 与真实场采样仍是显著计算，非仅重复摘要校验。

下一切片建议先验证 **GPU 批量定位＋原生 cubic 场采样**：
固定几何系数驻留 GPU，保持相同 Newton/阻尼/域外行为，先与 CPU 参考逐项对照，
再接有界真实事件；不能只报 kernel 速度而忽略传输和 fallback。
原子过程、沉积/稀疏投影分别测量后决定迁移，不能假设全部迁移必然更快。
跨窗口 CPU worker/GPU owner 流水可以补充吞吐，但必须有独立状态/RNG和权重版本绑定；
同一窗口下一事件仍等待当前预测场与 shard/诊断提交。
这些并行/GPU 粒子后端**本轮未实现**，也未开展训练。

## 当前实现复测命令

以下重放的是现有优化实现与保存的固定初态模型，不会重建历史旧源码版本。
历史旧实现计时以保留的 baseline 产物为证。输出目录须使用新名字。

```bash
.torch-cuda-env/bin/python -m tools.profile_native_coupled_latency \
  --run data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400 --anchor 4110 \
  --output-dir output/native-cpu-latency-2026-10-09/recheck-v400 \
  --no-profile --model-state output/native-cpu-latency-2026-10-09/baseline-v400-bound-no-profile/chain/initial-model.pt
.torch-cuda-env/bin/python -m tools.profile_native_coupled_latency \
  --run data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner --anchor 5680 \
  --output-dir output/native-cpu-latency-2026-10-09/recheck-inner \
  --no-profile --model-state output/native-cpu-latency-2026-10-09/baseline-inner-no-profile/chain/initial-model.pt
```

GPU 运行需要实际 CUDA 访问权限。可另加 `--deterministic` 做严格重复性检查，
但不能将确定性模式与默认模式的耗时直接作优化对照。
`--capture-arrays` 可保存数值场/历史/记忆及临时密度输入，存在额外传输开销，
同样不可混入无捕获的性能对照。

## 验证与最终边界

统一回归 **92 passed / 9.35 s**，命令：

```bash
.torch-cuda-env/bin/python -m pytest -q \
  tests/test_native_density_preparation.py tests/test_native_locator_latency.py \
  tests/test_native_nonlocal_event_binding.py tests/test_native_source_batch.py \
  tests/test_native_production_birth.py tests/test_native_production_density.py \
  tests/test_native_production_nonlocal.py tests/test_native_shard_dynamics.py \
  tests/test_native_production_events.py tests/test_native_production_torch_integration.py \
  tests/test_profile_native_coupled_latency.py tests/test_native_latency_comparison.py
```

出生 fixture 的重复 shard 校验由 5291 次降至 5 次；定位 fixture 的未使用场收缩由 48 次降至 0；
3-shard nonlocal fixture 的完整摘要由 6 次降至 2 次。独立公式、输入变更拒绝、
零步/边界/阻尼、RNG、状态回滚及两事件路径均有相关回归。
真实全云计时、真实密度公式对照和 CUDA 固定输入重复性是另外实际执行的证据，不混入 pytest 项数。
`git diff --check` 通过；未删除旧产物，未修改 JOREK/原始数据/roadmap/ADR，未训练。

验收限于三项 CPU 实现优化、两个有界完整案例的延迟下降和上述分层数值证据。
未宣称全事件 GPU 逐位等价、所有未来状态等价、统计稳定加速、GPU 跑满或正式训练就绪。
