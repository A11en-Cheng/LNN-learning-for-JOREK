# 公共耦合接线与完整云有界源诊断（2026-10-09）

本轮不训练、无 optimizer 更新、未修改 JOREK/原始数据。主线程整合 A/B 机制切片；使用 codebase-design/tdd，在已确认的源推进、完整事件、检查点 seam 验证。保留现有工作树修改。**真实生产完整耦合事件仍为 0**，下述完整云源窗口不是生产初始化或九场闭环认证。

## 已接线

- `NativeMarkerSource(execution_budget=...)` 显式传递 `MarkerExecutionBudget`，预算进入配置指纹；不提供预算仍保留旧 10000 活粒子防线。没有裁剪云或更改粒子数。
- `NativeSourceMesh.locate` 使用完整 cubic 几何的 Bernstein 控制网凸包作保守包围盒，空间候选筛选后进行参考域内阻尼 Newton；无效候选不再导致整个搜索抛错。不能只用顶点包围盒截掉曲边。保留元素/局部坐标 hints，跨元素时再搜索。此数值定位策略登记 `bernstein_bounds_damped_newton_hints_v1`，不是移植 JOREK locator 或证明任意畸变网格都可解。
- `advance_source_event` 先构造保存 rawD 的时间 sampler，再将同一 sampler 交给生产 births 与 source.advance。旧参考 shard 保留原 Interface。
- `advance_coupled_reference` 新增显式 `equilibrium_provider`；生产模型没有配置 provider 时提前失败。provider 为无隐藏状态的场诊断依赖，接收本次预测完整 U 与 SI 时刻，不能读取未来标签。返回必须绑定相同场 digest/时刻。
- `refresh_coupled_diagnostics` 在私有场/shard候选产生后、完整事件返回前更新 volume/nonlocal 缓存，仍处于模型/RNG回滚范围内；刷新失败不产生可提交状态。缓存和 provider 配置 digest 保存在 shards；不同配置不能静默续跑。
- Reference/Torch Adapter 共用上述 helpers。Torch 保留跨步场/rawD/记忆图，源/shard/ES 仍为 NumPy 停止梯度路径；不是端到端可微物理链。

诊断顺序直接核对只读 JOREK `core/mod_jorek_timestepping.f90:414–425,446,510`：OLD 场 nonlocal 初始化（若需）与 shard update 在 update_values 前；新场 update_equil_state 与 int3d_new 在后。所检查版本延续 A 报告的 `54c675bcc39c33b082c0c35046e28122dc53b3aa`；不据此证明 dirty 生产构建等价。

## 独立验证与兼容范围

预算入口测试先因缺 execution_budget 参数失败；接线后源公开入口通过，并验证子步预算拒绝。定位测试用独立解析曲边 `R=2+s−0.4s², Z=t` 与远处矩形，旧实现 Newton 跑出参考域而抛错；新实现在正确单元得到给定局部坐标，真实域外仍返回未找到。

生产事件测试先证明旧 driver 没有拒绝缺 ES provider；新 driver 提前拒绝。小解析生产 fixture 使用显式解析 ES（不是生产 provider），完整 source/shard 机制两事件、rawD 出生、诊断时钟/场指纹、失败 RNG 回滚及新模型恢复通过。原 fixture 在 driver 外部刷新诊断的 wrapper 已移除；刷新现在由公共事件完成。

新增 locator/预算配置会影响模型配置指纹；旧配置不得被视为本路径的精确恢复。分块浮点求和不保证不同 chunk 配置逐位相同；相同配置恢复测试与跨配置容差测试分开。

## 两组真实云：资源参考，不是生产闭环

从现有认证清单只读载入完整云。外部父进程对每个任务设置 **60 秒 / 1536 MiB** 硬监控；内部源预算 chunk=64、最多 1 个粒子子步，冻结完整场，dt=1 ns，无新出生、零反馈初始化、未滤波独立参考投影。该初始化仅为明确的资源诊断，不能当作合法普通/restart 生产初态。

| 量 | v400 4110 | injInner 5680 |
|---|---:|---:|
| 保留槽位 | 1000000 | 1000000 |
| 输入/输出活粒子 | 11168 / 11168 | 52604 / 52604 |
| 实际处理粒子步 | 11168 | 52604 |
| chunks | 175 | 822 |
| 源窗口耗时（秒） | 8.8106 | 37.1738 |
| 完整子进程耗时（秒） | 12.1274 | 39.8580 |
| 进程峰值 RSS（bytes） | 1080217600 | 1100705792 |
| 完成参考源窗口 | 1 | 1 |
| 完成真实生产完整事件 | 0 | 0 |

另一个只读成本诊断验证两云首个活粒子的无 hints 定位分别恢复原元素 2903/398；局部坐标最大差约 2.79e−10 / 6.11e−16。全云源执行使用已有单元 hints，真实推进没有再触发原无保护 Newton 错误。少量点值成本的线性外推不是完整事件成本，本表使用实际窗口耗时。

机器证据：`output/native-coupling-coordination-2026-10-09/native-coordination-locator-cost-20261009.json`、`native-coordination-source-attempt-20261009.json`；产物保留 training=false、full_coupled_events=0、qualification_granted=false。

## 回归与复现

初轮跨模块回归 **117 passed / 15.66s**，JUnit 为同目录 `native-coordination-regression-20261009.xml`。它涵盖预算、生产机制、时间、边界、源/场、稀疏投影、检查点和训练接口（不执行 curriculum/optimizer）。后续专项与最终结果见下文追加。

最终主线程独立复跑 **128 passed / 16.07s**，证据为同目录 `final-v2.xml`。增加的 11 项覆盖两种 Adapter 缺 ES 提前失败、rawD 出生与内部诊断刷新、CPU Python/NumPy/Torch RNG 完整回滚、两事件跨步 BPTT、三事件分段/新模型恢复、ES 配置改变拒绝和注册启动事件。解析 fixture 不授予真实数据资格。

保留 `final-regression.xml` 的中间失败证据：127 passed / 1 failed，仅第二次累计时间 `3.0000000000000004e-9` 与字面量 `3e-9` 精确比较失败；测试改为诊断时间精确等于 runtime 时钟，对预期物理时间使用显式容差。未改物理推进或放宽因果绑定。所有本轮编辑文件（包括未跟踪文件）逐文件 whitespace check 与相关模块编译通过。

```bash
.torch-cuda-env/bin/python -m pytest -q tests/test_native_coupling_coordination.py tests/test_native_production_events.py tests/test_native_production_torch_integration.py
PYTHONPATH=. .torch-cuda-env/bin/python tools/diagnose_native_cloud_cost.py \
  --anchors output/native-closure-certification-2026-10-08/certified/anchors.json \
  --output /tmp/native-new-source-attempt.json --mode attempt \
  --seconds 60 --memory-mib 1536 --chunk-size 64
```

128 项复跑文件集合（传给同一 `pytest -q` 命令）：

```text
tests/test_native_coupling_coordination.py tests/test_native_production_events.py
tests/test_native_production_torch_integration.py tests/test_native_production_nonlocal.py
tests/test_native_production_density.py tests/test_native_production_birth.py
tests/test_native_shard_dynamics.py tests/test_native_bounded_cloud.py
tests/test_native_boundary_learning.py tests/test_native_field_boundary.py
tests/test_native_execution_features.py tests/test_native_source_time.py
tests/test_native_source_time_integration.py tests/test_native_joint_source.py
tests/test_native_coupling.py tests/test_native_sparse_source.py
tests/test_native_projection_n0.py tests/test_native_checkpoint_control.py
tests/test_base_coupled_training.py tests/test_base_coupled_noise.py tests/test_coupled_metrics.py
```

工具拒绝覆盖既有输出；复现请用新路径。本轮无 field 准确率或物理加速结论。

## 尚未完成及下一最小工作

1. 实际预测场的 ES diagnostic provider：轴/边界磁通、LCFS 状态、X-point 类型及位置/磁通需要对当前 U 因果计算或合法缓存；当前仅解析 fixture provider。不能把下一真实文件的诊断读入预测步骤，也不能永久冻结初始 ES 冒充动态诊断。
2. 与该 provider 一致的合法初始 nonlocal/volume 状态、目标有效配置及控制绑定；此处初始化与普通/restart 时钟差异需显式处理。生产随机分布路径仍是 MT19937 模型重放，非生产 PCG32/MPI 逐粒子复现。
3. 目标过滤投影 + 全 shard + 出生 + 诊断 + 九场的总成本与两完整事件尚未实测；本轮单源资源结果不能推算其必然满足相同预算。
4. Mach1/natural-open 仍是参数和适用关系进入联合学习，不等于其弱式响应已学会。未训练网络不以准确度为机制验收门槛。

下一轮应集中实现/验证当前场 ES 诊断，再尝试合法真实一步、第二步消费预测场和恢复；不扩展训练计划、不授予训练资格。

### 收尾只读核查：ES 是可实现的诊断缺口，不是必须等待新数据

Franklin 的只读核查确认，两锚点同刻 HDF 的轴/边界/X 点 ψ、R/Z 历史及目标 namelist 搜索参数可用于初始化。两锚点 `psi_axis_t` 分别约 0.504467596 / 0.503745896，`psi_bnd_t` 约 0.113268358 / 0.112567132；这些是初始缓存，不可从后续真实文件刷新预测态。`xcase=3` 是 double 配置，不应与动态 active_xpoint 混同。

主线程另直接复核 `models/equil_info.f90:206–260,1280–1310` 和 `communication/mod_import_restart.f90:1542`：边界由有效远轴 X 点距磁轴的磁通差选择，无合适 X 点时需 limiter 分支；`LCFS_lost` 依赖 restart 的 index_start 前轴历史到边界元素中心距离判据，不是随意每个预测步重设的布尔值。代理按完整轴历史重算的两组最小距离约 0.74245 / 0.73586 m，大于 R_geo/30=0.037 m，支持两个初始化的 false；该数值重算为代理证据，主线程本轮仅复核判据源码。

因此后续应补的是**完整 cubic ψ 的驻点搜索、X 点/limiter 边界选择与所需搜索状态持久化**，不是完整平衡/PDE 求解器，也不是默认要求重新导出训练数据。当前公共 provider 按无隐藏状态依赖设计；若采用有跨步种子的实现，必须将种子纳入显式完整状态及回滚/checkpoint，不能藏在 callback 对象里。
