# 空间热点 GPU 可行性与完整事件对照（2026-10-09）

## 范围与验收边界

继续上一轮 [CPU 延迟优化](native-cpu-latency-2026-10-09.md)，使用 diagnosing-bugs
的真实计时—独立数值对照回路。用户授权继续优化并复用子代理；不训练、不更新
optimizer、不读取预测段真实场回授、不裁剪 marker 云、不修改物理步长、JOREK 或原始数据。

Plato 实现空间后端及单元对照，Ramanujan 实现有界计时与真实空间对照，Franklin
实现完整状态比较，主代理负责源事件接线、验证协调与报告。真实 GPU 作业串行执行。

## 实现

- `src/models/native_cuda_spatial.py`：显式 FP64 Torch 空间后端，固定几何驻留设备；
  五项原生 cubic 求值输出与原接口一致。粒子位置不缓存，场值使用事件内有名、独立
  所有权的快照，每个新采样时刻显式替换；static 模式每事件上传一次。
- `NativeMarkerSource(spatial_backend='cuda_fp64_v1')`：只替换源推进中的批量场/反馈
  采样及有 hint 的定位；原子过程、电荷推进、沉积、稀疏投影、出生和 shard 更新仍为 CPU。
  完整云、rawD 和投影历史不裁剪。整个源链仍是 NumPy 边界，不声明端到端可微。
- Newton 保留 25 次上限、原阻尼序列、边界和残差容差；接近舍入阈值、边缘或
  不适定 Jacobian 时，以及未定位/跨元素粒子，交给原 CPU 标量候选搜索。
  不承诺 GPU 与 NumPy 逐位相同；精确元素身份和相对数值误差分别核查。
- 事件开始验证完整网格/配置摘要，同步事件内只读几何不重复摘要；结束或异常均
  清空全部场快照。独立公共调用仍逐次验证。生产入口只接受 CUDA，设备不可用明确
  失败；CPU device 仅供测试，不是生产自动回退。
- source 配置记录 backend、精度、几何绑定、设备和数值策略，生产续跑配置守卫不放松。
  默认 CPU `batched_v1`/原有 `scalar` 选择不变。

## 复现与捕获

目录：`output/native-cuda-spatial-2026-10-09/`。
使用现存各案例完整初态模型快照（参数、谱缓冲区、RNG），固定 chunk=2048、threads=2。
开启确定性场算法以隔离上一轮默认 CUDA 归约的末位差异；不与上一轮默认模式计时
直接计算加速比。数值捕获保留全部 1000000 slots 及两事件逐场/degree、rawD、memory、
云、投影值/增量、shard、时钟与 RNG；捕获开销与无捕获计时分开。

初始参考运行 `cpu-v400-capture`、`cpu-v400-capture-v2` 均被严格配置守卫拒绝，
失败产物保留。逐项复核的实际差异仅为资源预算：旧快照 `max_seconds=880`，
命令默认 900。后续显式使用 `--wall-seconds 880`，未放松预算守卫。
另完善了工具恢复顺序：先事务性恢复参数/缓冲区，再核查由缓冲区生成的配置；
若不匹配，恢复原模型参数及各子模块模式。不能把这项工具修订误称为上述预算拒绝
的原因，也不能用同 seed 代替完整快照绑定。

`tools/compare_native_spatial_rollouts.py` 输出逐场与逐 degree、五通道投影、活粒子
匹配身份及 shard 的原生单位 relative L2；零参考显式区分零/零与非零/零，拒绝 NaN。
比较程序成功只表示报告完整，不等于数值等价或性能验收。

## GPU 空间可行性结果

两个真实初态完整 slots=1000000，实际活粒子分别 11168/52604。
`spatial-v400/result.json` 访问全部活粒子，3 次重复；`spatial-inner/result.json`
为有界单次测量，首调用包含 CUDA 首次 kernel/库启动，不能把两组冷/热数字混比。
测量包含粒子输入 H2D、全部五项输出 D2H 和 CPU fallback。

| 初态 | CPU 九场采样 | GPU 热驻留采样 | 实际位置定位 CPU/GPU | 2 mm 移动定位 CPU/GPU |
|---|---:|---:|---:|---:|
| v400（中位） | 0.01648 s | 0.01136 s | 0.00955 / 0.01523 s | 4.82275 / 5.16265 s |
| injInner（单次） | 0.09251 s | 0.04423 s | 0.04735 / 0.23313 s（GPU 冷调用） | 12.11528 / 13.60109 s |

所有 actual/moved/显式无效 hint 案例元素身份完全一致；移动后 local 坐标最大绝对差
分别 4.06e-13、2.80e-13。这是定位坐标的原生尺度诊断，不用于混量纲的物理误差判据。
这组微测量早于最后的场/反馈共享几何求值修改，不能作为最终组合 kernel 的速度证明。
据测量增加 `points_frozen_many`：源事件只上传一次坐标、求一次几何，场和反馈仍分别
按原算式收缩；单元测试核查共享一次与逐项调用精确一致、坐标每次刷新。

## 完整 GPU 事件诊断（有数值捕获，不是无捕获性能基线）

| 初态 | CPU 事件1/2 | GPU 事件1/2 |
|---|---:|---:|
| v400 | 31.738 / 6.642 s | 34.241 / 6.855 s |
| injInner | 67.877 / 16.980 s | 80.739 / 20.278 s |

同模型、同物理初态、同 chunk/预算、同确定性模式，各执行两次完整自主事件。
没有裁剪/真实未来场回授/训练，原生 full-slot 捕获保留。两例中 GPU 路径诊断耗时均
较大，**不将其设为默认，也不据此宣称整体加速或 GPU 满载**。
精确数值比较另见 `comparison-v400.json`、`comparison-inner.json`；报告成功仅表示
比较完整。v400 的场/rawD 精确一致、离散云身份一致，但这不能推广到另一 run。

**injInner CUDA 数值资格失败**：两事件零基 slot 600505 的 CPU 元素为 3370、CUDA
为 3354，身份 `('001', 600505, 1)`、liveness、weight、出生时刻均相同，排除了 slot
复用。该粒子初始位于元素 3354、local=(0.97448,0.99745)，事件后位置与速度显著不同。
全云分量 relative L2 最大值：st 6.79e-3、V_prev 0.769、投影值（逐通道）1.27e-3、
投影 deltas（逐通道）1.01e-2；memory 1.90e-4，逐场/degree rawD 4.92e-8。
九场本身误差较小（逐场/degree 1.78e-12）不表示闭环一致。
RNG、时钟及其它离散身份一致；单粒子 R/Z 位置差贡献超过 99% 的对应云平方误差。
详细原生状态见 `slot600505-inner-detail.json`。不将这些传播后的差异笼统归为统计/舍入误差。

尚未定位第一处分歧，暂不修改物理算法或扩大容差。下一次 CUDA 诊断需用原始
2048-row chunk 重放同输入：场/反馈求值→速度→push→定位，记录 Newton、阻尼、
Jacobian 与 fallback；只看事件后快照不足以区分定位分支、插值放大和 batch-wide
异常回退。源码注释已明确此后端不具训练/生产数值资格；默认保持 CPU。

## CPU 共享几何切片

按独立计时反馈转向减少重复工作：`NativeSourceMesh.points_many` 同一次查询只求
一次 cubic 基函数、坐标/Jacobian，九场与五通道仍分别执行原收缩，不改变浮点算式。
`deposit_batch` 保留几何守卫与原 `np.add.at` 累加顺序，去掉未使用的虚拟场值/梯度。
不缓存粒子或可变 ndarray，不修改源配置和物理分支。

`tests/test_native_shared_spatial.py` 用独立冻结的旧算式核查空 batch、曲网格、带符号
size、重复 DOF、非法几何及完整源事件；逐项精确一致，操作次数测试确认两组场共享
一次几何、沉积虚拟场收缩由 3 次变为 0。真实完整事件的数值与计时结论以下单独登记。

历史 `after-*-no-profile` 的计时比较被严格工具拒绝：旧环境没有记录
`CUBLAS_WORKSPACE_CONFIG`，新环境明确记录 null。source/model/控制指纹等一致也不能
填补该元数据缺口；保留 `shared-cpu-v400-timing.json` 失败证据，不放松守卫。
后续在同一环境用显式冻结旧算式重新测量参考，再与新路径比较。

### 完整 CPU 状态保真

`shared-cpu-v400-comparison.json`、`shared-cpu-inner-comparison.json` 使用同一确定性场
模式及完整初态模型，分别比较共享前后两事件捕获。**不启用空间 backend 白名单**，
source 配置、完整模型及初态绑定严格一致。两例各两事件全部捕获数组逐位一致：
九场（全部 degree）、rawD、memory、完整云 storage、投影值/原始 deltas、shard 及
瞬时独立密度输入，时钟与 RNG 也一致；定义有效的 relative L2 均为 0。
保真仅针对上述两个有限真实片段，不代表任意长程预测已验收。

### 同环境无捕获计时

`tools/native_spatial_reference.py` 显式冻结旧逐场 geometry/虚拟场沉积算式，
`--cpu-shared-reference` 仅在诊断工具中绑定本实例，不更改生产类、物理配置或检查点
守卫，不导入 tests。公式及完整源事件测试核查其结果/RNG 与共享实现一致。
这是当前环境重新执行的旧算式参考，**不是重建某个完整历史 commit**。
旧 `points` 原本没有逐调用 full-field 校验，外层 source 已校验；新公开共享接口新增
逐调用 shape/finite 校验，其成本也包含在新路径实测中，不给参考人为添加新工作。

两组均默认非确定性场模式、无 profiler、无捕获，同 snapshot/RNG、同预算与当前
完整环境记录；使用 `report_native_cpu_latency` 严格检查，未归一化缺失元数据。

| 初态 | 冻结旧算式 事件1/2 | 共享几何 事件1/2 | 两事件 前→后 | 观察到的延迟减少 |
|---|---:|---:|---:|---:|
| v400 | 32.337 / 6.872 s | 31.281 / 6.489 s | 39.210→37.770 s | 3.67% |
| injInner | 68.254 / 16.795 s | 64.534 / 16.152 s | 85.048→80.686 s | 5.13% |

产物 `fresh-shared-cpu-{v400,inner}-timing.json`。各一次成对观测，不是重复测量的
统计置信区间，也不是完整训练吞吐；没有宣称 GPU 利用率已达到某个百分比。
默认模式 field/torch_state 摘要不一致，两文件仍保留
`all_event_components_bitwise_equal=false`，工具相应返回非零；云、投影、shard、时钟
及 RNG 相同。默认归约差异不能由确定性捕获报告隐去，两种证据分别保留。
全量确定性捕获精确一致说明共享计算本身在上述有限片段保真；它不是默认模式
每次逐位重复性的保证。

复现命令（输出目录必须新建；GPU 可见环境）：

```bash
.torch-cuda-env/bin/python -m tools.profile_native_coupled_latency \
  --run data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400 --anchor 4110 \
  --wall-seconds 880 --no-profile --cpu-shared-reference \
  --model-state output/native-cpu-latency-2026-10-09/baseline-v400-bound-no-profile/chain/initial-model.pt \
  --output-dir output/native-spatial-reference-v400-recheck
.torch-cuda-env/bin/python -m tools.profile_native_coupled_latency \
  --run data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400 --anchor 4110 \
  --wall-seconds 880 --no-profile \
  --model-state output/native-cpu-latency-2026-10-09/baseline-v400-bound-no-profile/chain/initial-model.pt \
  --output-dir output/native-spatial-shared-v400-recheck
```

injInner 使用 `phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner`、anchor=5680
及对应 `baseline-inner-no-profile/chain/initial-model.pt`。数值捕获另加
`--deterministic --capture-arrays`，不与上述计时混比。

## 下一切片

CPU 仍为主路径；收益有限，主要定位热点尚未消除。下一步优先评估 CPU 定位编译化
或独立窗口的 CPU worker/GPU owner 流水；窗口状态/RNG/权重版本必须隔离，同一
窗口仍按因果事件顺序推进，不能预计算依赖未来预测场的源。
CUDA 首处分歧用原 chunk 独立重放定位；未取得数值资格前不用于训练。
本轮未开展训练、改学习容量或宣称 GPU 满载，未修改 roadmap、ADR、JOREK 或原始数据。

### 回归

相关统一无真实训练回归：**184 passed，1 skipped**（10.68 s）。skipped 为默认
沙箱不可见 CUDA 的可选测试；在显式设备许可下单独执行当前最终版本的两源事件
小型 CUDA 回归：**1 passed**（1.78 s）。该小例不能覆盖已失败的真实 injInner 分支。

命令：

```bash
.torch-cuda-env/bin/python -m pytest -q \
  tests/test_native_shared_spatial.py tests/test_native_cuda_spatial.py \
  tests/test_native_cuda_source_integration.py tests/test_native_spatial_reference.py \
  tests/test_profile_native_coupled_latency.py tests/test_compare_native_spatial_rollouts.py \
  tests/test_benchmark_native_cuda_spatial.py tests/test_native_source_batch.py \
  tests/test_native_source_time_integration.py tests/test_native_locator_latency.py \
  tests/test_native_density_preparation.py tests/test_native_nonlocal_event_binding.py \
  tests/test_native_production_birth.py tests/test_native_production_density.py \
  tests/test_native_production_nonlocal.py tests/test_native_shard_dynamics.py \
  tests/test_native_production_events.py tests/test_native_production_torch_integration.py \
  tests/test_native_latency_comparison.py
```
