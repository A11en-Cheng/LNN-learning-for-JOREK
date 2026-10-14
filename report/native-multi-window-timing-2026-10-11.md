# 多窗口完整耦合前向计时（2026-10-11）

两组 phaseA 各连续执行 **6 个完整切片**，全部成功。后续5个切片平均耗时：
**v400 5.7942 s；injInner 42.2104 s**。本轮冻结最近验证过的数值配置，
未进一步调整精度、未训练、未创建 optimizer、未修改 JOREK 或原始数据。
测试跨上海时区午夜；产物目录沿用启动时的 `2026-10-10`。

## 口径与配置

一个切片指公开 `TorchRecomputedSourceEvents.advance` 完成一次耦合事件：
独立 shard 密度源 → marker 出生/推进/五通道投影 → 九场空间网络/CfC →
OLD 场 shard 运动、侵蚀和 rocket → 预测场平衡/nonlocal/体积诊断刷新 → 状态提交。
不是只测给定真实场的 `Source.advance`，也不是一个 epoch。

只有登记初态读取真实完整场；后续 Psi 及其它场均由网络推进，持续回授源/shard。
后续 HDF 只读 `t_now/tstep/t_norm/index_now` 组成 `recorded_schedule_replay` 控制，
不读未来场/投影/粒子作为输入。没有教师替换、标签读取、loss、反向传播或优化器更新。
因此这里是**完整自主前向的成本**，不是完整训练 iteration 的成本。

沿用自主诊断 builder 的固定、非零、未训练权重与物理参考尺度，3147 个参数；
没有迁移旧训练检查点。这不是已训练模型的轨迹/精度评估，成本也不能保证覆盖所有
已训练预测状态。NumPy 源与 shard 仍停止梯度。

- RTX 5070 Ti；AMD Ryzen 7 9800X3D，8核16线程，WSL。
- Torch 2.11.0+cu130，NumPy 2.5.2；CUDA FP64 九场网络，CPU `batched_v1` 源。
- 两组串行，seed=20261010；Torch threads=2；OMP/MKL/OPENBLAS threads=1。
  源没有新增并行 worker。未独占整卡，开始时 NVML 显示已有2393 MiB占用和8%利用率；
  未持续采样利用率，不将阶段时间占比冒充 GPU 利用率。
- 出生概率表64格源宽；chunk=2048；百万 marker 槽全部保留、不裁剪云。
- 显式 `jorek_nearby_v1` + `local_subcycling_boundary_v2`；原筛选、64最少细步、
  1024最多细步、末段退出判据均未改。项目原生固定步默认未改变。
  该局部模式仍是 ADR-0023 登记的非默认数值近似，不是原生固定子步复刻。
- 显式 deterministic CUDA，`CUBLAS_WORKSPACE_CONFIG=:4096:8`。

使用 `diagnosing-bugs` 的性能测量分支：固定输入/种子、先实测、区分首次与后续成本，
复用直接子项扣除的计时树。不进入数值修复/参数搜索阶段。

## 逐切片结果

墙钟单位秒，已同步 CUDA；初始化/建网格不包含在切片时间中。
两组流体控制分别恒为0.05 / 0.2 JOREK 时间单位，不人为改为相同步长。
表中的 base 数是**本次源窗口实际执行的粒子 base 子步**，不是流体更新次数。

| 切片 | v400 流体目标编号 | 总耗时 | 粒子 base 数 | injInner 流体目标编号 | 总耗时 | 粒子 base 数 |
|---|---:|---:|---:|---|---:|---:|
| 1 | 4112 | 10.1494 | 40 | 5682 | 184.4674 | 81 |
| 2 | 4113 | 5.8605 | 4 | 5683 | 42.0520 | 16 |
| 3 | 4114 | 5.7576 | 4 | 5684 | 41.6641 | 16 |
| 4 | 4115 | 5.7968 | 4 | 5685 | 41.7256 | 16 |
| 5 | 4116 | 5.8095 | 4 | 5686 | 41.6134 | 16 |
| 6 | 4117 | 5.7468 | 4 | 5687 | 43.9967 | 17 |

首次包含导入后的较长源时钟衔接与第一次 CUDA 场执行，不可作常规切片成本。
injInner 的流体间隔为16.2个固定粒子 base 间隔；残余时间持久累积，
最后一窗实际执行17步，不是修改步长或重复某个流体事件。
其源推进从约35.3 s增至37.65 s，不能把最后切片较慢全部解释成随机计时抖动。

| 汇总 | v400 | injInner |
|---|---:|---:|
| 后续5窗均值 | 5.7942 | 42.2104 |
| 后续5窗中位数 | 5.7968 | 41.7256 |
| 后续5窗最小—最大 | 5.7468—5.8605 | 41.6134—43.9967 |
| 仅16个粒子 base 的4窗均值 | 不适用 | 41.7638 |
| 6个完整切片合计 | 39.1206 | 395.5193 |
| 诊断入口 elapsed | 50.2370 | 406.2180 |

入口 elapsed 包括 CUDA 初始化、建模、初态校验、摘要/报告等；不是 shell 启动到退出时间。
每组仅一条连续轨迹，后续5窗不是5次同初态重复，范围不是置信区间或长程稳定性结论。

## 后续切片的非重叠分项

下面是切片2—6的均值；百分比分母为相应组后续完整切片均值。
`source.advance` 已包含场采样、ADAS、电荷演化、运动/定位、沉积及投影求解，
**不含出生**；本轮没有进一步拆其内部步骤。接口准备不是 JOREK 九场矩阵求解。

| 阶段 | v400 秒/切片 | 占比 | injInner 秒/切片 | 占比 |
|---|---:|---:|---:|---:|
| marker 出生 | 0.8136 | 14.04% | 1.8208 | 4.31% |
| marker 完整源推进与投影 | 0.4071 | 7.03% | 35.8381 | 84.90% |
| 独立 shard 密度源 | 1.2538 | 21.64% | 1.1937 | 2.83% |
| shard 运动/侵蚀/rocket | 0.4907 | 8.47% | 0.3954 | 0.94% |
| shard 诊断刷新/nonlocal/体积 | 1.3101 | 22.61% | 1.4676 | 3.48% |
| 场接口准备/context 核验 | 1.0474 | 18.08% | 1.0215 | 2.42% |
| 其余状态校验 | 0.1553 | 2.68% | 0.1555 | 0.37% |
| 源输入转换 | 0.0021 | 0.04% | 0.0017 | <0.01% |
| 九场空间网络/CfC 前向 | 0.0738 | 1.27% | 0.0682 | 0.16% |
| 其余事件工作 | 0.2403 | 4.15% | 0.2480 | 0.59% |
| 合计（未舍入） | 5.7942 | 100% | 42.2104 | 100% |

直接子项从父墙钟扣除，每窗 accounting closure 均通过；不能再加 inclusive 父项。
计时包装有 CUDA 同步/日志成本，未启用 cProfile、torch.profiler 或全云数组 capture。
这不是对旧 `native_fixed` 的配对提速试验；近期局部细分改变了计算量，
不能直接与旧完整链报告的运行时间作算法提速结论。

injInner 首窗接受4,376,204个粒子实际推进段，其中111,208个为细分/末段，
计时组113,314个；普通16-base窗细分段约22,528—22,820，17-base窗23,936。
v400 全部窗口没有触发细分。因而本轮观察到：
**injInner 被 CPU marker 局部细分主导；v400 的剩余开销分散在密度、诊断和接口核验。**
本轮不对此做进一步算法调整。

## 初态、资源与验证

v400 活粒子11168→11207；injInner 52604→52741，百万槽始终保留。
源、字段、rawD、反馈、shard、CfC记忆和各自时钟都跨事件持久延续，不逐窗重新导入。
源窗口与流体目标编号相差一步，按现行契约分别记录，没有靠同名文件配对。

两组模型/buffer和输入 runtime 摘要前后不变；运行期间源码依赖 hash 未变。
这证明本次是固定算法/权重的只读诊断，不是完整全状态物理精度或模型恢复验收。

| 资源/准备 | v400 | injInner |
|---|---:|---:|
| 原生网格/云/模型构建 | 8.5535 s | 8.6457 s |
| CUDA 转移 | 0.0909 s | 0.0639 s |
| adapter 初始化 | 0.5148 s | 0.4358 s |
| 进程 RSS 峰值 | 2.902 GiB | 2.951 GiB |
| Torch 显存分配峰值 | 245.44 MiB | 245.44 MiB |
| Torch 显存 reserved 峰值 | 284.00 MiB | 284.00 MiB |

峰值是进程/allocator 累计峰值，不是各阶段相加，也不包含整卡其它进程。

新入口测试4项；与既有前向/反向计时测试合计 **18 passed，1条已知 profiler 警告**。
六事件公开路径与不加计时的路径最终状态及全局RNG摘要一致，失败保留已完成切片和未完成 timer；
标量控制读取测试不提供未来场数据集，并验证拒绝跨 restart。

## 复现与机器证据

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.torch-cuda-env/bin/python -m tools.benchmark_native_multi_window \
  --run-index 0 --windows 6 --deterministic \
  --output-dir output/native-multi-window-timing-2026-10-10/replay-v400

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.torch-cuda-env/bin/python -m tools.benchmark_native_multi_window \
  --run-index 1 --windows 6 --deterministic \
  --output-dir output/native-multi-window-timing-2026-10-10/replay-inner

.torch-cuda-env/bin/python -m pytest -q \
  tests/test_benchmark_native_multi_window.py \
  tests/test_profile_native_coupled_latency.py tests/test_profile_native_coupled_backward.py
```

需要实际 CUDA 可见权限；两组串行执行，输出目录必须不存在；进程墙钟上限900 s、RSS预算8 GiB。

- [机器汇总](../output/native-multi-window-timing-2026-10-10/summary.json)
- [v400 完整计时树/逐切片结果](../output/native-multi-window-timing-2026-10-10/v400/chain/result.json)
- [injInner 完整计时树/逐切片结果](../output/native-multi-window-timing-2026-10-10/injInner/chain/result.json)
- [v400 配置和源码 hash](../output/native-multi-window-timing-2026-10-10/v400/result.json)
- [injInner 配置和源码 hash](../output/native-multi-window-timing-2026-10-10/injInner/result.json)
- [计时入口](../tools/benchmark_native_multi_window.py) / [测试](../tests/test_benchmark_native_multi_window.py)

入口 SHA256：`894c88516d4d0196c6f3c96e9668c6c4882c0af0b926382a43d770b68730fdca`。
原始机器结果保留完整小数、配置、源执行计数、确认退出事件及前后源码 hash 核验。
