# 当前完整耦合链基准（2026-10-10）

## 结论与范围

当前默认后端仍由 CPU 源／shard／状态工作主导，不是 CfC 或反向传播。
两组完整链的非重叠计算＋标签读取耗时为 **45.9414 s / 94.5096 s**。
v400 最大单项是出生（32.07%）；injInner 最大单项是 marker 完整推进（73.60%）。
场网络前向与反向合计分别仅占 2.52% / 1.17%。

这是一次新的固定权重、无 optimizer 更新的基准，不是训练或物理精度验收。
采用 `diagnosing-bugs` 的性能测量流程：先粗测，再补针对出生和配置准备的细分 timer，
直接子项扣除父项，保留未分类残差。不进行性能修复或改变生产默认路径。

严格恢复的权重：
`output/native-cuda-curriculum-2026-10-09/seed1-run3/epoch-3-parameters-optimizer.pt`。
两次无梯度完整物理前缀、B=0 记忆重置、两次完整自主预测、四项监督及对象指标、
一次两步反向；另做独立跨步连通检查。**不是四步 BPTT，也不是一个 epoch**。
未来节点标签只在两次预测结束后读取；无真实 Psi 或其它场回授。
控制仍为已登记步长的 schedule replay，不宣称自主控制输入资格。

百万 marker 槽位全部保留，初始活粒子 11168 / 52604；不截断云。
源为 CPU `production_cpu_batched_v1`；Torch 场／CfC／loss／反向为 CUDA FP64。
保留当前采样与几何复用优化；此前密集 lookup、系数缓存、多 worker 实验均未启用。
NumPy marker、出生、shard 和平衡诊断仍停止梯度，不宣称全物理链端到端可微。

## 硬件与配置

- RTX 5070 Ti；AMD Ryzen 7 9800X3D，8 核／16 逻辑线程，WSL 环境。
- Torch 2.11.0+cu130，NumPy 2.5.2。
- Torch CPU threads=2、marker chunk=2048、birth proposal 上限=1000000；
  threads=2 不代表源推进已采用两个 Python worker。
- 显式 deterministic CUDA，`CUBLAS_WORKSPACE_CONFIG=:4096:8`；
  OMP/MKL/OPENBLAS 线程环境未改写。
- 两组串行运行；未独占整卡。开始时整卡还存在约 2170 MiB 占用和 7% 利用率。
  未持续采样 NVML 利用率；阶段墙钟不能直接换算成整卡 GPU 利用率。

## 无重叠耗时分布

单位秒；百分比分母为该组完整计算＋标签读取，不包含构建／恢复／额外诊断。
每个出生、源推进、密度、shard 更新、刷新和场前向都实际执行 4 次。
`source.advance` 不包含出生；“推进”不是仅位置更新。
`field_solver_prepare` 是 `reference_solver` 接口创建与 context 核验，
**不是 JOREK 九场矩阵求解**；它有 18 次调用，其中部分嵌套在状态校验中。

| 非重叠阶段 | v400 秒 | 占比 | injInner 秒 | 占比 |
|---|---:|---:|---:|---:|
| marker 出生（含出生位置采样/定位） | 14.7336 | 32.07% | 2.0334 | 2.15% |
| marker 完整源推进（含采样、电荷态、沉积和投影） | 8.6031 | 18.73% | 69.5548 | 73.60% |
| 独立 shard 密度源 | 5.0559 | 11.01% | 4.7758 | 5.05% |
| shard 运动／侵蚀／rocket 更新 | 2.2772 | 4.96% | 1.6553 | 1.75% |
| shard 体积／nonlocal 诊断刷新 | 5.2677 | 11.47% | 5.8983 | 6.24% |
| 场参考接口准备／context 配置核验 | 4.7364 | 10.31% | 4.6913 | 4.96% |
| 其余状态校验（扣除嵌套接口准备） | 0.6955 | 1.51% | 0.7300 | 0.77% |
| 源反馈输入转换 | 0.0080 | 0.02% | 0.0077 | 0.01% |
| 九场空间网络＋CfC 前向 | 1.0917 | 2.38% | 1.0192 | 1.08% |
| 四项监督 loss＋逐场／源／位置指标 | 0.7880 | 1.72% | 1.4619 | 1.55% |
| 预测后的真实标签读取 | 0.1344 | 0.29% | 0.1397 | 0.15% |
| 一次两步 loss 反向 | 0.0670 | 0.15% | 0.0832 | 0.09% |
| 其余事件工作（平衡诊断、复制、驱动等） | 2.4830 | 5.40% | 2.4590 | 2.60% |
| 合计 | 45.9414 | 100% | 94.5096 | 100% |

原始计时保留父子区间和 ID；exclusive=父墙钟−直接子项墙钟。
不要额外加上 `prefix`、`burnin_and_forecast`、`reference_validation`
或 `torch_state_validation` 的 inclusive 总量，否则会重复计数。
“其余事件工作”仍有约 2.5 s，没有假装全部归到出生或某一 CPU 算子。

## 首次与后续事件不能混为稳态

按四次完整事件的调用顺序（前缀 1、前缀 2、预测 1、预测 2）：

| 调用 | v400 出生 | injInner 出生 | v400 marker 推进 | injInner marker 推进 |
|---|---:|---:|---:|---:|
| 1 | 12.7503 | 1.0601 | 6.4406 | 43.7729 |
| 2 | 0.6208 | 0.1796 | 0.6941 | 8.6919 |
| 3 | 0.5087 | 0.5384 | 0.7277 | 8.6371 |
| 4 | 0.8537 | 0.2554 | 0.7407 | 8.4530 |

首次前缀事件包含原生导入后的 marker 时钟衔接，源窗口不等于后续普通单步成本；
不能据此宣称每个普通流体步都需 43.77 s，或用四次均值作稳态吞吐。
第一次场前向包含 CUDA 首次执行开销：
v400 0.8689 s，injInner 0.8140 s；
后续三个分别共 0.2228 s /
0.2052 s。
未插入热身来替代真实首事件。

## 构建、恢复与总时间

| 额外／顶层时间 | v400 秒 | injInner 秒 |
|---|---:|---:|
| 检查点读取 | 0.0049 | 0.0054 |
| 原生网格／云／模型构建 | 8.0839 | 8.3947 |
| 严格权重／buffer／配置恢复 | 0.4548 | 0.4594 |
| 模型 CUDA 转移 | 0.0634 | 0.0630 |
| 额外跨步 autograd.grad（不在分布表中） | 0.0096 | 0.0326 |
| 诊断入口 elapsed | 56.2840 | 105.1912 |

入口 elapsed 包含 CUDA 初始化、摘要／指纹检查、报告准备等未落入上述子计时的开销，
但不包含 Python 导入以及入口最后写 result.json 的时间；不是启动到退出的 shell 墙钟。
GPU 时间为同步边界墙钟，包含 CPU 发起调用与等待，不是纯 kernel 时间。

粗计时单独保留：
v400 计算＋标签读取 46.6788 s，
injInner 94.2765 s。
粗测和细测 hook 数不同，不将两者平均或把约百分之一的差别解释为提速／退化。
本轮每种配置、每组各一次；无 cProfile／torch.profiler 的实测，但同步 hook 本身仍有开销。

## 内存、梯度与保真范围

两组 Torch 显存分配峰值均 413615616 bytes（394.45 MiB），
reserved 峰值 484.00 MiB。
进程 RSS 峰值 3.439 / 3.440 GiB；
这是累计进程峰值，不能逐阶段相加，也不是整卡 NVML 内存。

两组 success=true；28 个可训练张量（3147 参数）全部有限非零梯度。
第二预测节点 loss 对第一预测节点 U/rawD/memory 的梯度均有限非零。
模型／buffer 和输入 runtime digest 前后未变；optimizer 创建／更新均为 0。
同组粗测与细测 loss 完全相同：0.2046684205563342 / 1.3075729366752518。
这不是逐位全状态对照、消融率误差或预测质量验收。

`backward(retain_graph=True)` 只为之后的独立连通检查保留图；
反向计时和显存释放行为不与普通训练完全相同。

## 复现与证据

```bash
.torch-cuda-env/bin/python -m tools.profile_native_coupled_backward --run-index 0 --deterministic --fine-components --output-dir output/native-full-pipeline-benchmark-2026-10-10/replay-v400
.torch-cuda-env/bin/python -m tools.profile_native_coupled_backward --run-index 1 --deterministic --fine-components --output-dir output/native-full-pipeline-benchmark-2026-10-10/replay-inner
.torch-cuda-env/bin/python -m pytest -q tests/test_profile_native_coupled_backward.py tests/test_profile_native_coupled_latency.py tests/test_native_latency_comparison.py
```

需要真实 CUDA 可见权限；目录必须不存在。两命令应串行运行，不与训练或重型测试争抢。
本轮诊断测试 **21 passed，1 条已知 profiler cycle 警告**。
新增父子扣除、缺父项和兄弟区间重叠测试，并在公开完整两事件路径检查出生 timer、
accounting closure、前后权重与初态未变。计时包装退出后恢复实例方法。

证据：

- [机器汇总](../output/native-full-pipeline-benchmark-2026-10-10/summary.json)
- [v400 细分原始结果](../output/native-full-pipeline-benchmark-2026-10-10/v400-fine-timing/result.json)
- [injInner 细分原始结果](../output/native-full-pipeline-benchmark-2026-10-10/inner-fine-timing/result.json)
- [v400 粗测](../output/native-full-pipeline-benchmark-2026-10-10/v400-timing/result.json)
- [injInner 粗测](../output/native-full-pipeline-benchmark-2026-10-10/inner-timing/result.json)
- [诊断入口](../tools/profile_native_coupled_backward.py)
- [诊断测试](../tests/test_profile_native_coupled_backward.py)

生产源 SHA256：`f6dd533258fe83451c301888297a1154cf0b9d5fcdfb9c7ade23432d90bd551b`（未变）。
细分诊断工具 SHA256：`32abe5544f444867c3ad69278cd3eae6ff7382642ea07f5b8a178841bbfc8364`。
粗测工具冻结副本：
`output/native-full-pipeline-benchmark-2026-10-10/profile_native_coupled_backward_before_fine.py`，
SHA256 `e7f0d84988014014237ed7974f998a10c06d78180cfa4b2f008ad72fd40db454`。
权重 SHA256：`056c11c34cb7029b05a0daf7303f1c4f3624e2ec5691de4de4217dd962c4a328`。

## 接下来如何使用结果（本轮不实施）

injInner 优先继续定位／采样与标量回退；v400 先单独剖析首次出生的候选生成与空间确认，
不要把 14.73 s 出生漏出总成本。两组公共的 context 配置准备约 4.7 s，
可以研究在不放松绑定／原子提交守卫的前提下减少重复核验。
密度源和 nonlocal 刷新也值得独立优化；仅增加 batch 或迁移小矩阵无法覆盖这些 CPU 等待。
已有系数缓存微小收益不构成默认启用依据；不改变物理子步／源反馈语义来取得表面提速。
