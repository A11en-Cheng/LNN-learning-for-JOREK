# 现有优化汇总与 CPU 多线程小基准（2026-10-09）

## 结论与范围

**本次真实 v400 首个完整 marker 源窗口中，全部算子分块多线程更慢；仅场采样用
2 worker 可减少约 4.24% 的 source 耗时。没有修改默认后端或启动训练。**
这是 `NativeMarkerSource.advance` 的 CPU 基准，不是完整 shard—场耦合链、九场
网络训练或跨步反向吞吐基准。没有执行神经网络预测、shard update 或 optimizer。
初始化后固定当前原生 U/rawD/前投影与 shard 出生实现，不读取未来真实场作回授。
保留百万槽位、全部活粒子、40 个粒子子步、原生 cubic 求值、默认 Newton 定位、
沉积/滤波投影及历史更新；不能把单个 source 窗口推及全部训练窗口。

本轮使用 `diagnosing-bugs`：采用固定输入差分循环和阶段计时，而非修改默认算法。
这是性能探索，不是新功能故障修复，未强行构造功能红测或做无关缺陷二分。
事先提出采样/ADAS 可并行、定位/fallback 与调度可能抵消收益、刷新受带宽限制
三个可证伪假设。原有公开源事件测试 18 passed；新增入口测试和相关回归见下。
独立子代理只读复核既有报告与新工具，没有并发运行重负载。

## 已有成果：保真路径和实验路径分开

1. 当前含源九场残差/stencil/CfC 与 shard→出生/云→投影→联合场步→shard 接口
   已有有限真实执行证据。NumPy 源/shard 仍有梯度边界；单 rank 出生、nonlocal
   历史兼容及学习开放边界仍是登记的差异，不代表等价于完整 JOREK 或预测达标。
2. **已在默认 CPU 路径的采样优化**：按实际消费通道求值、事件局部只读快照、
   同一 ADAS 表共享查找。ADAS 表本来就一次载入内存，不是每个循环重读文件。
   两组四事件完整状态/时钟/RNG exact；marker 耗时减少 6.74%/7.88%，主链
   减少 3.52%/7.58%，见 [采样报告](native-cpu-sampling-optimization-2026-10-09.md)。
3. **已在默认 CPU 路径的定位优化**：复用接受试探点的几何，不改 Newton 次数、
   阻尼、容差或 fallback。后续独立配对 marker 10.3430→8.6369 s / 83.4992→
   69.3628 s，主链 48.1101→45.8699 s / 108.7326→93.7263 s，分别减少
   4.66%/13.80%，见 [定位报告](native-cpu-locator-optimization-2026-10-09.md)。
   两轮收益不能相加；完整反向对照是 loss/梯度统计，不是所有梯度数组认证。
4. **密集拓扑/场表与 cubic-confirmed 定位仍为实验**：最终布局修正后的 source
   耗时减少 25.46%/13.44%，但 injInner 的末 R 速度相对误差仍约 14.03%。
   不能默认替换。表刷新 28.2–31.2× 是特定刷新子项的旧对照，不是整链收益，
   见 [拓扑缓存报告](native-marker-topology-cache-2026-10-09.md)。
5. **CUDA 空间后端仍未验收**：已有全云运行更慢且存在定位/速度/投影分歧，
   见 [CPU/CUDA 报告](native-cuda-spatial-performance-2026-10-09.md)。反向 profiler
   中定位累计 64.66 s、小 solve 仅 2.31 s，8 次 marker+density 投影仅 0.0777 s；
   单搬矩阵求解不是主要收益来源，见 [反向热点报告](native-coupled-backward-performance-2026-10-09.md)。

旧完整主链计时包含前缀、两事件预测、loss/指标及 backward，不含标签 I/O、
构建和快照证据开销。每组旧对照仅一个配对；不能宣称统计显著或 GPU 跑满。

## 本次基准契约

- CPU：Ryzen 7 9800X3D，8 核/16 逻辑线程，WSL/Hyper-V；NumPy 2.5.2。
- v400 注册初态 4110、启动场 4111，3986 元素/4125 节点、完整九场×四 degree×H=1。
  初始活粒子 11168，共享一次出生 29，全部 1000000 槽位、40 子步，无裁剪。
- 固定外层 chunk=8192，worker 内连续切片、按原顺序拼接。所有候选使用同一
  chunk；不能直接与旧 chunk=2048 或未控制 BLAS 线程的计时混为一谈。
- 用真正 `ThreadPoolExecutor` worker，不以 `torch.set_num_threads` 代替并行测试。
  采样、定位和 ADAS 只读计算可分块；电荷更新、状态写回、沉积/投影与子步时钟
  均保持串行。未并行推进独立事件，也未共享可变 lookup 缓存。
- 进程启动前 OPENBLAS/OMP/MKL/NUMEXPR 线程环境均设为 1；Torch=1。正式 v2
  另只读查询 NumPy/SciPy 两个已加载 OpenBLAS 库，实际线程数均为 1。
  v1 只记录启动环境，未独立查询运行时线程数。
- 一次串行完整窗口预热；各配置三次，第二轮反序。输入克隆、sampler 创建、
  RNG 恢复、worker 启动及 SHA256 比较在 source 计时外；调度和结果拼接在计时内。
  记录进程 CPU time/wall，但该比值不是整机核占用或 GPU 利用率。
- 本 agent 的重测试/计时串行执行；未保证整机其它进程独占。未固定 CPU affinity，
  三次重复不足以建立跨窗口统计结论。预算 8 GiB、600 s；SIGALRM 可中断主线程，
  executor 仍会等待在运行的 worker，**不是硬杀进程 deadline**。

## 结果 A：全部三个只读算子都分块多线程

单位为秒，均为三次中位数；子项中位数不能作为可精确相加的单次父耗时。

| worker | 完整 source | 采样 | 定位 | ADAS | 相对串行耗时 |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.908 | 0.779 | 4.125 | 0.401 | 1.00× |
| 2 | 8.873 | 0.516 | 7.098 | 0.641 | 1.50× |
| 4 | 38.909 | 2.421 | 22.973 | 12.935 | 6.59× |
| 8 | 52.337 | 3.542 | 28.960 | 19.218 | 8.86× |

2/4/8 worker 的平均进程 CPU/wall 约 1.23/1.48/1.50 核，增加 worker 没有按比例
增加有效并行。任务过细、Python/GIL 与调度争用符合这个现象，但本次没有
context-switch/GIL profiler，**不能将具体根因写成已证实**。

原始机器文件 [v400-all.json](../output/native-cpu-threads-2026-10-09/v400-all.json)
完整运行 352.05 s、峰值 RSS 1.876 GiB。13 次含预热的完整输出数组/时钟/metadata
SHA256 和末 RNG 均与串行预热一致。但 v1 错把 `execution.elapsed_seconds` 和
`peak_rss_bytes` 放进诊断 exact 比较，连串行重复都误报 false；原文件
`success=false`、退出码 1 **原样保留，不追改为通过**。v1 未另外保存去资源计数的
物理诊断 digest，因此只认证已记录的输出 components/RNG，不宣称全部诊断通过。
冻结脚本 `output/native-cpu-threads-2026-10-09/benchmark_native_cpu_threads_v1.py`，
SHA256 `24b7284f7fe233c034acbc5aa1d01f014c2741a93abf29246f68799af01d6eb5`。

## 结果 B：正式 v2，只并行场采样

修复比较器：仅排除上述两个资源计数，保留 energy/rad/moments、实际子步数、
处理量与 sampling 诊断；另外加入 cloud group 指纹。没有放宽浮点误差标准。

| worker | 完整 source 中位耗时 | 采样中位耗时 | 完整证据 |
|---:|---:|---:|---|
| 1 | 5.814 s | 0.753 s | exact |
| 2，仅采样 | 5.567 s | 0.522 s | exact |

source 耗时减少 **4.24%**，采样子项减少 **30.72%**；定位仍约 4.05 s，占大头。
七次含预热的完整数组 shape/dtype/bytes SHA256、group、反馈历史/metadata/时钟、
RNG 与物理诊断均 exact，输入不变。它们与 v1 的共有完整输出 components 也一致。
有定义的连续分量相对误差为零，零参考处也无变化，不做 SI/Alfvén 反换算。

机器文件 [v400-sample-only.json](../output/native-cpu-threads-2026-10-09/v400-sample-only.json)：
`success=true`；65.20 s、峰值 RSS 1.385 GiB；每窗口 80 次调用/每算子处理
447880 粒子步。RSS 不可与完整模型训练内存比较。
冻结实测 v2 脚本 `output/native-cpu-threads-2026-10-09/benchmark_native_cpu_threads_v2_measured.py`，
SHA256 `89b38713bcd658678718a52cb879220751fc0d38f0718475758a02093dcb78a6`。

## 场表刷新与验收

另在 v1 源窗口后测 33×33 的固定拓扑缓存刷新，九场表 312534288 bytes，133 个
互不重叠的元素块，同缓冲更新，预热后各五次。1/2/4/8 worker 中位数为
**21.43/20.56/21.60/22.86 ms**，输出与串行 cached-matmul 数组逐值一致。
这些重叠的短时计时不支持明显线程扩展收益。未测旧 einsum，本轮不能再认领
28×；也没有通过这个表刷新实验验收 bilinear 定位/轨迹精度。

最终工具还补强了严格原生 mesh 守卫、刷新 equality 纳入 success，以及 deadline
措辞。这些是防误用修订，未重跑真实基准；实测绑定上述冻结版本。
相关回归 **69 passed in 1.45 s**，其中新增公开完整源事件测试 10 项：实际多 worker、
普通/跨元素/丢失、失败回滚及改变输入后的重试；不以私有采样微测替代入口验收。
工具/测试编译及 `git diff --check` 见本轮最终执行。默认 `native_coupling.py` 未改，
SHA256 保持 `f6dd533258fe83451c301888297a1154cf0b9d5fcdfb9c7ade23432d90bd551b`。

## 下一步建议（本轮未实施）

1. 保留串行定位/ADAS；只并行采样为小收益候选，先复核 injInner 和完整自主耦合+
   BPTT/恢复一致性，再决定是否接训练后端。本轮没有这两项证据。
2. 优先保留默认定位轨迹，缓存原生布局元素系数/静态拓扑，减少重复 gather，
   并评估编译批量定位内核减少 Python 开销，不先改变定位容差或物理子步。
3. 如要更大并行吞吐，再另立小规模进程/编译后端实验，保持只读几何共享、
   确定性沉积顺序和内存预算。不能以本次线程失败推断多进程一定成功。

复现使用新的输出文件名；可由调用者添加外层 `timeout` 硬进程限制：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .torch-cuda-env/bin/python -m tools.benchmark_native_cpu_threads \
  --run-index 0 --workers 1 2 4 8 --repeats 3 --refresh-repeats 5 \
  --output output/native-cpu-threads-recheck/v400-all.json

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .torch-cuda-env/bin/python -m tools.benchmark_native_cpu_threads \
  --run-index 0 --workers 1 2 --kernels sample --repeats 3 --refresh-repeats 0 \
  --output output/native-cpu-threads-recheck/v400-sample-only.json

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .torch-cuda-env/bin/python -m pytest -q tests/test_native_cpu_threads.py \
  tests/test_native_lookup_experiment.py tests/test_native_marker_sampling.py \
  tests/test_native_locator_events.py tests/test_native_atomic_interpolation.py
```
