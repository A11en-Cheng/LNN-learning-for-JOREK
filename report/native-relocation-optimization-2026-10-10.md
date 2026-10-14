# 移动后重新定位：保序回退与调用内几何复用（2026-10-10）

本轮只优化 CPU 定位工作，不训练、不更新 optimizer、不改 marker 子步或 JOREK。采用 diagnosing-bugs：先建立完整 `Source.advance` 的确定性工作复现，再跟踪必要/重复几何，回归与完整真实云严格 A/B，最后将独立无 profiler 计时与 profiler 工作诊断分开。

## 复现与判断

此前[真实工作普查](native-locator-work-diagnosis-2026-10-10.md)记录：v400/inner 仅 0.424%/0.151% 的定位访问回退，却占 58.8%/42.4% 的定位耗时。此前[邻接实验](native-neighbor-locator-experiment-2026-10-10.md)虽快，却在 inner 改变了一个最终元素编号；那个实验仍不接默认。

本轮毫秒级场景：粒子刚穿过两个相邻 cubic 曲面单元的公共边，但仍在旧单元保守包围盒内。入口：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .torch-cuda-env/bin/python -m pytest -q tests/test_native_relocation_work.py
```

初始红灯实测 `near-edge relocation repeated geometry 22 times`。初始试设预算 12；针对调用跟踪最小化后发现实际需要 13 个不同几何点（旧单元 9 个，最终候选 4 个），另 9 次是批量算过后标量重新计算的同一点。因此工作回归改为经跟踪确认的 13 次上限，并另用冻结原方法断言原来确为 22 次；不是放宽几何残差或物理误差门槛。优化后为 13 次，完整源事件状态与冻结原路径逐值相同。

向用户说明并检验的三项假设：旧单元批量/标量重复；夹紧后的阻尼试探重复；全局候选先全部排序。跟踪显示六级阻尼的坐标虽近却不相同，不能把它们全部视为同一点并删掉。确认可消除的是跨批量/标量路径的精确重复，以及包围盒不可能命中的候选距离计算。保留 25 次上限、原六级阻尼和 `1e-10` RZ 容差，不跳过 Newton，也不以“批量失败”推断标量必失败。

## 两个透明优化

- **先包围盒筛选，后距离排序。** `native_location_candidates.py` 按原 `1e-9` 包围盒容差批量排除不合格候选；幸存者仍用原逐元素 `np.linalg.norm` 排序键、稳定 tie 次序与旧 hint 优先级。没有改为最近节点/邻居优先，也不缩小全局搜索半径。随机 500 组候选对照、边界 nextafter、重复候选、缺失 hint、ties、非有限 fallback 与输入变化均有测试。100 个候选只保留 2 个时，距离键求值从 100 次变成 2 次。
- **仅本次批量调用内复用几何。** 保存初始几何和部分边界 trial 的 `(element, float64 st bytes) → RZ/Jacobian`；标量回退仍按原候选/初值/阻尼/收敛路径执行，只读取已有的相同点计算结果。不复用场值、不冻结 Jacobian、不跨步保存地图。边界缓存上限 32,768 条，另有至多一个 chunk 的初始候选数据；满后正常重算，不改物理结果。

缓存仅用于未覆写的原生 CPU 网格、float64 st 和 C-contiguous float64 几何；子类、实例几何/locate 覆写和窄 st 类型保留原行为。测试保留 stateful 异常/回滚语义、整数/float32 hint 写回，以及网格变化后的下一事件。标量/批量几何的 bitwise 对照和完整真实 A/B 共同检验复用前提；这不是对任意第三方 geometry evaluator 的等价承诺。

修改只涉及 `NativeSourceMesh` 的搜索调度与新 helper；原接受 trial 复用、完整 cubic 求值、ADAS、沉积、filtered 投影及全部物理事件保持原算法。定位配置语义仍为 `bernstein_bounds_damped_newton_hints_v1`；这是相同数值路径的工作消除，不引入新的邻接/近似定位模式。

## 完整真实云 A/B

工具：[benchmark_native_relocation.py](../tools/benchmark_native_relocation.py)。冻结参考是 `tests/native_relocation_reference.py` 中本轮前的两方法，**包括上一轮已有的 accepted-trial 几何复用**，不是更老的 full-point 慢版本。单线程诊断作用域临时替换 class 方法，保持原 native-type 守卫；退出/异常恢复，禁止 Python 并发和多 Torch CPU 线程。该机制只用于 benchmark，不用于生产。

每组保留 100 万槽，固定一次生成的全部出生云，不裁剪出生数；64 指每最小源宽的格数，不是 64 个粒子。同一初态、字段、rawD、出生和 RNG；每模式一次预热后两轮反序配对，再分别跑 profiler。构建、出生、输入克隆、比较及 profiler 不计入正式推进时间。

当前已经完成 v400：初始活粒子 11,168，出生 29，40 个粒子子步，447,880 次粒子×子步访问。正式计时 baseline 为 6.23451/6.24176 s；candidate 为 4.66187/4.66331 s。中位数 **6.23813 → 4.66259 s，减少 25.26%**。六个未插桩推进和两个 profiler 推进全部严格一致；原输入/其它上下文/源码前后不变。进程峰值 RSS 2.430 GiB，包含完整比较的 longdouble 临时量，不能当作缓存增量。

injInner 也完成：初始活粒子 52,604，出生 56，81 个粒子子步，4,265,327 次粒子×子步访问。baseline 正式计时 43.08656/43.35614 s；candidate 为 38.00121/38.29668 s。中位数 **43.22135 → 38.14894 s，减少 11.74%**。全部八次推进也严格一致，包含旧邻接实验曾分歧的这个真实初态；本轮没有启用那套邻接算法。进程峰值 RSS 2.372 GiB。两组均确认 native float64/C 几何缓存资格，且源码/输入/RNG 检查通过。

| 首个完整源窗口 | 原方法中位数 | 当前方法中位数 | 减少 |
|---|---:|---:|---:|
| v400 | 6.23813 s | 4.66259 s | 25.26% |
| injInner | 43.22135 s | 38.14894 s | 11.74% |

该表是本轮相同 CPU 环境的直接配对，不将旧不同出生/配置/线程报告的时间当作基线。

`_compare` 比较全部槽上的全部 marker 数组、alive、五通道 values/deltas 各完整轴、身份/metadata、clock、gather、物理诊断与所有 RNG 状态；要求逐值一致，不是哈希相同就算通过。位移/速度增量及各通道相对误差同时输出。只有运行时间、进程峰值这类非物理诊断排除在 exact 比较外。每模式实际 `Source.advance` 内核也执行；没有用保存的输出代替候选计算。

## profiler 与边界

独立 profiler 中 `locate_batch` 累计时间，v400 为 5.66079 → 3.52857 s，inner 为 30.88017 → 24.77386 s。父项含全部回退，不能再加 scalar 子项。当前候选通过私有 `_locate` 回退；profile JSON 中公共 `locate` 为零**不意味着没有标量回退**，该工具只导出两个公开入口的函数计数。profiler 时间带插桩开销，不与未插桩 Source.advance 时间混成加速指标。

两组首窗包含注册启动后的时钟衔接，不代表普通流体步；固定字段诊断也不是无真实场回授的自主九场预测。无 GPU 迁移、无训练、无完整训练吞吐加速主张。NumPy 源的梯度边界不改变。

## 复现、验证和证据

真实 benchmark 串行执行，环境 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`，Torch CPU threads=1、chunk=2048、seed=1、64-cpw、出生提议上限 1,000,000、单作业 600 s/8 GiB。SIGALRM 是主线程协作预算，底层 native 调用可能推迟中断，不是 watchdog 硬杀。两样本中位数仅为本机观测，不宣称普适速度或显著性。

```bash
.torch-cuda-env/bin/python -m tools.benchmark_native_relocation --run-index 0 --repeats 2 --wall-seconds 600 --output output/native-relocation-optimization-2026-10-10/v400.json
.torch-cuda-env/bin/python -m tools.benchmark_native_relocation --run-index 1 --repeats 2 --wall-seconds 600 --output output/native-relocation-optimization-2026-10-10/inner.json
```

输出文件独占创建，复跑请换新路径。每文件保存 run、anchor、单位/完整源配置、输入指纹、源码与冻结参考哈希、每次 exact/相对比较、RNG、执行计数、原始墙钟/CPU 时间、独立 profile 和进程 RSS。

最终主代理合并执行 17 个测试文件，**219 passed in 13.27s**；不重复累计此前子集重跑。覆盖完整源事件、原生几何、覆写/失败回滚、窄坐标、邻接实验隔离、空间共享、生产事件/跨步反传、CPU worker、时间历史/投影、出生与 debug seed。编译及 `git diff --check` 通过。

真实跨窗口补充入口为 `tools/validate_native_relocation_history.py`。v400 j4111→j4112 已通过：29→2 次出生、40→4 子步，两个模式各自延续完整云与非零投影历史、连续 RNG；第一窗完整 marker/RNG 序列化后恢复并重做第二窗，全部源状态、五通道、历史、物理诊断和 RNG 逐值一致。每臂序列化约 173.3 MB，完整槽位无删减。

injInner j5681→j5682 也通过：出生 56→11、子步 81→16。两个独立臂的连续结果及各自序列化恢复后第二窗，全部严格一致且保留非零第一窗反馈历史；每臂序列化 173,333,380 bytes。两组补充输入/其它上下文/源码均不变，进程峰值 2.925/2.940 GiB。验证入口总耗时 35.25/141.12 s 包含构建、出生、复制、比较、序列化等，**不是 source 性能表**。

该补充明确使用记录的当前场/rawD/shard/源体积，不执行流体 NN 或 shard 运动更新。首窗口经 `_initial_pairing` 认证后保留既有逻辑启动时间，并登记实际 H5 时间差（本次两组 `recorded_minus_logical_si` 均为 0）。接口另外测试同一认证事件的微小舍入差与大幅错误配对的拒绝；只有验证同一事件之后才允许源体积时间标签绑定到既有逻辑事件时间。普通第二窗仍执行 restart、单位、步长和未来体积守卫。不修改源体积数组，不放松生产未来输入校验。该序列化测试仅是 marker 与 RNG 的 trusted in-memory pickle 恢复，不冒充完整神经模型训练检查点。

```bash
.torch-cuda-env/bin/python -m tools.validate_native_relocation_history --run-index 0 --seed 1 --output output/native-relocation-optimization-2026-10-10/v400-history.json
.torch-cuda-env/bin/python -m tools.validate_native_relocation_history --run-index 1 --seed 1 --output output/native-relocation-optimization-2026-10-10/inner-history.json
.torch-cuda-env/bin/python -m pytest -q tests/test_native_relocation_work.py tests/test_native_location_candidates.py tests/test_native_locator_events.py tests/test_native_locator_geometry.py tests/test_native_locator_latency.py tests/test_native_neighbor_locator.py tests/test_native_shared_spatial.py tests/test_native_marker_sampling.py tests/test_native_production_torch_integration.py tests/test_native_source_batch.py tests/test_native_cpu_threads.py tests/test_native_source_time_integration.py tests/test_native_projection_n0.py tests/test_native_birth_positions.py tests/test_native_debug_seed.py tests/test_benchmark_native_relocation.py tests/test_validate_native_relocation_history.py
```

机器证据：[v400 benchmark](../output/native-relocation-optimization-2026-10-10/v400.json)、[inner benchmark](../output/native-relocation-optimization-2026-10-10/inner.json)、[v400 history/resume](../output/native-relocation-optimization-2026-10-10/v400-history.json)、[inner history/resume](../output/native-relocation-optimization-2026-10-10/inner-history.json)。这四个作业均成功完成，无物理执行失败，原始失败工作预算和后续回归过程保留在本报告。

后续应先复测完整训练事件的非重叠耗时，再决定是否优化 bulk 的 cubic basis/Jacobian 调度。不能从这里的 source 窗口减少百分比直接推算训练加速，也不能把有离散分歧的旧邻接实验作为本轮默认。
