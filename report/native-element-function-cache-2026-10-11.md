# 单元场函数缓存与编译求值：真实云 A/B（2026-10-11）

## 结论与范围

用户提出的“保存每个单元各场的函数关系”已做成**非默认、可显式选择的实验路径**。
每个 H=1 单元/场保存 16 个双三次多项式系数，值与梯度来自同一函数，几何 Jacobian
也由完整原生几何计算。不是密集点值表，不采用 bilinear 重采样或近似定位。

热查询约快 5 倍（单粒子）、23 倍（2048 粒子）、32 倍（整个初始活云）。
完整自主耦合事件的观察耗时减少 **v400 1.47%、injInner 8.85%**，不是几个数量级。
这是**系数缓存、预展开和编译融合求值的综合收益**，不能全部归因于缓存。
配对次数少、未保证整机独占，不作统计显著性或训练吞吐承诺。

两组完整云、出生/丢失身份、时钟、子步数量和 CPU RNG 对照一致；不是全部浮点
状态逐位一致。injInner 严格历史检查失败，补证确认只有一个已出界粒子的末段步长
差异，相对差 `3.4404e-10`。保留 `success=false` 的原始严格判定，不改阈值、
不清零差异、不把确定性舍入当作抽样统计误差，也不自动启用默认后端。

本轮**未训练、未更新 optimizer、未修改 JOREK/原始数据/roadmap/ADR**。真实基准
使用构造器的固定非零未训练诊断权重，不是此前训练检查点；无未来真实场回授。
真实基准仅前向，不能据此宣称训练吞吐、长时间稳定、预测准确或 C0 验收。

## 实现与接口

- [native_element_functions.py](../src/models/native_element_functions.py)：
  `NativeElementFunctionSource(source, kernel=...)` 包装原生 CPU `batched_v1` 源，
  支持组合当前 `NativeNearbyMarkerSource`；配置登记非默认模式与内核内容指纹。
- [native_element_functions.cpp](../src/models/native_element_functions.cpp)：
  FP64 批量 Horner 求值；显式 `g++ -O3 -fno-fast-math -ffp-contract=off`，
  不用 fast-math/FMA 放宽，不引入线程池。编译显式执行、单独计时，不自动装依赖。
- 全部四 degree、顶点顺序、signed size 均进入系数转换，不把 degree 当成普通
  Hermite 导数、不以 dof0 替代完整原生场。完整九场和五反馈都保存系数，实际查询
  只计算原生 marker 公式消费的通道。
- 每次源 `advance` 新建缓存，事件内不可写快照共有一次系数准备；动态时间采样
  换快照则重建，不跨事件按外部 ndarray 身份缓存。缓存是可重建工作空间，
  不是 marker 物理历史，不保存进物理检查点。每个 wrapper 拒绝并发推进。
- 本轮用全网格系数，未实施“按未来真实 marker 范围激活”。全网格仅 MB 级，
  能覆盖预测轨迹改变和进入新单元，无需额外范围推断。
- public 源入口明确要求 marker `(s,t)` 为 FP64；不静默把原生 FP32 局部坐标的
  基函数算术替换为 FP64。H>1、GPU 粒子源、自定义 mesh 扩展不在候选支持范围。

**未改变**：网格定位/nearby/全局 fallback、ADAS 查询、电荷演化、出生概率、
粒子子步、末段边界算法、OLD 位置沉积、投影滤波与矩阵求解、shard 更新和耦合顺序。
source/shard 仍是 NumPy stop-gradient；Torch 场和 CfC 的跨步图继续保留，不宣称
新增了源/shard 的端到端可微性。值—梯度—几何多项式代数等价，但乘加顺序不同，
**不保证 bitwise 等价**。

## 基准契约

工具：[benchmark_native_element_functions.py](../tools/benchmark_native_element_functions.py)。
5070 Ti 执行 FP64 九场/CfC，CPU 执行完整源与 shard；BLAS/OMP/MKL threads=1，
Torch threads=2，固定 seed=20261011，CUDA 确定性配置。无 optimizer/loss/backward。

共同条件：百万槽位、不裁剪活云、3986 单元、4125 节点、原始流体增量、完整前投影
及独立输运历史；chunk=2048、64 格源宽出生、`jorek_nearby_v1` 和
`local_subcycling_boundary_v2`。后一模式仍是 ADR-0023 的非默认数值近似，
不是 JOREK 固定粒子子步复刻。

原生共同前缀执行一次，再从同一完整状态与 Python/NumPy/Torch-CPU RNG 分叉。
每臂随后执行两次完整 source→field→OLD-field shard→诊断→提交事件，标签不进入
推进。未来 HDF 只读控制标量，控制来源仍是 `recorded_schedule_replay`，不是自主
预测下一控制的输入资格。v400 两轮配对，第二轮反序；injInner 第一轮严格失败后
停止额外重复。补充历史作业只执行每臂一个后续事件。

## 耗时与内存

热查询是同一初始活云位置上的 30 次交替微基准，排除系数准备；不是粒子完整推进。

| 查询粒子数 | v400 native→函数 | injInner native→函数 |
|---|---:|---:|
| 1 | 100.44→19.70 μs，5.10× | 101.19→19.86 μs，5.10× |
| 2048 | 3.265→0.139 ms，23.50× | 3.287→0.143 ms，23.05× |
| 初始完整活云 | 17.800→0.547 ms，32.55×（11168） | 91.438→2.812 ms，32.52×（52604） |

完整事件计时包括每次缓存重建、源、九场、shard、提交；不包含初始化、共同前缀、
输入克隆/RNG 恢复和末态证据计算。下表为算术平均，不混用 median。

| run | native | 函数缓存 | 耗时减少 | 对照事件数/每臂 |
|---|---:|---:|---:|---:|
| v400 | 5.7573 s | 5.6729 s | 1.47% | 4（2轮×2事件） |
| injInner | 42.1859 s | 38.4525 s | 8.85% | 2（1轮×2事件） |

injInner 补证单事件：41.3208→37.8482 s，观察下降 8.40%，不并入上表。
共同前缀 v400 10.455 s、injInner 184.550 s（补证 180.279 s），其 source 分别含
40/81 个 base 子步；不是普通输出切片。后续 v400 两事件各 4 个 base 子步，
injInner 各 16 个。进程总耗时为 72.29/362.53 s，补证 276.11 s。

函数系数 **8,163,328 bytes = 7.79 MiB**（九场＋五反馈＋R/Z），拓扑/size 缓冲
637,760 bytes；准备工作空间估计峰值 15,433,792 bytes。预算不代表整进程内存，
快照、粒子查询输出、原生数组与证据比较另计。主作业峰值 RSS 3.98/4.07 GiB，
包含并存的完整前缀/reference/candidate 状态；不能称为缓存增量。CUDA peak
allocated 约 251.22 MiB，亦不等于整个 GPU 占用。

injInner 第三事件有 22880 次查询、865440 粒子采样，候选查询合计约 0.911 s，
其中 C++ 内核约 0.644 s（嵌套计时，不能相加）。其余定位、电荷、沉积等过程仍
需逐粒子实际推进；本轮没有进一步划分这些剩余成本，不能断言其中谁是第一瓶颈。

## 保真证据与历史差异

初始整云采样相对 L2：

| run | 场值 | 梯度 | 几何 R/Z |
|---|---:|---:|---:|
| v400 | 3.75e-16 | 3.78e-13 | 3.94e-16 |
| injInner | 4.22e-16 | 6.94e-14 | 3.90e-16 |

上述是拼接通道范数；逐通道/方向指标、零参考与最大点态比值保存在 sampling.json，
不能据总体范数声称所有点态相对误差都相同。未补 epsilon，未做误差单位反换算。

两事件末态：完整九场 current/previous 均逐值相同；rawD 与 CfC memory 有更小舍入
差异。这仅说明固定小诊断更新的数值对照，**不是预测准确性证据**。五通道投影
values 最大相对 L2 为 v400 `1.71e-13` / injInner `2.66e-14`；deltas 最大为
`1.17e-11` / `6.20e-13`。持续存活同身份粒子的 R/Z 位置增量相对 L2 分别为
v400 `2.11e-13 / 4.05e-13`、injInner `1.89e-13 / 1.04e-13`。
全槽（含已失活）连续数组的最大相对 L2 为 `1.17e-11 / 3.66e-11`，不包括输运
历史严格逐位要求，也不代表逐点上限。

原严格判定同时要求所有连续比较的整体相对 L2≤1e-8、身份/clock/RNG 一致及
**整份输运历史 bitwise 相同**。v400 通过；injInner 失败。补证没有改该判定：

- level、lifetime、configuration、initialization 全部一致。
- 百万槽 `previous_dt_si` 只有 **槽位 801014** 不同；两臂均已退出 element=-3375、
  edge=1、lifetime=1；其余 999999 槽相同，包括所有仍存活粒子的步长历史。
- 原生末段 `3.025997141278244e-12 s`，候选 `3.025997140237176e-12 s`；对**改变的
  单个槽本身**算相对差 `3.440413670613398e-10`，不是用百万槽范数稀释。
- 退出事件时间存储值相同；末段几何残差均通过既有检查。速度/梯度舍入扰动进入
  既有边界求交、改变最后实际驻留间隔，是与此证据相符的机制解释。
- 它不是缺历史或少执行子步。该槽已失活；原生插入新身份时会显式重置步长/级别
  （[marker_state.py](../src/models/marker_state.py) `advance_window` 出生段），但本轮
  不据此豁免完整状态的严格判定，也不推断任意更长运行均安全。

## 测试、版本与复现

新增 17 项测试，相关回归 **129 passed in 6.36 s**，无 skipped。包括随机完整 cubic、
C/F/负 stride、signed size、独立展开的 Bezier 导数 degree 与物理梯度、普通/动态
rawD、跨单元/丢失、失败回滚重试、预算、同数组变更、两事件物理检查点恢复、nearby
组合、真实事件输出文件契约、细分共同电荷/沉积/history，以及小网格 Torch 两步
损失向初态/CfC 反传。真实基准未执行 backward，保留原 NumPy 梯度边界。

先运行现有系数缓存 22 项建立反馈循环，再运行新测试红灯、实现并验证；舍入相消
用例明确按速度向量范数判断，不能用该断言冒充每个近零分量的相对精度保证。
FP32 `(s,t)` 守卫测试先确认未拒绝，再补明确拒绝及“不提交”验证。

机器证据：

- [v400-r2/result.json](../output/native-element-functions-2026-10-11/v400-r2/result.json)
- [inner/result.json](../output/native-element-functions-2026-10-11/inner/result.json)
- [inner-history/result.json](../output/native-element-functions-2026-10-11/inner-history/result.json)

主作业包装器 SHA256 为 `232fd7b66df2a55d0f68a7cd216134fb85b63728154d047abc422529ce63c328`；
末版仅补 FP32 `(s,t)` 入口拒绝，SHA256 为
`07748f1b48a7eafb0fed314dece6f1bb92771f34baf19fabe1c5332e8868dbf9`。
最终回归针对末版；该末版未另重跑完整真实基准。被测 FP64 求值/缓存实现未变。
所有实测 C++ SHA256 均为
`44cb825d5eeffb4f4a2107a1840a87fe714f90781c4680a670f60bc57f6d42c5`，
二进制 SHA256 均为
`4781d198ea19fc146d1401de3b2751ef77bfd453a82df2b952a1cc082fe838af`。
工具后续补充历史分解/单事件参数，版本分别登记在对应 result.json；三作业各自
检查上述三个依赖在运行期间未变，不声称核验了全仓库依赖锁定。

首个 v400 诊断因独占创建的 progress 文件名重复而失败，保存在
`output/native-element-functions-2026-10-11/v400/`；修复工具命名后完整重跑为
v400-r2。失败目录不纳入正式比较，不覆盖或删除已有产物。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 .torch-cuda-env/bin/python \
  -m tools.benchmark_native_element_functions --run-index 0 --repeats 2 \
  --output-dir output/native-element-functions-recheck/v400
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 .torch-cuda-env/bin/python \
  -m tools.benchmark_native_element_functions --run-index 1 --repeats 1 \
  --forecast-events 1 --output-dir output/native-element-functions-recheck/inner-history
```

injInner 保持严格历史判定时会返回 exit=1，应读取 comparison 的分解而不是将其误报
成物理推进异常。另建新输出目录复现，勿与性能作业并发运行重测试。

使用 `codebase-design` 将系数准备/失效与查询封装在同一源接口之后；使用
`diagnosing-bugs` 固定配对反馈循环、先测基线、保留严格失败并补证。未修改训练入口；
后续若要正式切换，需显式制定连续历史相对误差的验收范围及更长窗口/实际训练权重
的对照，不能把本轮微基准直接替代生产资格。
