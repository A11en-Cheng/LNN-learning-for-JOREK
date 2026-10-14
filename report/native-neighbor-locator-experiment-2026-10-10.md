# 原生曲面邻接定位实验（2026-10-10）

## 结论

**隔离实验已实现；v400 source 耗时减少 38.60%，injInner 减少 22.48%，
但 injInner 有 1 个 marker 最终元素编号不同，验收失败，因此不接默认链。**
injInner 全云速度增量相对 L2 差异 3.80%、位移增量 0.144%，不能把它仅当作
浮点舍入或统计误差忽略。v400 同口径差异分别 3.56e-11 / 1.76e-12。
未训练、未宣称物理预测精度、真实多事件完整链或训练吞吐改善。

## 范围与实现

本轮实现一个显式可选的 `NativeNeighborMarkerSource`，未替换默认定位器、
未训练、未更新优化器、未修改 JOREK 或原始数据。实验目标是削减移动后换单元
时的全局搜索，而不是删除物理子步或改用插值场表。

- [实验 Adapter](../src/models/native_neighbor_locator_experiment.py) 保留默认
  原单元批量 Newton。只有未收敛点先尝试拓扑邻居，失败再调用原全局定位。
- 邻接由有向顶点边构造，保存相邻元素、对应边和方向；不按坐标距离猜邻居。
  缺边、非流形边、角点歧义、近边根和耗尽 hop 预算都保留原全局策略。
- Newton 预测只用于选边；把边界坐标按方向转换到邻居，再调用原生单元内
  批量 Newton 做完整 cubic 几何确认。仍为 25 次、六级阻尼、RZ 范数小于
  `1e-10`，没有放宽容差、冻结 Jacobian 或减少粒子步。
- 本轮没有把跨边转换直接融合进旧单元 Newton 主循环：旧单元上的夹紧试探
  成本仍在。也没有优化 shard 定位、出生、ADAS、电荷求解、沉积或投影。
- 拓扑与几何校验快照常驻内存，字段值仍按事件更新。每事件用本地浅复制
  native mesh/source，更换 fallback，不修改共享对象或覆盖几何求值方法，
  因而保留原生接受试探点几何复用。锁禁止同 Adapter 并发推进。
- 实验只支持未覆写的原生 CPU `batched_v1`、float64 marker 局部坐标和静态
  网格；配置或网格变化须重建 Adapter，失败不部分提交。实验版本与几何
  内容进入配置指纹，`production_qualified=false`，不得沿用默认模式检查点。
  缓存可由网格重建，不是需要保存的 marker 物理状态。

## JOREK 一手依据及差异

只读源码位置：`/home/ac_desktop/JOREK_code/jorek`。

- `particles/examples/marker_spi_source.f90:655–661`：计算 VBpsiU、
  Adams–Bashforth 移动后，调用 `find_RZ_nearby`，携带旧元素与局部坐标。
- `grids/grid_utils/mod_find_rz_nearby.f90:92–161`：原单元 Newton 预测，
  将越界步限制到边，再转换邻居坐标；负邻居进入全局查找。
- `grids/mod_neighbours.f90:140–199`：从边顶点顺序判断方向，沿边坐标必要时
  `x -> 1-x`，按邻居边还原 `(s,t)`。
- Fortran 的收敛阈值为平方残差 `<1e-24`、最多 8 次；无邻居的丢失规则与
  当前 Python 不同。本实验沿用 **当前 Python** 的容差和全局边界处理，
  不是逐算术操作复刻 JOREK，也没有声称 JOREK 级轨迹一致性。

## 验证与真实基准

工具：[benchmark_native_neighbor_locator.py](../tools/benchmark_native_neighbor_locator.py)。
两组分别使用注册 post-startup 初态 4110/4111 和 5680/5681；每组有
3986 个元素、4125 节点、16419 个共享自由度、百万 marker 槽位。完整保存
11 电荷态、V_prev、身份、五通道投影及 deltas；没有裁剪云或用新的场反馈
替代固定初态。这里是首个 source 窗口，包含注册启动场区间，不是稳态流体步
或完整训练窗口的耗时。

| run | 初始活粒子 | 出生 | 子步 | source baseline 中位数 | neighbor 中位数 | 耗时减少 |
|---|---:|---:|---:|---:|---:|---:|
| v400 | 11168 | 29 | 40 | 6.632488 s | 4.072030 s | 38.60% |
| injInner（离散契约失败，不可接默认） | 52604 | 56 | 81 | 43.573885 s | 33.779507 s | 22.48% |

每模式两个计时样本，不建立显著性或通用加速结论。v400 baseline 为
6.678040 / 6.586936 s，candidate 为 4.192106 / 3.951954 s。
injInner baseline 为 43.431235 / 43.716536 s，candidate 为
33.913014 / 33.646001 s；候选的契约失败，因此这些只是诊断耗时，不是
可部署的合格加速结果。
v400 另跑计数窗口：240 个 batch、447880 粒子×子步访问；1900 次邻接
接受，0 次全局兜底。这里是访问次数，不是唯一粒子数；不能与不同 RNG 的
前轮 census 1899 次直接宣称逐点对应。默认路径没有插桩统计 scalar 次数。

以下相对 L2 来自同一初态的完整物理窗口，而非单次 geometry 测试：

| 分量 | v400 | injInner（失败候选） |
|---|---:|---:|
| 末位置信息 x（活粒子） | 8.98e-16 | 2.35e-6 |
| 位移增量（初始且末存活的同身份粒子） | 1.76e-12 | 1.441e-3 |
| 局部坐标 st（活粒子，异元素时不代表同坐标系） | 2.83e-10 | 4.521e-3 |
| V_prev（活粒子） | 1.08e-11 | 4.033e-3 |
| 速度增量（持久活粒子） | 3.56e-11 | 3.801e-2 |
| 完整 P_imp（活粒子） | 1.61e-13 | 4.34e-8 |
| 投影 values / deltas（各自全部轴） | 3.98e-11 / 3.98e-11 | 2.010e-4 / 2.010e-4 |

五通道相对 L2 分别如下，不以通道间单位不同的合并范数代替逐通道验收：

| 原生 channel | v400 values/deltas | injInner values/deltas |
|---|---:|---:|
| 0 | 2.965e-11 | 1.636e-6 |
| 1 | 4.545e-11 | 5.088e-6 |
| 2 | 3.974e-11 | 2.000e-4 |
| 3 | 4.181e-11 | 2.622e-4 |
| 4 | 4.879e-11 | 2.612e-4 |

两组当前初态是注册零反馈历史，故本窗口 values 和 deltas 误差相同；不能据此
认证真实非零跨窗口投影历史。v400 局部坐标逐点最大相对差异为 5.97e-6，
其最大差异除参考最大幅值为 1.63e-8；投影逐点相对最大为 6.42e-7。
这些尾部不隐去，但不能将局部坐标的相对比值当作物理位移相对误差。

机器证据：[v400.json](../output/native-neighbor-locator-2026-10-10/v400.json)、
[inner.json](../output/native-neighbor-locator-2026-10-10/inner.json)。
v400 离散元素、生命周期、出生时刻、权重、存活、group、metadata、时钟、gather、
RNG 均严格相同；浮点数组不逐位相同。输入/源码检查通过，v400 作业
84.42 s，峰值 RSS 2.38 GiB。邻接数组加静态校验快照共 1063424 bytes
（1.014 MiB），不是数 GiB 的场表；进程 RSS 包含完整 reference/candidate
状态和全数组比较的 longdouble 临时内存，不等于邻接缓存增量。

injInner 的元素数组百万槽位中 999999 个相同、1 个不同。存活严格相同，
末活粒子均为 52658，不是额外丢失；生命周期、出生时刻、权重、group、
metadata/时钟/gather/RNG 也严格相同。其 2106 batch、4265327 粒子×子步
访问均相同；候选有 6460 次邻接接受和 2 次全局兜底，没有边界歧义计数。
窗口内虽通过了候选残差检查，却发生离散分歧，说明现有歧义守卫不充分。
重复计时与计数 pass 的比较结果稳定重现；没有追加超参/容差搜索。进程
336.50 s、峰值 RSS 2.43 GiB，源码和全部输入检查通过，无超预算。

injInner 投影逐点最大相对差异 2.10，V_prev 为 1.18；这些不能用相对 L2
较小抹去。能量/辐射诊断相对差异为 8.64e-6 / 2.61e-5。当前机器报告没有
保存首次分歧的 slot/子步/candidate trace，因此尚不能把轨迹误差确定归因于
该 1 个最终异元素 marker，也不能断定是网格重叠、多个根或边界舍入哪一种。
明确登记此诊断缺口，不把机制假设写成已证明原因。

已通过公开 `Source.advance` 事件测试，覆盖完整状态/投影、曲面跨边、反向
邻居、两单元跨越与有限预算回退、外边界/丢失、无共享拓扑回退、边界歧义、
两事件历史与 fresh Adapter 恢复、输入不提交及配置守卫。另通过
`TorchRecomputedSourceEvents` 两个完整小规模耦合事件的跨步反传测试；它
验证场/history/memory 图不被断开，不代表 NumPy marker 定位端到端可微。

本轮以完整原生状态相对误差和离散事件契约为准，不能仅凭摘要哈希或几何残差
认证轨迹/投影，也不做 Alfvén 单位反换算。

## 测试与可复核命令

新增 [公开事件测试](../tests/test_native_neighbor_locator.py) 20 项，以及
[基准工具测试](../tests/test_benchmark_native_neighbor_locator.py) 14 项。
基准工具的拒绝用例包括元素/lifetime/group/clock/gather/RNG 分歧、输入被
修改及零参考通道不能加 epsilon 隐藏差异。完整相关回归结果以如下命令为准：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .torch-cuda-env/bin/python -m pytest -q \
  tests/test_native_neighbor_locator.py tests/test_benchmark_native_neighbor_locator.py \
  tests/test_native_locator_events.py tests/test_native_locator_geometry.py \
  tests/test_native_locator_latency.py tests/test_native_marker_sampling.py \
  tests/test_native_production_torch_integration.py \
  tests/test_native_atomic_interpolation.py tests/test_native_source_batch.py
```

上述相关回归 **114 passed（2.88 s）**；编译检查及 `git diff --check` 通过。
回归通过不覆盖真实 injInner 离散失败。

## 源码指纹与复现契约

默认 `src/models/native_coupling.py` SHA256 仍为
`f6dd533258fe83451c301888297a1154cf0b9d5fcdfb9c7ade23432d90bd551b`。
本轮未改变其物理算术或默认选择。

实验模块 SHA256：
`ce694360db24ffefbdd94dd61bc3ce75384930240f92f2a1f17db09443b37f96`。
benchmark 工具 SHA256：
`f1b06a368b07554135ef401ba59bca6cf7cf67b4cc00ab883cad19ae676d51fd`。

JOREK 只读文件 SHA256：

| 文件 | SHA256 |
|---|---|
| `marker_spi_source.f90` | `4f6e832b107569d7a8a0155ce5995d04d59cc50771d256cdf99c1e692318991c` |
| `mod_find_rz_nearby.f90` | `b1cbbd3b8cbc3250a81132ac9c8f638561d60e832c0113fa13dd984f8999dcf4` |
| `mod_neighbours.f90` | `9e1be99b70f3319db5677b1e77d0ab25bbffbf3435ff2d9e1c7c743d86b6f203` |

基准保持百万槽位，出生一次共享，固定初态 U/rawD；恢复同一出生后 RNG。
单线程 BLAS/OMP/MKL/NUMEXPR，chunk=2048，seed=1，max_hops=4，
每模式一窗口预热后两轮反序配对。建立网格/出生/输入复制/比较不计 source
耗时；推进内的 Adapter 验证、Newton、采样、ADAS、电荷、沉积与投影计入。
计数另跑完整窗口，不与计时相混，任何模式都不覆写几何或定位方法计时。

工具 `success` 只表示 shape/dtype/有限性及离散身份、时钟、gather 和 RNG
契约有效，**不是**浮点误差容差门槛或生产资格。完整数组、活粒子分量、相对
初态的位移/速度增量和五通道反馈/历史都有直接数值比较。相对 L2 不加 epsilon；
零参考有非零差异时明确登记未定义及零参考不匹配数。逐点最大相对误差须连同
参考量级解释，不能拿其代替整个投影误差。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  .torch-cuda-env/bin/python -m tools.benchmark_native_neighbor_locator \
  --run-index 0 --repeats 2 --chunk-size 2048 --max-hops 4 --seed 1 \
  --wall-seconds 600 --output output/native-neighbor-locator-2026-10-10/v400.json
# run-index=1 对应 injInner，使用独立 inner.json 输出；工具拒绝覆盖已有结果。
```

单作业预算 8 GiB / 600 s，SIGALRM 是主线程协作终止，底层 native 调用可能
推迟响应，不是独立 watchdog 硬杀。真实性能作业串行，不宣称整机独占。

## 资格边界与下一步

邻接根通过 cubic 几何检查，并不保证它就是默认全局候选排序选择的根；跨边
后的初始坐标也改变了浮点收敛路径。因此默认路径不切换，不把“同一物理
残差阈值”当作“相同离散事件”。两组首窗口不能认证长期自主轨迹或训练吞吐。

下一步首先按真实窗口的元素/存活差异排查候选选择和边界附近的路径敏感性，
必要时回退歧义区域，而不是放宽误差门槛。之后才做真实自主多事件完整状态、
检查点续跑和 BPTT 的 A/B；本轮的小规模跨步反传不代替这个资格。
只有这些通过后才考虑显式接入主链，并重新测整链耗时。出生、密度/诊断准备
等未改变，不能把 source 百分比直接称为整个训练加速百分比。

本轮以 `codebase-design` 隔离 Adapter，`tdd` 在已确认公开事件接口上迭代
测试，`diagnosing-bugs` 固定默认基线与实验对照；没有扩展到训练或物理模型
重设计。子代理只写 benchmark 工具及其测试，重性能作业由主代理串行执行。
