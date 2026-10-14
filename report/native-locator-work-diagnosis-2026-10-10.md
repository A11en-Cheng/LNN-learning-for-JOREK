# 移动后定位为何耗时（2026-10-10）

## 结论

主要浪费不是位置加法或 2×2 小矩阵求解，而是 **少量换单元粒子的昂贵回退**，
以及曲面 cubic 逆映射反复求基函数、Jacobian 和阻尼试探的累计成本。

新测两个完整初始源窗口：v400 仅 0.424% 的定位访问回退，
占 58.8% 的定位时间；injInner 仅
0.151%，占 42.4%。
回退后成功者全部改变单元 ID，没有一个最终回到原单元。
这支持首先优化跨单元局部定位，而不是把所有粒子重新做全局映射，
或优先迁移小矩阵求解。

这是诊断，不是定位器修复或新算法速度验收。使用 diagnosing-bugs 方法：
既有固定输入 full-chain profile 作为基线，再对真实窗口统计点访问与回退候选，
先明确粒子计数和函数调用计数不是同一对象。没有训练、optimizer 更新、
生产算法修改或 JOREK 修改。

## 新测范围和结果

入口 [diagnose_native_locator_work.py](../tools/diagnose_native_locator_work.py)。
百万槽位保留，初始活粒子 11168 / 52604，
新增 29 / 56。chunk=2048、Torch CPU threads=2、
seed=1030/1031，默认 batched_v1。

测量固定原生初始场/rawD 的第一个完整 source 窗口。
出生在定位统计范围之外；无神经预测或反向，不是完整四事件训练基准。
初始窗口分别执行 40/81 个粒子子步，含导入后的时钟衔接；
不能用此窗口总时间冒充普通一个流体步。没有读取未来场用于推进。

| 工作计数／计时 | v400 | injInner |
|---|---:|---:|
| 粒子×子步定位访问 | 447880 | 4265327 |
| 回退访问 | 1899 | 6458 |
| 回退访问比例 | 0.424% | 0.151% |
| batch 调用数 | 240 | 2106 |
| 有回退的 batch 调用数 | 239 | 1979 |
| 定位总秒 | 4.9604 | 26.7198 |
| 其中标量回退秒 | 2.9150 | 11.3321 |
| 回退时间占定位比例 | 58.77% | 42.41% |
| 其余批量定位秒（扣回退） | 2.0454 | 15.3877 |
| 完整 source.advance 秒 | 7.0508 | 44.0312 |
| 每次树查询平均候选 | 139.44 | 53.75 |
| 最大树候选数 | 171 | 171 |
| 回退后单元 ID 改变 | 1899 | 6456 |
| 回退后仍为原单元 | 0 | 0 |
| 回退后未找到 | 0 | 2 |

访问次数不是去重粒子数；同一粒子可以在多个子步换单元。
候选数是 KD 查询返回、包围盒过滤前的数量，不等于每个候选都做了 Newton。
“单元 ID 改变”也不是精确记录跨越了哪条边，边界多解仍需后续检查。
不能把 94% 左右的 batch 有回退解释成 94% 粒子回退。

边界包装记录返回结果，不覆盖 _location_geometry/_location_geometries，
不改变原生类型或接受试探点复用守卫。候选树代理仅计数原查询结果。
计时包含统计开销，不能与无统计跑法作提速对照。
输入完整 runtime digest 前后相同，processed_particle_steps 与统计访问数完全一致；
两组 success=true，最终云通过校验。这不是未插桩全云输出 A/B capture 认证。
测试在真实公共 locator seam 比较原／包装输出逐值相等及异常恢复。

证据：
[v400](../output/native-locator-work-2026-10-10/v400.json)、
[injInner](../output/native-locator-work-2026-10-10/inner.json)、
[机器汇总](../output/native-locator-work-2026-10-10/summary.json)。

## 为什么换一个单元会变昂贵

1. 曲面 cubic 几何的 RZ→st 是逆映射，不是最近节点查询。
   原生几何系数固定，但粒子的 st 改变后，basis/Jacobian 仍要重新求值。
   当前以旧 st 为初值反解，最多 25 轮；每轮可尝试 6 个 damping。
   这不是每个粒子都会执行满 25×6。
2. batch 先尝试旧单元，但把 trial st 裁到 [0,1]。
   真正到了别的单元时，在旧单元内部未必存在满足残差的解。
   当前没有在越边时直接切邻接单元；原单元尝试失败才转 scalar。
3. 包围盒不是曲面单元本身。回退访问仍落在原单元保守包围盒内的比例为
   99.47% / 97.88%，
   因而仅靠 bbox 判断不能提前识别大部分此类换单元访问。
4. scalar 采用一个由全网格最大包围半径定义的中心树查询半径，
   再按中心距离排序、旧单元优先、逐候选 bbox/几何确认。
   某些区域返回几十到上百候选；排序和标量 NumPy 小数组调用也有开销。
   旧单元若仍通过 bbox，还可能再次从旧 hint 重试。

源码：[候选构建](../src/models/native_coupling.py:112)、
[scalar](../src/models/native_coupling.py:254)、
[batch](../src/models/native_coupling.py:298)。
本轮没有按粒子统计 Newton 轮数、真实候选尝试数、失败异常或几何条件数，
所以不把全部残余批量时间断言为“25轮耗尽”或“网格病态”。

## 既有四事件 profile 的辅助证据

源码 SHA 与本轮相同；这是完整前缀＋两预测的独立 cProfile，不是本轮单窗口时间。
通过 pstats caller 边区分 marker 回退与 shard／出生定位：

| injInner 四事件非重叠定位分解 | 秒 |
|---|---:|
| batch 几何累计 | 20.4229 |
| batch→scalar 回退累计 | 23.4942 |
| batch 内 np.linalg.solve | 1.2301 |
| 其余 batch 工作（差额） | 4.7052 |
| batch 总量 | 49.8524 |

全场 scalar locate 的 25.4039 s 另含 shard 和出生调用，不能全部算成 marker 回退。
batch 3354 次调用、几何 57522 次调用、batch solve 14994 次调用；
前者是批次／轮次函数调用数，不能解释成每粒子迭代次数。
具体 profiler 目录：
`output/native-cpu-locator-2026-10-09/after-inner-profile/chain/`，
prefix.prof、forecast_event_1.prof、forecast_event_2.prof。
这些累计时间存在 profiler 开销，不混到最新无 profiler 基准里。

## JOREK 的对应机制

一手源码 `particles/examples/marker_spi_source.f90:655–661`：
速度求值、Adams–Bashforth push 后调用 find_RZ_nearby。
`grids/grid_utils/mod_find_rz_nearby.f90:92–161`：
从旧单元/st开始；Newton step 到边界时限步，coord_in_neighbour 转换邻接单元和 st，
在新单元继续求几何。异常等情况才调用全局 find_RZ；无邻接边还涉及 lost 规则。

它不是把旧定位原样沿用，也不是“完全不迭代”，但避免普通换单元直接退到全局搜索。
当前 Python 默认 batch 尚没有这条邻接推进路径。
Fortran 和当前 Python 的容差、步限制、候选次序及边界处理不同；
不得直接复制后宣称逐位等价或已提速。
更完整证据见 [JOREK采样核查](native-jorek-shard-sampling-evidence-2026-10-09.md)。

源码 SHA256：
- marker_spi_source.f90：`4f6e832b107569d7a8a0155ce5995d04d59cc50771d256cdf99c1e692318991c`
- mod_find_rz_nearby.f90：`b1cbbd3b8cbc3250a81132ac9c8f638561d60e832c0113fa13dd984f8999dcf4`
- 当前 native_coupling.py：`f6dd533258fe83451c301888297a1154cf0b9d5fcdfb9c7ade23432d90bd551b`（未改）
- 诊断入口：`b5e84e7c0b455284abeaef5b675a4b58217dc88c80085a62bb5d1e3cbae46532`

## 后续工作建议（本轮未实现）

先做显式实验：旧单元→带方向／坐标变换的邻接单元→严格原生几何确认→全局兜底。
保留原始全局定位器作为 oracle／失败 fallback，保持粒子子步、采场和沉积顺序。
验证元素 ID、局部坐标、lost/身份、轨迹/速度、电荷态、五通道与投影历史的相对误差，
另用独立不带工作统计的计时判断收益；边界多解不能仅凭几何残差认证。

随后再优化 bulk 的 basis/Jacobian 调度与小数组分配。
现有证据不支持“只搬 2×2 solve 上 GPU”就解决总体定位等待，
也不支持通过减小子步数、放松容差或忽略少数回退来表面加速。

## 复现与检查

```bash
.torch-cuda-env/bin/python -m tools.diagnose_native_locator_work --run-index 1 --seed 1031 --output output/native-locator-work-2026-10-10/replay-inner.json
.torch-cuda-env/bin/python -m tools.diagnose_native_locator_work --run-index 0 --seed 1030 --output output/native-locator-work-2026-10-10/replay-v400.json
.torch-cuda-env/bin/python -m pytest -q tests/test_diagnose_native_locator_work.py tests/test_native_locator_latency.py tests/test_native_locator_geometry.py tests/test_native_locator_events.py
```

目录内目标文件须不存在，两个真实 census 串行执行，无 CUDA 需求。
相关回归 **50 passed**；新增两个包装透明性／异常恢复测试通过。
本轮仅新增诊断工具、测试、机器证据和报告索引，不改生产定位或开展训练。
