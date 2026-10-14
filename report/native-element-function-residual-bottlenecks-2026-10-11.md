# 单元场函数缓存之后的剩余瓶颈（2026-10-11）

## 结论

对当前非默认函数缓存路径补做完整自主耦合更新：

- **injInner：38.229 s/普通事件，marker 源推进 31.736 s，占 83.0%。**
  其中缓存场查询仅 0.890 s，已不是主要瓶颈。函数剖析中最大项为
  **ADAS 系数插值，其次为移动后定位，再其次为 OLD 位置沉积**。
- **v400：5.836 s/普通事件，marker 源推进仅 0.386 s，占 6.6%。**
  主要成本在源体积/诊断刷新、独立 shard 密度源、重复场接口准备和出生概率表准备。
- 电荷态显式更新和最终 marker 投影矩阵求解不是当前主要瓶颈；不能据此建议
  把“marker 解矩阵”搬到 GPU，因为 marker 运动和电荷推进并不是这里的大矩阵求解。

需纠正“查到下一位置场信息已经很快”的范围：本轮缓存加速的是已知
`element,(s,t)` 后的场值/梯度求值。**移动后的 R/Z → element,(s,t) 反解没有使用
该编译缓存**；nearby 仍在原生几何接口中求值、迭代和跨边。因此缓存场函数并没有
同时消除移动后定位或沉积时的基函数/几何计算。

本轮只诊断，**没有训练、optimizer、backward、未来真实场回授或生产算法修改**。
没有切换默认后端；沿用 ADR-0023 的非默认局部细分/末段驻留近似，不能将其开销
视为原生 JOREK 固定粒子步实现的开销。此前缓存与 native 之间的末段历史舍入差异
仍见 [缓存 A/B 报告](native-element-function-cache-2026-10-11.md)，本轮不豁免该差异。

## 测量契约与透明性

工具：[profile_native_element_function_latency.py](../tools/profile_native_element_function_latency.py)。
使用真实完整百万槽云，不裁剪活粒子，5070 Ti 上执行 FP64 九场/CfC，源和 shard 在 CPU；
chunk=2048、64 格源宽出生、nearby、`local_subcycling_boundary_v2`、固定 seed=20261011，
BLAS/OMP/MKL threads=1，Torch threads=2。

每组先执行一次 native 共同前缀，再从其同一完整状态及 RNG 分叉，回放同一个普通
耦合事件：①仅大阶段透明计时；②同一路径加入 cProfile。两个回放都使用函数缓存、
构造器的固定非零**未训练诊断权重**，不是训练检查点。未来 HDF 只读控制时间标量，
仍为 `recorded_schedule_replay`，没有读取未来场来替换预测。

两组 `success=true`；计时/剖析回放的完整 cloud、输运历史、九场/rawD/memory、
feedback、shard、时钟及 Python/NumPy/Torch/CUDA RNG 分量指纹一致。
共同前缀及模型权重未变。此处证明的是**诊断仪器没有改变同一缓存路径的结果**，
不是 native 与缓存逐位等价、预测准确或训练吞吐认证。

共同前缀：v400 10.274 s、injInner 179.734 s；均单列、不计入下表普通事件。
每组仅一个后续物理事件的两种计时回放，不是多窗口统计；未保证整机独占。

## 普通完整事件：未启用 cProfile 的阶段计时

下表各阶段按直接子计时扣除，互不重复；总 wall 与阶段树根差小于 0.001 s。
缓存场查询是 marker 源推进的子项，**不能再与 marker 行相加**。

| 阶段 | v400 / s | injInner / s |
|---|---:|---:|
| marker 源推进（含采样、细分、电荷、移动、定位、沉积、投影） | 0.3856 | 31.7363 |
| marker 出生（含概率表准备、初速/定位） | 0.8478 | 1.9219 |
| 独立 shard 密度源 | 1.2342 | 1.1883 |
| shard 体积/非局域等诊断刷新 | 1.3065 | 1.4500 |
| 场求解接口准备（本事件调用 4 次） | 1.0177 | 1.0053 |
| shard 状态更新 | 0.4945 | 0.3902 |
| 九场/CfC 网络前向（边界同步） | 0.0633 | 0.0726 |
| 状态校验，不含已单列的接口准备 | 0.1544 | 0.1547 |
| 耦合输入准备 | 0.0017 | 0.0016 |
| 其他事件工作 | 0.3302 | 0.3077 |
| **完整事件 wall** | **5.8361** | **38.2290** |
| 缓存场查询（上面 marker 行内部，仅作说明） | 0.01484 | 0.89004 |

injInner 该事件执行 16 个 marker base 子步、23236 个推进批次、865447 粒子子步访问，
其中细分访问 22820；场/ADAS 采样回调 23167 次，结束后 52668 个活粒子、一次确认出界。
v400 执行 4 个 base 子步、24 个推进批次、44796 粒子访问，无细分、无出界。
访问次数不是独立粒子数量。二者的批处理结构差异不能只用活粒子数量解释。

## injInner 内部：函数剖析排序

**以下都是 cProfile 插桩后的时间，不是上表的普通 wall 时间。**
插桩后的完整事件为 48.033 s、marker 源推进为 38.265 s；不能把这些函数时间
直接填回未插桩的 31.736 s，也不能把嵌套父子行相加。

| 不重叠的主要调用边界 | 调用数 | inclusive / s | 占插桩 source / % |
|---|---:|---:|---:|
| ADAS provider：SCD/ACD/PLT/PRB | 23167 | 12.9858 | 33.94 |
| 移动后 nearby 定位外层边界 | 23235 | 9.9760 | 26.07 |
| OLD 位置 `deposit_batch` | 23236 | 4.0493 | 10.58 |
| 缓存场查询 | 23167 | 1.4005 | 3.66 |
| 电荷态 `charge_step` | 23236 | 0.3305 | 0.86 |
| 最终 marker 投影（按 `advance_window` caller 分离） | 1 | 0.01384 | 0.04 |

前三项合计约为插桩 source 的 70.6%。具体证据：

1. **ADAS 慢在插值，不在读文件。** 表在 `NativeNeonAtomicRates.__init__` 中已存于
   内存；`__call__` 按当前温度/电子密度查询四张表。本事件产生 92668 次
   `interpolate_charge_table` 调用，累计 12.3804 s，其中函数 self 10.0381 s。
   当前实现仍逐电荷行执行 NumPy 索引/算术，再 `stack`；大量小批次放大该开销。
   该 12.3804 s 是 provider 12.9858 s 的子项，不能相加。
2. **移动后定位仍重复几何计算。** `NativeNearbyLocator._geometry` 47568 次，
   其下原生 `_location_geometries` 合计 7.1164 s，是 9.9760 s 定位的子项。
   缓存 wrapper 只替换 `_marker_points`，没有替换 locator 的几何接口或反解规则。
3. **沉积也未走新场缓存。** `deposit_batch` 每批调用 `_points_geometry` 获取基函数，
   本事件该子项 3.3116 s，包含在沉积的 4.0493 s 中。每批还新建
   `[5,16419]` 完整 RHS，再在 `advance_window` 中做 `rhs + contribution`。
   按数组形状和调用数推算，单是这些新建 RHS 的累计数组容量约 14.21 GiB，
   不是峰值 RSS，也不是实测内存带宽；全数组累加自身尚未独立计时，不能把
   沉积所有成本归为内存分配。当前最直接的实测子项仍是基函数/几何计算。
4. **调度和数组操作也是剩余成本。** `transport_batches` inclusive 20.5700 s
   包含采样/ADAS与边界检查，不能当成额外 20.57 s；`advance_window` self
   1.4524 s 也不是纯 RHS 累加计时。以上分解没有把这些父级误算为独立瓶颈。

v400 的内部定位虽占其插桩 source 约 33%，但 source 在全事件中仅 6.6%；
继续只优化 marker，不能大幅改善该组总 wall。其四次场接口准备重复进行 context
验证、深拷贝与指纹计算；剖析记录 `canonical` 约 473 万次递归调用，说明存在
实现层的重复工作，不意味着可以删除配置一致性守卫。

## 接下来应该优化什么（建议，未实施）

对 injInner 优先顺序：

1. **融合/编译 ADAS 的四表、全电荷态查询**，保持各表自身轴、clipping、单位和
   PLT0:9/PRB1:10 映射；先走代数保真路径，不先改粗糙插值表或减少物理子步。
2. **把静态几何函数缓存延伸到定位的 R/Z/Jacobian 查询**，保留当前邻接、Newton
   容差、边界与失败语义。它不是仅继续细化现有场查询表；现有 locator 未消费缓存。
3. **复用沉积基函数工作并减少全 RHS 临时数组**，明确保持 OLD 位置、时间权重和
   累加语义。跨批聚合会改变浮点求和顺序，需比较身份、历史、轨迹增量与五投影，
   不能只比较总场范数或承诺逐位一致。

对 v400，优先审查静态 context/求解接口准备复用及 shard 源体积/密度计算，保留
完整校验和失败回滚；不是把所有成本转移到 GPU 就自动解决。

本报告没有测新的线程数、GPU marker 后端或 backward，不报告这些方案的收益。
上述优先级由现有调用量和剖析支撑，但没有未经实现验证的加速倍数承诺。

## 复现、资源与交付

机器产物：

- [v400/result.json](../output/native-element-function-residual-profile-2026-10-11/v400/result.json)
- [inner/result.json](../output/native-element-function-residual-profile-2026-10-11/inner/result.json)
- 各目录 `boundary_timed.json`、`profiled.json` 为阶段分解；`profiled.prof` 为完整
  caller/callee 原始剖析，`profiled-profile.json` 为摘要。最终投影行使用原始 caller
  分离 marker 投影与独立密度源投影，没有把二者混在一起。

进程 wall 37.715/278.624 s，峰值 RSS 3.254/3.266 GiB；CUDA peak allocated
均约 247.87 MiB。这些包含初始化、共同前缀、两次回放及证据计算，不能视为单事件
延迟或缓存额外内存。`src` 下 Python、缓存 C++ 及本诊断工具运行前后哈希相同，
完整哈希存于 result；没有声称对所有辅助脚本、系统库或硬件状态做了锁定。

新增诊断测试及现有剖析/分叉回归：**10 passed，1 个既有 profiler warning**。
测试验证完整事件、RNG 与计时树闭合；不等于已验证新的物理或训练能力。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 .torch-cuda-env/bin/python \
  -m tools.profile_native_element_function_latency --run-index 0 \
  --output-dir output/native-element-function-residual-profile-recheck/v400
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 .torch-cuda-env/bin/python \
  -m tools.profile_native_element_function_latency --run-index 1 \
  --output-dir output/native-element-function-residual-profile-recheck/inner
```

输出目录必须不存在；不覆盖旧产物。本次只增加诊断工具、测试、报告及索引，未修改
JOREK、原始数据、roadmap、ADR 或模型/源的生产实现。
