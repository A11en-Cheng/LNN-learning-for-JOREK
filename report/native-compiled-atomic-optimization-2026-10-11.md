# ADAS 查询融合优化与完整耦合 A/B（2026-10-11）

## 结论与范围

本轮完成第一项优化：**显式、非默认的 Ne ADAS 编译插值 provider**。
四张表的搜索及逐电荷双线性算术合并到一次 C++ 调用，原 NumPy provider 保留。
定位几何缓存和沉积优化没有在本轮实现；没有训练或切换默认后端。

两个真实完整百万槽云，每臂两个连续普通耦合事件、两次配对复测（第二次反序）：

| 完整事件，四次观测的平均值 | NumPy ADAS / s | 编译 ADAS / s | 完整事件耗时变化 |
|---|---:|---:|---:|
| v400 | 5.6659 | 5.6838 | 增加 0.32%，未观察到整步收益 |
| injInner | 37.9143 | 27.5225 | **减少 27.41%，约 1.38×** |

injInner 的 ADAS provider 平均 **11.4070 → 1.7282 s**，减少约 84.85%；
源推进整体 **31.5709 → 21.1276 s**。v400 的 ADAS 为 **0.04480 → 0.01975 s**，
原占比很小，所以局部加速不能转化为显著整步收益。

每次两事件末态：全数组相对 L2 最大值为 **0**，离散契约通过；
全部物理分量及 RNG 的逐字节指纹一致，包括非活跃槽的输运历史。
这里的对照是**同一场函数缓存路径只替换 ADAS provider**，不是原始 native
场采样与函数缓存之间的对照，也不豁免此前已登记的末段历史舍入差异。

机器结果：[v400/result.json](../output/native-compiled-atomic-2026-10-11/v400/result.json)、
[inner/result.json](../output/native-compiled-atomic-2026-10-11/inner/result.json)。
不是多窗口精度统计或生产资格认证；仅两个物理事件的重复观测，未保证整机独占。

## 保持不变的物理与数值契约

接口：[native_compiled_atomic.py](../src/models/native_compiled_atomic.py)，
内核：[native_compiled_atomic.cpp](../src/models/native_compiled_atomic.cpp)。

- 继续使用 `NativeNeonAtomicRates(te_ev, ne_si)` 的输入/输出约定；不是读表优化，
  原路径的表本来就已在内存中。
- SCD、ACD、PLT、可选 PRB 各自独立的密度/温度轴、截断端点、
  `searchsorted(side='left')-1` 和电荷行保持不变，不假定四张表的网格相同。
- 输入密度到 cm⁻³ 的乘法及 `log10` 先按原输入 dtype 在 NumPy 执行；
  双线性四项的乘法/加法顺序保持不变。
- `10**coefficient` 仍按原长度、逐电荷行在 NumPy 执行，避免 scalar libm 与
  NumPy SIMD 幂函数的舍入差异。编译参数禁用 fast-math 和 FMA contraction。
- SCD/ACD 的 cm³/s → m³/s 乘数不变；辐射仍为 PLT 电荷 0…9 与 PRB 电荷
  1…10 的原有组合并乘电子密度，不丢弃完全剥离 Ne 的 PRB。
- 没有更改 marker 物理步长、局部细分、运动、电荷推进、出生、出界、
  OLD 位置沉积、滤波投影、shard OLD 场反馈或九场/CfC 更新。

使用已有 callable 接缝新增 adapter，不扩大 `NativeMarkerSource.advance` 的接口；
旧实现同时作为独立数值 oracle。新 provider 自有不可写的 FP64 表快照，
不会因原表后续修改或共享 ndarray 身份缓存而改变结果。真实四表快照为
232,288 B（约 226.8 KiB）；预算只约束持久快照，初始化临时副本及查询输出另占内存。

显式 `build_kernel(destination)` 才编译；导入不自动编译、安装或覆盖已有 `.so`。
配置指纹包含表内容、预算、内核源码/二进制哈希和新版本
`compiled-Ne-log-bilinear-numpy-row-power-v1`。旧 provider 的检查点不能静默恢复
到新配置。诊断工具的同状态分叉是受控实验，不是绕过生产恢复校验的入口。
存储表必须为 FP64；更低精度表明确拒绝，不通过隐式升精度改变轴差的计算。

## 完整事件计时契约

工具：[benchmark_native_compiled_atomic.py](../tools/benchmark_native_compiled_atomic.py)。
v400 从 4110、injInner 从 5680 初始化，先用缓存场函数＋NumPy ADAS 执行共同前缀；
然后复制同一完整状态，恢复相同 Python/NumPy/Torch CPU/CUDA RNG 分叉。
权重为构造器固定非零**未训练诊断权重**；未来 HDF 只读时间控制标量，控制来源为
`recorded_schedule_replay`，不读取未来真实场替换预测。

保留完整百万槽，初始活粒子分别为 11,168 / 52,604，无裁云。普通两事件末态活粒子
分别为 11,201 / 52,686；injInner 包含一次确认出界。每事件仍实际执行完整链条。
v400 四个 marker base 子步，injInner 十六个；injInner 两事件的粒子子步访问
865,447 / 865,440，细分访问 22,820 / 22,528；两臂执行结构相同。

5070 Ti 上 FP64 九场/CfC，marker/shard 在 CPU；chunk=2048、64 格源宽出生、
nearby、`local_subcycling_boundary_v2`，seed=20261011，Torch threads=2，
BLAS/OMP/MKL threads=1。两组顺序运行；没有并发性能任务或 cProfile。
大阶段计时树闭合；ADAS 是源推进的内部子项，不能再次与总耗时相加。
状态复制、对照/指纹、编译、建模和共同前缀不计入普通事件的 wall。

共同前缀为 v400 **9.1618 s** / injInner **161.6380 s**，没有用编译 ADAS
重新测量前缀，不能据普通步收益外推首窗收益。整个诊断进程耗时分别为
72.7806 / 441.6434 s，进程峰值 RSS 3.904 / 4.103 GiB，CUDA 峰值 allocated
251.22 MiB；这些资源包含双臂对照及副本，不是单窗口或训练峰值。

原始事件秒数按各臂的执行次序保留：

| run / provider | repeat 0 两事件 / s | repeat 1 两事件 / s |
|---|---|---|
| v400 / NumPy | 5.8337, 5.6324 | 5.6162, 5.5812 |
| v400 / compiled | 5.6984, 5.6417 | 5.5583, 5.8367 |
| injInner / NumPy | 38.3241, 37.5333 | 38.1648, 37.6349 |
| injInner / compiled | 27.5638, 27.2649 | 27.7477, 27.5137 |

两组的物理全数组、原始场历史、CfC memory、反馈 values/deltas、全部云数组、
输运历史、shard 数组、时钟以及 RNG 分量指纹均一致；模型权重及共同前缀未变。
原始源码集在各任务执行期间的哈希未变，未用放宽阈值获得通过。

## 独立查询与无训练回归

实际 Ne 表，30 次交替顺序查询，逐输出数组精确一致；中位数加速：

| 查询点数 | v400 作业 / × | injInner 作业 / × |
|---|---:|---:|
| 1 | 9.53 | 9.05 |
| 2,048 | 2.28 | 2.24 |
| 52,604 | 2.86 | 2.94 |

最终相关回归 **130 passed in 6.13 s**，新组件/基准测试共 41 项。
覆盖独立旧逐行公式、不同轴/数组布局、真实所有表节点和轴外截断、FP32/FP64 查询、
相邻浮点点位、极小密度、单位和 11 电荷映射、表快照、资源/输入拒绝、
完整 marker 窗口、失败不部分提交、历史序列化续跑。
三线程同时查询仅验证独立输出和一致性，**没有宣称 CPU 并行吞吐收益**。

小网格实际 Adapter 的两步损失可反传至初态和 CfC 参数；新模型保存/恢复继续一致，
错误旧 provider 指纹拒绝。真实大云性能作业 `backward_calls=0`；不把小网格梯度
测试称为真实云全量 BPTT 性能验证。源/shard 的 NumPy 梯度边界没有改变。

复现回归：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .torch-cuda-env/bin/python -m pytest -q \
  tests/test_native_compiled_atomic.py tests/test_benchmark_native_compiled_atomic.py \
  tests/test_native_atomic_interpolation.py tests/test_native_production_radiation.py \
  tests/test_native_element_functions.py tests/test_native_marker_sampling.py \
  tests/test_marker_local_subcycling.py tests/test_native_nearby_locator.py \
  tests/test_native_production_torch_integration.py tests/test_base_coupled_training.py
```

完整 CUDA A/B（输出目录必须不存在；两个命令顺序执行）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .torch-cuda-env/bin/python -m tools.benchmark_native_compiled_atomic \
  --run-index 0 --output-dir output/native-compiled-atomic-replay-v400
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .torch-cuda-env/bin/python -m tools.benchmark_native_compiled_atomic \
  --run-index 1 --output-dir output/native-compiled-atomic-replay-inner
```

版本透明性：完整真实作业测试的 Python adapter SHA256 为
`9d3f6ddde82d01bb8a017b7b29c8c56c2d631340de2c44b860220a798e8aa468`。
作业结束后增加了非 FP64 存储表的拒绝守卫；最终 Python SHA256 为
`36d2040412f31bcfb901ccb71717a9745892d58b05412b806525d5548066d2be`。
该守卫不改变真实 FP64 表的计算或配置内容；最终上述回归含真实所有表节点通过。
**没有在守卫加入后重跑完整大云计时**。C++ 未变，SHA256 为
`4a687ad0c4f3961d782fc7f2f17b91f56e2f886646dd3dcfd5da9cf8f185d428`；
两个作业的 atomic 二进制 SHA256 为
`6c56e19abfa89a9329ec7be392ad942f3afdcc78e70d8b5518788fb2c0cfa08a`。

## 剩余工作

编译 ADAS 后 injInner 的源推进仍为 21.13 s（完整事件约 76.8%），场网络前向
约 0.053 s。下一项优先复用静态 R/Z 与 Jacobian 单元函数到 nearby 定位，保留
原容差、跨边、fallback 与 lost 判据；随后优化 OLD 位置沉积的基函数重复计算
及完整 RHS 临时分配。排序依据是上一轮定位/沉积剖析和本轮 ADAS 减负，
**尚未对优化后的源内部做新一轮 cProfile，不宣称新的精确子项占比**。

v400 仍主要花在 shard 源/诊断刷新及重复场接口准备，不能期望仅优化 marker 内核
就让这组整步提速或让 GPU 跑满。本轮未修改 JOREK、原始数据、roadmap、ADR，
未增加 optimizer 更新；新路径仍是显式选择的实验后端。
