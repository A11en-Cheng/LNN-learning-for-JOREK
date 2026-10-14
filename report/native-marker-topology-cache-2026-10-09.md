# Marker 固定拓扑刷新与原生 cubic 确认试验（2026-10-09）

## 范围与问题

用户要求复用网格子节点拓扑、逐步只更新场值，并核查 JOREK shard 的定位/采样
是否比现有路径更好。本轮延续显式实验，不训练、不更新 optimizer，不改 JOREK、
原始数据、roadmap、ADR 或默认 `src/models/native_coupling.py`。
源码研究由 `research` 子代理写入
`native-jorek-shard-sampling-evidence-2026-10-09.md`；主代理独立进行数值实现与计时。

上轮密集 bilinear 的场值/投影误差小，但 injInner 少数轨迹偏离并不合格。
本轮比较的是相同初态、同出生/RNG 的 Python source 参考与显式实验，
不是 JOREK Fortran 全状态逐值认证，不执行九场网络或 shard 后续更新。
只使用合法当前完整场、rawD、前投影，不读未来场为该 source 窗口回授。

## 固定与更新的边界

静态网格下保持常驻：元素节点索引、signed size、完整原生坐标系数、
子节点 `(s,t)` 的 cubic 基函数矩阵、R/Z 几何表和 KD-tree。
并非每次改变场都重建网格、邻接关系或空间树。

场变化后必须更新数值，否则粒子看见的是冻结旧场。两个互斥实验模式：

- `bilinear`：固定求值矩阵作用于新的元素场系数，以批量 matmul 刷新密集
  子节点值表；已有同形 float64 缓冲原址复用。局部值/梯度仍是上一轮的
  bilinear 近似，刷新优化不能自动修复轨迹精度。
- `native_cubic_confirmed`：只刷新每个元素的完整九场/五反馈原生系数缓存，
  不建立全网格密集场值表。移动后用密集几何表获得 bilinear 逆映射候选，
  一次原生 cubic Newton 校正，再检查内域、正 Jacobian 与 R/Z 残差
  `<1e-10 m`；不满足就回退现有原生定位。之后在当前局部坐标计算完整
  cubic 场值及兼容梯度。没有用单独的低阶梯度表，也没有跳过物理子步。

第二模式不是完全无 Newton，也不宣称原生搜索控制流逐位相同；一次校正是
上限明确的快速候选确认，fallback 才可能迭代。容差与当前 Python 原生参考
一致，不冒充 JOREK nearby Fortran 的更严容差。

缓存 owner 强引用事件只读快照，新事件重新刷新；准备前失效 owner，失败
不提交物理状态，重试不把部分新缓存认作旧缓存。数值缓存会原址改变，不能
并发共享同一个实验 mesh 推进两个事件；旧物理输出不引用此 scratch cache。
每次公共 source 校验仍核对 geometry 内容 SHA，不支持调用中修改几何。

JOREK shard 的实际位置更新无需隐式动力学迭代：在柱坐标转换后用旧速度
显式移动；旧烧蚀率侵蚀半径，随后在新位置定位、采 OLD 场并更新烧蚀率及
rocket 对下步速度的作用。`pellet_module.f90:275–278,310–311,329,360` 与
`mod_jorek_timestepping.f90:418,424` 已由主代理复读。postproc 的
`mod_position.f90:235` 仍调用同一个全局 `find_RZ`，已知新 `(element,s,t)`
的入口才免逆解；不能只把旧 st 直接用于移动后的采样。本轮没有据此改 shard
运动公式或把 marker 查表自动接到 scalar shard；继续优化主要云工作负载。

## 验证口径

沿用已确认公共 `NativeMarkerSource.advance`。新增原生 cubic 模式用独立
三次多项式密度的 `.043184` 电离概率手算验收，先观察不支持该模式失败，
再实现通过。扩展同输入 ndarray 数值改变后的刷新、非零持久反馈、
跨单元/丢失、原子率失败不提交并换新场重试、几何变化拒绝及曲单元对照。
其余已有行为为回归，不把所有扩展都宣称红→绿。

初版完整原生系数缓存采用 C-order 重排；后续公共 first-velocity 小测试
显示首次、移动前速度已有浮点差异（不涉及定位候选）。只改为保留原生
gather 的 vertex/field/degree strides 后此测试由红转绿。新指纹声明
`native_coefficient_layout=stride_preserving_element_vertex_field_degree_v3`。
这证明小测试的求和路径问题得到修复，不证明它解释或修复全部真实轨迹尾部。

真实计时使用完整百万槽位与全部活 marker，共享一次出生结果；native 与各
候选串行。source 时间含当窗口 cache 准备，几何整个构造单列。无 CUDA。
Torch CPU threads=2；未额外限制或假定 NumPy BLAS 线程数。source 每配置
一次完整观测，不能据此宣称统计显著或端到端训练加速。

独立表刷新诊断在 source 完成后执行。相同实际当前场的不同只读微扰副本
供三算法共用：冻结原始 einsum、新 matmul 分配、新 matmul 原址复用。
各有一次预热；三次暖刷新按循环顺序交错；输入准备、哈希、误差归约、
预热和缓冲初始分配不算暖刷新耗时。专用诊断缓冲不与真实 source 表共享，
验证返回对象及字节地址复用、source owner/table 不变、不消耗 RNG。
FP64 matmul 可改变加法顺序，误差要实测，不宣称逐位相同。

继续使用原生单位的相对 L2、增量误差、存活身份/单元离散差异。末态几何
残差用于确认映射，不代替相对误差，也不能据此推定首个分歧子步的原因。

## 测量结果

### 已完成：v400 g33 固定场表刷新

机器结果 `output/native-topology-cache-2026-10-09/v400-bilinear-g33-refresh.json`
成功。相同运行内 native source 6.418 秒，新 bilinear source 4.005 秒
（含冷场/反馈缓存准备 0.276 秒），完整几何构造 0.857 秒。source 耗时降低
37.60%，含首次几何成本降低 24.25%；没有将独立刷新加速倍数写成 source 加速。

3986 元素、每元素 33×33 子节点、九场值表共 312,534,288 bytes。三次暖刷新：

| 算法 | 第1次秒 | 第2次秒 | 第3次秒 |
|---|---:|---:|---:|
| 冻结原始 einsum（新分配） | 1.036 | 0.921 | 0.910 |
| 固定映射 matmul（新分配） | 0.174 | 0.0462 | 0.0601 |
| 固定映射 matmul（同缓冲更新） | 0.0332 | 0.0320 | 0.0322 |

配对旧/新复用刷新比约 28.2–31.2 倍；各输入不同且旧/新共用同一个输入，
不是返回同 identity 的缓存命中。复用对象/字节地址、source 表和 owner 不变
检查通过；matmul 的数值相对 L2 差约 `1.56e-16`，不是逐位相同。
整个进程峰值约 3.06 GiB，包括真实云副本、不同刷新算法临时表与误差归约。

本路径的 R 位移增量相对 L2 仍约 `4.565e-4`，投影五通道约
`3.09e-5`–`4.11e-5`，与旧 bilinear 近似量级相同。刷新提速不能消除其
空间近似；不能因本次 v400 成功就放行上轮 injInner 的轨迹尾部风险。

### 保留的 v2 confirmed 对照（非最终缓存布局）

`v400-confirmed-g17.json` / `inner-confirmed-g17.json` 保留的是 C-order 元素
系数缓存：source 6.408→3.823 秒、43.307→28.625 秒；没有额外丢失或单元
编号差异，cubic 几何末态最大残差都约 `1e-10 m`。
v400 R/Z/phi 位移增量相对 L2 为 `3.44e-11/8.18e-11/8.95e-11`；
injInner 为 `4.95e-4/1.48e-3/3.27e-6`，但末 R/Z 速度误差仍为
`0.1403/0.04665`。不能仅凭位移和投影较小就宣称高保真。

v2 模块 SHA256 `c8c924a83d40eb575a3ba9b55aa11898ee9035945cce98750b54217396d5a0f4`。
其后只有缓存布局及版本声明改变；上述旧产物不覆盖。最终 stride-preserving
模块 SHA256 `c2cbd13804a7c2e6aa02f0b1e1f3d478dde9e628aec9b8374b5a31499029dd03`。
工具 SHA256 `648c8bbb0454b40918f72ecbb3551d24990337b77723878ac5393b7dcbcbcd74`；
默认 source SHA256 仍为
`f6dd533258fe83451c301888297a1154cf0b9d5fcdfb9c7ade23432d90bd551b`。

### 最终 v3：保留原生系数布局

两个 `*-confirmed-strides-g17.json` 均完整执行成功；九场初始值与 R/Z 梯度
probe 在所有初始活/birth 位置逐值相等。但这不是所有子步轨迹相等：

| run | native source秒 | v3 source秒 | 几何构造秒 | source耗时降低 | R位移增量相对L2 | 末R速度相对L2 | 五通道投影相对L2范围 |
|---|---:|---:|---:|---:|---:|---:|---:|
| v400 | 6.337 | 4.724 | 0.213 | 25.46% | 3.44e-11 | 4.11e-11 | 4.47e-10–1.44e-9 |
| injInner | 43.026 | 37.245 | 0.215 | 13.44% | 4.95e-4 | **0.1403** | 2.46e-7–3.97e-5 |

含首次几何成本后降低约 22.11%/12.94%。布局修正有成本，不沿用 v2
40%/34% 的 source 收益。v3 系数准备仅约 0.003 秒/窗口；整个 source
仍含定位、采样、ADAS、电荷步进及沉积/投影，不能将准备耗时当成全部成本。
指定常驻表/cache 数组约 34.73 MiB，整个过程峰值约 2.25/2.26 GiB。

双方末态活身份分别全部匹配 11197/52658，无额外丢失，百万槽位的单元编号、
寿命/出生信息无差异，权重相等，时钟/gather、RNG 与输入不变检查通过。
末态原生 cubic 几何残差最大约 `9.49e-11/9.99e-11 m`。
确认前 hint/tree 命中计数只是候选，不是确认后接受计数；fallback 为
5385/447880 与 362078/4265327（约 1.20%/8.49%），包括确认拒绝回退。

injInner 槽位 600505 占 R 位移增量误差平方和 99.928%，末 R 速度误差
平方和约 99.9999998%；其速度差约 `-9.64e4 m/s`。绝对速度仅用于追查
极端槽位，验收仍用原生单位相对误差。这说明 stride 修复消除了已复现的
首次求值数值差异，**没有修复真实轨迹的主要尾部差异**。初始采样一致而
后续差异出现，下一步应记录该身份首个分歧子步，隔离定位初值/容差轨迹
与梯度/速度敏感性，不能只看末态小几何残差就认定通过。

本轮不再添加敏感区阈值、改变物理公式/容差或搜索分辨率；保留固定拓扑
刷新优化与显式 confirmed 实验，**均不将近似定位路径切为默认训练 backend**。
低风险下一候选是保留默认定位轨迹，只复用原生布局的元素系数，先验证
整个窗口而非只验证初始采样。完整多事件耦合/BPTT、续跑以及真实非零前投影
均未在此次真实单窗口中验收；常量非零反馈由公共 fixture 覆盖。

## 复现与预算

最终公共 source 试验 **18 passed**；完整相关选择 **246 passed**（26.81秒，
1 条既有 Torch profiler 周期清理 warning），包含 shard 的现有源码顺序测试。
编译与 `git diff --check` 通过。独立只读复核未发现 matmul 布局/owner 安装及
缓存失败重试错误；同样明确数学表示一致不代表整轨迹逐位或数值资格通过。
专项测试不注入第二张表构建中途失败，不能声称穷尽所有内存失败路径。

完整相关选择的复验命令：

```bash
.torch-cuda-env/bin/python -m pytest -q \
  tests/test_native_lookup_experiment.py tests/test_native_locator_events.py \
  tests/test_native_locator_geometry.py tests/test_native_locator_latency.py \
  tests/test_native_marker_sampling.py tests/test_native_atomic_interpolation.py \
  tests/test_native_shared_spatial.py tests/test_native_source_batch.py \
  tests/test_native_source_time_integration.py tests/test_native_production_radiation.py \
  tests/test_native_coupling.py tests/test_native_production_events.py \
  tests/test_native_production_nonlocal.py tests/test_native_production_birth.py \
  tests/test_native_production_density.py tests/test_native_source_time.py \
  tests/test_native_bounded_cloud.py tests/test_native_joint_source.py \
  tests/test_native_production_torch_integration.py tests/test_native_state.py \
  tests/test_native_temporal.py tests/test_profile_native_coupled_backward.py \
  tests/test_compare_native_cpu_sampling.py tests/test_native_shard_dynamics.py
```

输出目标必须不存在。所有重型命令串行，不与测试或其它计时作业重叠。

```bash
.torch-cuda-env/bin/python -m pytest -q tests/test_native_lookup_experiment.py
.torch-cuda-env/bin/python -m tools.benchmark_native_lookup \
  --run-index 0 --grid-points 33 --sampling bilinear --refresh-repeats 3 \
  --output output/native-topology-cache-2026-10-09/v400-bilinear-g33-refresh.json
.torch-cuda-env/bin/python -m tools.benchmark_native_lookup \
  --run-index 0 --grid-points 17 --sampling native_cubic_confirmed \
  --output output/native-topology-cache-2026-10-09/v400-confirmed-strides-g17.json
.torch-cuda-env/bin/python -m tools.benchmark_native_lookup \
  --run-index 1 --grid-points 17 --sampling native_cubic_confirmed \
  --output output/native-topology-cache-2026-10-09/inner-confirmed-strides-g17.json
```

每命令 wall 预算 600 秒、RSS 8192 MiB；资源判据以整个进程 high-water RSS
而非表数组统计为准。源配置含 sampling 与刷新算子声明，旧训练检查点未
接入此试验；两种模式都不作为默认或训练资格认证。
