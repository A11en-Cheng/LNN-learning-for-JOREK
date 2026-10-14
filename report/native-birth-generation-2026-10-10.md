# Marker 出生位置：总源平台 + 无源强拒绝抽样（2026-10-10）

**版本后续**：当前 mass gate、clip→环向平均及真实链默认已修正，见
[native-birth-native-correction](native-birth-native-correction-2026-10-10.md)。
下述默认未切换、先平均再clip及旧尺度速度/误差仅记录初次实验，不代表当前版本。

范围：按用户要求先优化生成，不优化网格定位，不训练。交付显式可选位置模块、
真实 shard 快照的生成计时、独立公式/分布测试和小网格事件续跑验证。
默认出生算法及训练入口未切换；不宣称连续源精确采样、真实生产闭环或完整训练加速。

**后续核查补充（同日）**：见
[传统方法与原生对照](native-birth-traditional-comparison-2026-10-10.md)和
[完整调用链证据](native-birth-clipping-order-primary-evidence-2026-10-10.md)。
本报告 §2 沿用的是现有 Python number-source 接受尺度，不是 JOREK 字面
mass-source 尺度；两路径共有的 `central_mass/20` 回转换现已登记为原生差异。
下文两个快照“未饱和”限于现有 Python 模式，**不适用于 injInner 原生判据**。
本报告表分辨率比较及速度数仍保持原实验范围，不能代替后续传统/原生误差报告。

## 1. JOREK 一手证据与二维约定

源码只读根：`/home/ac_desktop/JOREK_code/jorek`。

- `particles/examples/marker_spi_source.f90:394-408`：shard 周围
  `±5*max(radius*ns_radius_ratio,ns_radius_min)` 构造 R/Z 出生包围盒。
- `particles/mod_initialise_particles.f90:391-393`：按 R/Z 维分别与域包围盒相交；
  无交集的维回退全域；phi 默认 `[0,2π]`。包围盒不等于有效曲边 FEM 域。
- `tools/mod_sampling.f90:81`：R 按平方坐标均匀抽取，Z/phi 线性均匀，
  即 proposal 对 `dV=R dR dZ dphi` 均匀。
- `particles/examples/marker_spi_source.f90:809-815`：
  `f=max(P*dv/P_norm,0)`，`dv=.01*.01*R_geo*(2π/n_plane)`；
  `particles/mod_initialise_particles.f90:531` 的均匀数比较隐式实现上限 1。
- `models/mod_source_shape.f90:32` 的环向高斯与
  `models/mod_injection_source.f90:377` 的 phi 参数没有因 ntor=1 消失。
  当前两个 run 的 `logfile_spitest` 中均 `n_tor=n_plane=1`，
  `jorek_namelist_spitest:501` 均 `ns_deltaphi=1000`。
  最短角差范围内环向核最大相对变化 `1-exp(-(π/1000)^2)≈9.87e-6`。

因此，用户所要求的“ntor1 无环向分布”实现为**显式轴对称模式**：
源核取连续环向平均，出生 phi 均匀占位，不依赖 shard 的 phi。
不是把旧出生代码解释为已经严格无角度。

当前已有 density 的 `toroidal_average=True` 是配置平面上的离散平均；
只有一个平面时，不等于连续环向平均。此次不改解析密度源/投影对象。

## 2. 数学目标与平台

令 `S=sum_i A_i K_i` 为总 number-source rate，单位转换沿用已有 birth 路径。
接受密度对应：

```text
a = float32(max(S * dv / local_norm, 0))
b = min(a, 1)
p(position) ∝ b(position) dV
```

平台作用在**总源**，不是各 shard 分别截断后相加。源强平台在舍入前对应
`S_cap=local_norm/dv`。平台只改变位置分布的形状；出生数量和实际 gather
对应的总权重仍由原路径决定，不因截断减少物质预算。

两种显式几何模式：

- `axisymmetric_n0`：连续环向平均源后再截断，即 `clip(mean_phi(a))`，
  RZ 分布不依赖 shard phi。要求声明 `n_tor=1`。
- `periodic_phi`：按最短周期角差计算原角核，在三维上截断总源；显式声明
  `n_tor>1` 和 `phi_cells`。目前只验证位置模块，不代表 H=1 场链路支持三维。

一般情况下 `clip(mean(a)) != mean(clip(a))`。旧三维拒绝分布的 RZ 边缘是后者，
所以新 n0 模式不无条件宣称与旧模式严格等价。当前两个快照未观察到饱和；
窄环向核且跨阈值的其他配置不能据此推广。float32 舍入也不是连续积分恒等式。

## 3. 实现与接口

新模块：`src/models/native_birth_positions.py`。

```python
sampler = NativeTabulatedBirthSampler(
    geometry="axisymmetric_n0", n_tor=1,
    cells_per_width=8, max_cells=2_000_000,
)
# 在 NativeProductionBirths 的现有构造参数外显式增加：
births = NativeProductionBirths(..., position_sampler=sampler)
```

核心接口 `prepare(kernel, shards, time_si, bounds, dv, norm)` →
事件拥有的不可写概率表；`sample(count)` → `[count,3]` 的 R/Z/phi。
没有跨事件源强缓存；每个源事件根据当时 shard/消融率/体积重建。

实现步骤：

1. 保留已有完整 shard 源核、解析/记录数值源体积、宽度及归一化。
2. 用 R/Z 高斯的可分离性，矩阵批量叠加所有有效 shard，不逐候选查询。
3. 在表顶点截断总接受密度到 `[0,1]`，按 `u=R²/2,Z,phi` 做非负三线性重建。
   n0 两个 phi 面完全相同。每个小单元积分是八顶点平均乘精确柱坐标体积。
4. CDF 选小单元；各条件密度均为线性，依次以稳定二次方程根抽取 u/Z/phi。
   不使用源强拒绝循环，也不使用迭代求根。每个位置消耗四个 uniform。
5. 接入现有 births 后，**仍用原 mesh.locate**；域外候选仍会重新抽取。
   初始化 V_prev 仍从 sim_time 对应的原始流体历史采样，不使用未来场。

源强截断顶点的重建是数值近似，尤其平台过渡线会受分辨率影响。
抽样对重建表精确，不等于对连续 Gaussian 总源精确。
表预算不足明确失败，不静默降分辨率、减少粒子数或跳过边界。

模式、分辨率、phi 策略及 RNG 协议进入 birth → shard → model 配置指纹；
运行时配置变更/移除绑定 sampler 明确失败。
旧默认七 uniform 拒绝路径的配置及已登记确定性流保持不变。
模型检查点不能静默跨新旧模式恢复。

## 4. 真实快照位置生成 benchmark

最终证据：
`output/native-birth-generation-2026-10-10/position-only-linear-final.json`。
包含快照/源码 SHA256、NumPy 版本、真实 OpenBLAS 单线程查询、分次 wall/CPU 时间。
早期常数表 `position-only.json` 和线性中间版本 `position-only-linear-v2.json` 保留，
不混用早期误差与最终性能。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .torch-cuda-env/bin/python -m tools.benchmark_native_birth_generation \
  --output output/native-birth-generation-recheck/position-only.json --repeats 3
```

固定三个 seed：2917/2918/2919；新旧顺序交替；中位数包含源准备/建表。
**不含** H5 读取、有效网格定位、出生速度初始化、全云推进、投影或网络。
旧对照原样执行 seven-uniform/source gate，主动排除 mesh gate以隔离生成成本。

| case | 全部 shard / 有效源 shard | 出生数 | 旧位置生成中位数 | 新建表+生成中位数 | 旧源强候选数（三 seed） |
|---|---:|---:|---:|---:|---|
| v400 jorek004111 | 846 / 844 | 29 | 11.913244 s | 0.001256 s | 201463 / 180309 / 157796 |
| injInner jorek005681 | 846 / 724 | 56 | 0.786457 s | 0.001656 s | 13627 / 15420 / 16263 |

新模式每次直接生成 29/56 个候选，无源强拒绝。
上述出生数来自记录 tstep×消融率/百万槽名义权重，是位置微基准的代表性预算；
不是从完整 marker 时钟 replay 推导的 gather，不作为真实完整事件验收。
没有裁剪原始 shard 集合、marker 云或网格；本 benchmark 根本不读取/推进 marker 云。

默认表大小分别 10920/27680 小单元；拥有的表数组约 0.355/0.894 MB。
不含构建临时矩阵及进程 RSS，不能把这些数当峰值内存。

## 5. 分布近似诊断

将分辨率从每最小核宽 8 格加倍到 16 格，两个方向精确积分线性重建到同一支撑。
第二个方向在细网格上比较，防止粗单元总量掩盖单元内部曲线失真。

| case | 粗支撑概率相对 L2 | 细支撑概率相对 L2 | 细支撑总变差 | 接受积分相对差 |
|---|---:|---:|---:|---:|
| v400 | 0.2124% | 0.2177% | 0.1099% | 2.93e-9 |
| injInner | 0.1226% | 0.1226% | 0.0511% | 5.58e-10 |

这是**两种分辨率表之间的位置概率差异**，不是连续源真实误差上界，
不是 marker 投影/流体场误差，也不是 Monte Carlo 统计显著性验收。
两个快照接受阈值最大值约 0.00653/0.08841，均未饱和。
平台分支另由人工重叠源与全平台案例、独立解析公式验证。

## 6. 测试与验收范围

最终相关回归 **109 passed**（11.67 s）；编译、`git diff --check` 通过。
最终 benchmark 记录的四个源码 SHA256 与交付源码逐一核对一致。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .torch-cuda-env/bin/python -m pytest -q \
  tests/test_native_birth_positions.py tests/test_benchmark_native_birth_generation.py \
  tests/test_native_production_birth.py tests/test_native_density_preparation.py \
  tests/test_native_production_density.py tests/test_native_production_events.py \
  tests/test_native_production_torch_integration.py tests/test_native_nonlocal_event_binding.py \
  tests/test_native_chain_execution.py tests/test_native_source_time_integration.py \
  tests/test_native_production_nonlocal.py tests/test_native_sparse_source.py \
  tests/test_native_joint_source.py
```

新增测试覆盖：独立高斯/体积/平台公式、柱坐标统计、抽样直方图、表分辨率收敛、
单元内部闭式反函数、非可分离三维条件抽样、周期角核、n0 角度独立、快照拥有权、
资源/配置/无源守卫、RNG 重放与 mesh 失败回滚、rawD 出生速度、完整 source 窗口。
另通过小网格两次完整耦合事件及新模型检查点恢复，比较完整云/反馈/场历史/shard，
并实际验证旧模式模型拒绝载入新模式检查点。

剩余工作：在不改 mesh.locate 的真实出生入口先比较有效域接受比例、出生空间矩和投影，
再讨论是否启用训练入口；之后单独处理定位。没有批准概率误差阈值，
不自行把上述细表差异认定为统计误差或生产资格。

本轮未训练、未更新 optimizer、未修改 JOREK/原始数据/roadmap/ADR。
使用 codebase-design 技能，将位置抽样复杂性封装在 prepare/sample 接口后，
同一出生接口保留旧拒绝实现与显式新表实现供比较。
