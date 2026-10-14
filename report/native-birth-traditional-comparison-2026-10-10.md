# Marker 出生：截断次序、传统方法对照及原生质量系数差异

**后续修正**：本报告记录发现问题时的旧实现与产物；质量尺度与n0截断/平均次序
现已修正，真实链构建器采用新概率路径。当前实现、原生对照与兼容性见
[修正报告](native-birth-native-correction-2026-10-10.md)。不将本报告旧16.229%当作修正后误差。

日期：2026-10-10。范围仅为出生位置生成；不优化定位、不训练、不改变默认算法，
不修改 JOREK、原始数据、roadmap 或 ADR。已有工作树修改保留。

## 1. 结论

1. 所核查 JOREK 调用链是 **shard 空间源求和 → 总源接受判据 → 概率饱和**。
   没有逐 shard 的上限平台再叠加；逐 shard 负值归零、宽度下限不是上限平台。
   当前新表的总源截断顺序不需要改为逐 shard 截断。
2. 新表相对**现有传统 Python 出生目标**的 RZ 概率相对 L2 误差为
   v400 **0.2733%**、injInner **0.1438%**（固定 24×24 比较支撑）。
   这是真实高斯目标与表格积分的对照，不是两种表格分辨率互相比较。
3. 新旧 Python 路径共同多乘了 `q=central_mass/20`，与检查的 JOREK
   字面出生判据不同。两个快照 `central_mass=1`，即 `q=0.05`。
   injInner 原生公式发生饱和，现有 Python 目标相对它的概率 L2 差异约
   **16.229%**；不能以“优化前后一致”宣称符合原生。这是系统差异，不是统计波动。
4. 以上均是 **RZ 外包框内、有效网格筛选之前的出生位置分布**，不是完整云、
   沉积、滤波投影、反馈或流体误差。质量差异尚未修复；不授予新模式生产资格。

## 2. 平台先后顺序的一手证据

独立后台研究及主代理复核见
[完整源码证据](native-birth-clipping-order-primary-evidence-2026-10-10.md)。
该报告记录实际源码行号、文件 SHA256、分支范围及调用者质量换算检查。

- [mod_injection_source.f90:404](/home/ac_desktop/JOREK_code/jorek/models/mod_injection_source.f90:404)：
  每个 shard 的核/消融率/体积归一化后，按 species 与质量比累加 impurity 源。
- [mod_initialise_particles.f90:513](/home/ac_desktop/JOREK_code/jorek/particles/mod_initialise_particles.f90:513)：
  再跨 injector 求和；[531](/home/ac_desktop/JOREK_code/jorek/particles/mod_initialise_particles.f90:531)
  将总 `source_tmp` 交给接受变换。
- [marker_spi_source.f90:809](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:809)：
  `f=real4(max(P*dv/P_norm,0))`。没有显式 `min(f,1)`；与 [0,1) 的随机数比较，
  有效概率自然为 `min(f,1)`。它截断的是**出生接受概率**，并不截断流体源或消融量。

用共同阈值写无量纲正贡献 `f_j`：

```text
原生概率       = min(1, sum_j f_j)
逐 shard 平台后直接叠加的源测度 = sum_j min(1, f_j)
```

二者归一化后一般也不同。若两个贡献均为 0.7，前者为 1，后者为 1.4。
但须注意：对于非负贡献、同一个上限 1，**如果叠加后仍作总截断**，
`min(1,sum_j min(1,f_j))=min(1,sum_j f_j)`。
这时逐 shard 截断是冗余而非不同物理。逐 shard 用不同归一化/阈值则不适用此等式。

饱和重叠源 fixture 中，逐 shard 平台后直接归一化叠加相对原生总平台的
概率 L2 差异 **8.0974%**。因此不以逐 shard 平台的独立混合代替总源概率。

## 3. 对照方法：明确区分三个参考

入口：`tools/validate_native_birth_distribution.py`。
最终机器证据：
[comparison-final.json](../output/native-birth-distribution-2026-10-10/comparison-final.json)。
较早的 `comparison.json` / `comparison-v2.json` 保留，前者没有完整原生对照，
后者只有原生上界审计；以下结论以 final 为准。

### A. 现有传统 Python 目标的解析参考

保留全部 846 个 shard 及记录的 `spi_vol_arr`，不裁剪源集合。
只从现有 source kernel 取得源系数；不用新表节点、概率或细表作为真值。
现有 Python 接受阈值全域上界分别为 **0.0088366 / 0.471303**，均小于 1。
因此连续位置目标是高斯线性叠加，可以独立解析积分：

```text
P(bin) ∝ sum_j A_j [∫Rbin R exp(-(R-Rj)^2/wj²)dR]
                        [∫Zbin exp(-(Z-Zj)^2/wj²)dZ]
```

共同最近周期角核的全圈积分在归一化后抵消；不是忽略圆柱 R 雅可比。
传统 real*4 接受舍入对归一化分布的 TV 界约 `5.96e-8`，不包含 FP64 源求值误差。
新 n0 表先取角均值再舍入，同样不是逐随机数重放；解析对照忽略这一极小舍入项。

新表按实际 `R²/2,Z` 双线性重建精确积分到独立固定 24×24 bins。
所有下列 L2 都为 `||P_new-P_ref||₂/||P_ref||₂`；相关度是概率向量 Pearson；
TV 为 `0.5 sum(abs(P_new-P_ref))`。比较支撑改变会改变 L2/相关度，
这些不是连续全空间误差界，也不能直接当作后续投影误差。

### B. 抽样波动与实际传统小批重放

每个真实案例每个 seed 抽 **50,000** 个位置，4 个 seed：31/47/89/113。
每个 seed 包含两个独立传统目标样本与一个新表样本。
大样本参考是另一 proposal 的拒绝抽样：按圆柱积分选 shard，截断高斯采 R，
保持 shard 身份不变，按 `R/Rmax` 接受修正雅可比，再采 Z 与周期 phi。
它与传统**连续、未饱和**目标同分布，不是原七随机数循环的大批逐位重放。
独立解析积分、柱坐标均值和原始 source gate 对照测试验证该参考。

另实际执行原 scalar `legacy_positions`，seed=2917，29/56 个位置，
分别消耗 **201463 / 13627** 个传统候选。原位置、表位置和低 N 指标都保留。
低 N 的 24×24 直方图非常稀疏，不据此判断算法保真；不同 RNG 协议不按相同
marker ID 比较位置相对误差。此处秒数也不用于性能结论。

饱和两 shard fixture 使用柱坐标均匀候选、原总源 float32 gate 的向量批量拒绝；
无需新表。接受判据逐点与旧 kernel 查询一致，并检验首个接受点重放。
独立 64 阶复合 Gauss/8 阶 phi 积分作确定性参考，32→64 概率 L2 差
`2.535e-6`；新表相对它的 L2 为 **0.3383%**。

### C. JOREK 字面质量源判据参考

不是编译执行 Fortran。依据已核实原生公式，对同一记录源系数作独立复合
Gauss 求值，在**每个积分点**总源求和、real*4 舍入、概率截断，保留实际 phi 核。
不使用新表的节点截断或插值；R/Z 每 bin 32 阶、phi 8 阶，另与16阶比对。
两数据 `ns_deltaphi=1000`，phi 核近似均匀，但积分并未强制 phi 均匀。
16→32 的概率 L2 差分别为 `1.029e-9 / 1.934e-5`；这是收敛诊断，不是严格误差界。

## 4. 定量结果

| 指标 | v400 4111 | injInner 5681 |
|---|---:|---:|
| 新表 vs 现有传统连续目标，相对 L2 | 0.2733% | 0.1438% |
| 同一比较的相关度 | 0.9999980 | 0.9999992 |
| 同一比较的 TV | 0.1407% | 0.06245% |
| 每核宽16格，相对传统目标 L2 | 0.06808% | 0.03600% |
| 每核宽32格，相对传统目标 L2 | 0.01718% | 0.009016% |
| 传统样本 vs 独立传统样本，L2 中位数 | 3.302% | 5.215% |
| 传统样本 vs 新表样本，L2 中位数 | 3.433% | 5.322% |
| 现有传统连续目标 vs JOREK 字面目标 L2 | 约3.17e-8% | **16.229%** |
| 新表 vs JOREK 字面目标 L2 | 0.2733% | **16.118%** |

新表离散偏差比本次 50k 抽样噪声小，但偏差明确非零且随网格细化下降。
仅4个 seed 不作统计等价或显著性保证，没有经批准的容差，不自动判定生产通过。
相对传统目标的抽样误差与相对原生目标的系统偏差不能混写。

## 5. 新发现：质量源与数目源转换

原生 `total_imp_source` 输出 Ne mass-normalized source：

```text
q = central_mass/20
P_native = sum_j (ablation_j * species_j * K_j/V_j) * t_norm/n_norm / q
norm = local_ablation_atoms_rate * t_norm/n_norm
f_native = P_native * dv/norm
```

调用者没有把 `phys_source` 换成质量速率：
[marker_spi_source.f90:410](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:410)
求 atoms/s，[436](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:436)
直接传入；[mod_initialise_particles.f90:377](/home/ac_desktop/JOREK_code/jorek/particles/mod_initialise_particles.f90:377)
直接赋给 `phys_source`。完整赋值核查见源码报告 §7.1。

现有 `NativeProductionBirths` 及新 `NativeTabulatedBirthSampler` 均将源额外乘 `q`，
令舍入/饱和前 `f_python=q*f_native`。两快照为1/20，不是泛指所有案例均1/20。

- v400：各 RZ 小格的独立高斯最大值与 phi 核≤1 给出原生全域阈值上界
  **0.13737<1**。双方无饱和，共同倍率在位置目标归一化中抵消，接受率仍不同。
- injInner：该保守上界为1.9266（上界>1本身不证明饱和）；另在独立积分点
  **实际求得原生阈值1.77018>1**，确有平台。乘0.05使现有 Python 不再饱和，
  改变归一化位置分布，带来上述16.229%差异。

原生先定位再计算源，本比较刻意排除 mesh gate。
因此16.229%不是已测的**有效FEM域最终出生分布**误差，尤其不能转述为云投影误差。
本轮只登记问题并量化字面公式；没有悄悄修复质量因子、重签旧检查点或切换训练。

## 6. 复现与回归

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .torch-cuda-env/bin/python -m tools.validate_native_birth_distribution \
  --output output/native-birth-distribution-recheck/comparison.json \
  --sample-count 50000 --literal-repeats 1
```

要求输出不存在，保留旧证据；机器文件包含快照/五个代码 SHA256、seed、
配置、上界、逐次指标及低N位置。查询确认 NumPy/SciPy OpenBLAS 均单线程。
final 文件的五个代码 hash 与交付源码逐一一致。

新增独立验证 **12 项通过**；包括 Gauss/解析公式、原判据逐点一致、
七随机数首接受点重放、圆柱雅可比、未证明不饱和时拒绝解析参考、
原生系数的无饱和抵消与有饱和差异、人工重叠平台及异常指标守卫。
相关回归 **121 passed（12.70 s）**；编译与差异检查通过。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .torch-cuda-env/bin/python -m pytest -q \
  tests/test_validate_native_birth_distribution.py \
  tests/test_native_birth_positions.py tests/test_benchmark_native_birth_generation.py \
  tests/test_native_production_birth.py tests/test_native_density_preparation.py \
  tests/test_native_production_density.py tests/test_native_production_events.py \
  tests/test_native_production_torch_integration.py tests/test_native_nonlocal_event_binding.py \
  tests/test_native_chain_execution.py tests/test_native_source_time_integration.py \
  tests/test_native_production_nonlocal.py tests/test_native_sparse_source.py \
  tests/test_native_joint_source.py
```

## 7. 下一步与技能使用

先对齐出生判据的质量系数：需要独立版本/配置迁移，不静默覆盖旧RNG与检查点语义。
之后在保留原 mesh.locate 的完整出生入口作有效域、空间矩与局部投影 A/B，
将概率离散偏差和有限粒子统计误差分别量化，再决定分辨率及是否切换训练入口。
不因概率相关度接近1就直接跳到全流程性能或训练精度结论。

使用 `research` skill，将 JOREK 调用链的一手核查交给后台研究代理；
主代理独立复核关键调用与质量换算，同时编写/执行传统目标数值对照。
该 skill 促使本轮明确分开原生事实、当前实现与数值推论。
