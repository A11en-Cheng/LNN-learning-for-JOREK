# Marker 生成修正：原生概念契约 + 概率分布算法

2026-10-10。本轮修正出生逻辑并验证，不启动真实训练、不更新现有训练模型/optimizer、不修改 JOREK、
原始数据、roadmap 或 ADR。旧证据/产物保留；不静默迁移旧检查点。

## 1. 已交付与范围

- 删除出生判据额外的 `central_mass/20` 数目源回转换。
- 新旧算法共用 `native_birth_acceptance.py::native_birth_threshold`：输入已经是
  impurity **mass-density rate**，执行原生 `real4(max(P*dv/P_norm,0))`。
- 新概率表先叠加 shard 总源，再截断；n0 改为 **总接受概率截断后取环向平均**。
- 真实链构建器默认 `native_probability_n0`；修正后的拒绝法保留为显式参考。
  提供 `native_probability_periodic`，允许 H=1 场配合原生 source phi 核抽样，
  不再错误地认为 n_tor=1 就不能有 source phi 分布。高谐波场推进仍不支持。
- 出生数量、MPI local rate、actual gather、float32 权重、RZ box/柱坐标测度、
  原 locator、rawD/sim_time 速度初始化、neutral P_imp、槽位复用语义未改变。
- 接受契约、birth 和 table 版本/模式都进入配置指纹；旧质量尺度不静默恢复。

只确认当前支持的纯 Ne、圆云、n_period=1、H=1 与所述两个真实初态。
表格重建、环向积分和 n0 均匀占位 phi 是登记的数值/表示近似；没有认证任意
混合 species、drift、拉长云、MPI/OpenMP/PCG32 逐随机数复现或生产闭环。

## 2. JOREK 到底怎样生成 marker

后台一手核查报告：
[native-birth-native-contract-evidence](native-birth-native-contract-evidence-2026-10-10.md)。
记录实际文件 SHA、行号、clock 与初始化细节，主代理复核关键调用。

之前“逐个 shard 各生成 marker”的描述不准确，混淆了 **候选位置内求源**
和 **外层填充 marker 槽位**。

1. 在当前读入的 shard 状态上求总速率
   `Q=sum_j spi_abl_j*spi_species_j`，并合并所有 impurity shard 的 RZ 出生 box。
   [marker_spi_source.f90:398](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:398)。
2. 对本 rank：`Q_r=Q/n_cpu`，`Δ=n_steps*timesteps`，名义 `w0=total_atoms/n_particles`。
   `N=int(Q_r*Δ/w0)`；仅 `Q_r*Δ>1e14` 时保证至少1个。不是每个 shard 各自取整。
   [422](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:422)。
3. 选择 N 个 `i_elm<=0` 的空闲槽。每个未成功槽位抽圆柱均匀候选：
   `R=sqrt(Rlo²+U1*(Rhi²-Rlo²))`，Z、phi 线性均匀。
   [mod_initialise_particles.f90:420](/home/ac_desktop/JOREK_code/jorek/particles/mod_initialise_particles.f90:420)。
4. 原生先 `find_RZ`；成功后插值 psi，再在该候选位置遍历 injector/shard 求源核、
   宽度、体积、时间开关与 species/mass 分配，最后汇总为总 impurity mass source P。
   [501](/home/ac_desktop/JOREK_code/jorek/particles/mod_initialise_particles.f90:501)。
5. 求 `a=real4(max(P*dv/P_norm,0))`，`P_norm=Q_r*t_norm/n_norm`，
   `dv=real4(0.01*0.01*R_geo*(2π/n_plane))`；`U4<a` 才接受。
   [531](/home/ac_desktop/JOREK_code/jorek/particles/mod_initialise_particles.f90:531)。
   a>1 表示总概率饱和，不表示流体源/消融强度被截断。
6. 接受后写 x、element、local st；在 `sim_time` 取速度给 V_prev，
   `P_imp(0)=1`、其余电荷态为0，`weight=real4(Q_r*Δ/N)`，`i_life+=1`。
   birth 不更新 t_birth；失败槽位继续重试，成功的不重生。原生无我们的安全尝试上限。
   原生 PCG 候选实际使用4个随机数；Python 参考七数流是独立协议，不宣称逐位复现。

因此本例每个合法候选会检查846个 shard，但没有846次独立 birth batch。
新算法把重复的空间源查询改为一次事件内的总概率表准备，然后直接抽 N 个位置。
表中候选仍经过原 mesh.locate；有效域外重抽，但不再作源强接受/拒绝。

## 3. 为什么不是每个 shard 截断后叠加

对同一 rank/事件，用同一个 `dv/P_norm` 定义各 shard 的非负无量纲贡献 f_j：

```text
原生接受函数 b(x) = min(1, sum_j f_j(x))
新分布目标 p(x)  = 1_box(x)*1_mesh(x)*b(x) / ∫box∩mesh b(y) dV(y)
dV = R dR dZ dphi
```

逐 shard 计算 K_j、强度及体积，然后**线性叠加**是正确的。
但 `sum_j min(1,f_j)` 直接归一化不是这个目标：在重叠区会保留大于1的测度。
例如两个贡献均为0.7，原生为1，逐项截断后相加为1.4；在别处只有0.7，
两目标归一化后的相对比重仍不同。

若最后还保留总截断、且各项非负/阈值同为1，则
`min(1,sum_j min(1,f_j))=min(1,sum_j f_j)`，逐项截断只是冗余。
如果每个 shard 用自己的 Q_j、阈值或出生数量，这个等式与同分布推导不能直接套用。
只有总源未饱和时，按正确的积分质量混合未截断分量才可直接等分布。

n0 RZ 边缘的目标是 `mean_phi(min(1,sum_j f_j))`。
不是 `min(1,mean_phi(sum_j f_j))`，也不是平均逐 shard 平台后混合。
保留用户 n0 无环向分布约定时，phi 均匀占位；它只保留原生 RZ 边缘，
不是原生 RZphi 联合分布。periodic 模式则保留 phi 条件分布的表格近似。

共同 phi 中心时，用连续高斯截断积分解析取平均，再作 real4 节点舍入；
这是浮点近似，不是逐 phi 原生舍入重放。多 phi 中心时至少64个周期中点，
随核宽/分辨率增加，超过4096点明确失败，不静默降分辨率。
当前两例各有844/724个不同的活源 phi 中心，实际均使用64点。
缓存各 block 的 R/Z 基函数；每角度以矩阵乘法总和，不为每个中心存整张 RZ 表。

## 4. 相关度、L2 与“分布误差”是什么

统一固定24×24的 RZ 比较 bins。p_i 是原生接受分布在第i格的**积分概率质量**，
q_i 是新表的积分概率质量；都满足总和1，积分已经包含 R 雅可比与 phi。
参考使用独立解析高斯积分（证明不饱和的 v400）或独立 Gauss 求值/逐点总截断
（injInner）；不是以细表自身当原生真值。新表的双线性重建精确积分到相同 bins。

### 相对 L2

```text
L2 = sqrt(sum_i (q_i-p_i)^2)
relative_L2 = L2 / sqrt(sum_i p_i^2)
百分数 = 100*relative_L2
```

这里不是逐 marker 位置误差、逐 shard 误差或流体 loss。改变 bins 会改变结果；
不称全连续空间误差界。参考全零时失败，不靠随意 epsilon 将错误藏掉。

### Pearson 相关度

```text
corr = sum_i (p_i-mean(p))*(q_i-mean(q)) /
       sqrt(sum_i (p_i-mean(p))^2 * sum_i (q_i-mean(q))^2)
```

接近1表示空间形状相似，不保证幅值、局部误差或物理守恒正确。
对恒定向量相关度未定义，机器结果为 null，不写成1。概率归一化已移除总源强度，
所以还要另外检查 N、weight 与 Q_r*Δ；不能仅凭这些分布指标判断注入物质总量。

### 总变差 TV

```text
TV = 0.5*sum_i abs(q_i-p_i)
```

它是离散比较支撑上的概率质量差，可解释为任意这些格子的集合上最大概率差。
TV=0.001407 表示约0.1407%的概率质量差，不是所有位置都偏移0.1407%。

有限样本指标另按 `hist_i=N_i/N` 计算：两个传统目标样本互比估计抽样波动；
传统与新样本互比检测总差异。每个案例4个 seed、每样本50k点。饱和参考通过
独立未截断高斯混合 proposal，再按 **总 f** 的 `min(1,1/f)` 接受修正；
这仅为验证器，不是正式新表生成路径，也没有逐 shard 截断。

## 5. 修正后结果

权威产物目录：`output/native-birth-native-correction-2026-10-10/`。
final 文件中的源码 SHA 与交付代码全部匹配，旧版/中间产物不覆盖。

| 相对 JOREK 字面目标，24×24 bins | v400 4111 | injInner 5681 |
|---|---:|---:|
| 每核宽8格，概率相对 L2 | 0.2733% | 0.1376% |
| 相关度 | 0.9999980 | 0.9999992 |
| TV | 0.1407% | 0.06120% |
| 每核宽16格，相对 L2 | 0.06808% | 0.03528% |
| 每核宽32格，相对 L2 | 0.01718% | 0.00914% |
| 传统样本互比，L2中位数 | 3.302% | 5.764% |
| 传统vs新表样本，L2中位数 | 3.433% | 5.767% |

细化后的偏差下降，不能称全为统计噪声。injInner 旧数目源尺度与原生目标的
16.229%系统差异已去掉；现在新表对原生目标约0.1376%，不宣称偏差为零。
独立原生32→64阶积分 L2 差约 `3.93e-10 / 5.10e-6`，是收敛诊断而非严格界。
这些概率误差不含 mesh 筛选，不能直接转述为完整云/投影/流体误差。

三 seed交替计时，单线程OpenBLAS，包含准备/建表：

| 位置生成微基准中位数 | 修正后的拒绝法 | 新概率表 | 倍率 |
|---|---:|---:|---:|
| v400，29 births | 0.5606 s | 0.01981 s | 28.29× |
| injInner，56 births | 0.05472 s | 0.03547 s | 1.54× |

不含H5读取、定位、速度初始化、云推进、投影或网络。此前数千倍对照使用错误的
接受尺度，不能作为修正后加速宣称。`position-timing.json` 是每phi中心单独RZ表的
较慢中间实现；最终使用 `position-timing-final.json`，没有选择性删除负结果。

## 6. 实际出生入口与兼容性

`real-entry.json`：真实完整网格、全部100万槽，原活云11168/52604粒子不裁剪。
使用登记 poststartup 初态、实际 particle schedule 与保存的 rawD；不读未来场作为输入。
两个生成算法各实际调用出生入口，生成29/56个 newborn，全数有效元素、st合法、
中性11态、有限V_prev；原云所有数组 hash 不变。
总权重相对 `Q_r*gather` 差分别 `4.54e-9 / 3.34e-8`，两算法一致。
实际gather为40/81个粒子步，不从 namelist 假定连续时钟。
仅测试出生入口，没有插入后推进完整真实云、源投影或场更新。

`native_probability_n0` 是真实链新默认。`smoke_native_autonomous` 提供
`--birth-position-mode native_rejection_reference` 或 `native_probability_periodic`；
分辨率分别由 `--birth-cells-per-width`、`--birth-phi-cells` 指定，均进入指纹。
Python底层 `NativeProductionBirths` 仍可无 position_sampler 构造显式参考。
没有自动退回旧物理尺度或自动按性能切换模式。

回归 **125 passed（12.63 s）**，包括独立 Fortran 接受/边界舍入、周期角统计、
clip→phi平均反例、MPI数量、实际gather权重、rawD速度、RNG回滚、两完整小网格事件、
新模型恢复及旧质量契约拒绝。编译、`git diff --check` 通过。

额外自主入口/数据审计/真实构建器相关回归 **27 passed，1 skipped（59.62 s）**，
合计 **152 passed，1 skipped**。额外回归中训练入口测试会在隔离的小型解析 fixture
上执行优化器更新与恢复，以验证接口兼容；未加载或更新现有真实训练模型/产物，
不计作真实训练。本机该次CUDA测试因设备可用性跳过，没有据此宣称GPU性能。

复现（均要求新输出文件）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 .torch-cuda-env/bin/python -m tools.validate_native_birth_distribution \
 --output output/native-birth-native-recheck/distribution.json --sample-count 50000
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 .torch-cuda-env/bin/python -m tools.benchmark_native_birth_generation \
 --output output/native-birth-native-recheck/timing.json --repeats 3
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
 .torch-cuda-env/bin/python -m tools.validate_native_birth_entry \
 --output output/native-birth-native-recheck/real-entry.json
```

后续如继续，需要在新物理/生成指纹下重新验证完整真实云事件及投影分布，
不能直接恢复旧质量尺度检查点或用此前加速倍率替代完整链路计时。
本轮未执行训练或完整真实云事件。使用 research skill 独立追踪原生调用链，
使用 codebase-design skill 集中共用接受判据，保持算法与物理契约的区别。
