# Native birth clipping 顺序：本地原生源码证据（2026-10-10）

## 结论与证据范围

针对所检查的 SPI impurity marker birth 调用链，**没有逐 shard 的强度上限阈值或平顶化再叠加**。实际顺序是：逐 shard `source_shape` → `inj_source`（包括负值归零）→ 按 species/mass_ratio 分配并在 injector 内累加 → 跨 injector 累加 impurity 源 → 一次 `f_source_imp` → 与 uniform 随机数比较。最后一步对接受概率自然饱和，不能称为流体源强度被截断。逐 shard 的负值归零确实存在；它不是上限平台。

本报告只依据磁盘上的一手源码；未训练、未运行 benchmark、未修改主代码。JOREK HEAD 为 `54c675bcc39c33b082c0c35046e28122dc53b3aa`，实际字节以末尾 SHA256 为准，不把 HEAD 当作工作树全部内容的证明。ADR 索引已读；本报告不以 ADR-0001..0017（历史）作流程证据，不修改架构文档。

以下 JOREK 路径缩写 J = `/home/ac_desktop/JOREK_code/jorek`，Python 路径缩写 L = `/home/ac_desktop/LNN-learning-for-JOREK`。行号均为当前文件实际行号；结论限于该源码分支与所述配置，未验证某个外部运行的编译宏、namelist 或生产轨迹。

## 1. 完整调用与两种不同的 sum

| 顺序 | 精确证据 | 实际操作 |
|---|---|---|
| A. birth 数量的全局速率 | [marker_spi_source.f90:398](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:398)，398–412 | 对 species>0 shard 累加 `spi_abl*spi_species`（410），不评估空间核 |
| B. MPI local 数量 | 同文件 [422](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:422)，422–426 | 除 n_cpu；int(rate_local*gather/weight)；只有总 local 注入量>1e14 才保证至少 1 marker |
| C. 进入 birth sampler | 同文件 [433](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:433)，433–437 | uniform=.false.，传 fluid_source=rate_local、transform_rej_f=f_source_imp，传 R/Z bounds，不传 phi bounds |
| D. 每个空间候选 | [mod_initialise_particles.f90:497](/home/ac_desktop/JOREK_code/jorek/particles/mod_initialise_particles.f90:497)，497–512 | 圆柱均匀候选 → find_RZ 成功 → 插值 psi → total_imp_source |
| E. injector 内 shard 累加 | [mod_injection_source.f90:355](/home/ac_desktop/JOREK_code/jorek/models/mod_injection_source.f90:355)，355–418 | 对 radius>0 shard 调 inj_source（377–381）；按 species 分配并累加（404–411） |
| F. 跨 injector 累加 | [mod_initialise_particles.f90:513](/home/ac_desktop/JOREK_code/jorek/particles/mod_initialise_particles.f90:513)，513–520 | source_tmp += source_imp_arr(i_inj)；bg 另存 source_bg_tmp |
| G. 总源一次 transform | 同文件 [530](/home/ac_desktop/JOREK_code/jorek/particles/mod_initialise_particles.f90:530)，530–545 | ran(4) < transform_rej_f(source_tmp,phys_source*t_norm/n_norm)，通过后赋 birth 位置、速度、weight |
| H. transform 本体 | [marker_spi_source.f90:809](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:809)，809–817 | real*4 dv、f；dv=0.01*0.01*R_geo*(TWOPI/n_plane)；f=max(P*dv/P_norm,0)，没有 min(...,1) |

A 的 sum 是 atoms/s 标量，决定数量与 normalization；E/F 的 sum 是候选位置上的空间源。两者不能互换，也不能把 A 当作“shard 平台叠加”。空间核在 E 的每个 shard 上先求；f_source_imp 在 F 完成以后才求。初始化的接口和宏见 [mod_initialise_particles.f90:290](/home/ac_desktop/JOREK_code/jorek/particles/mod_initialise_particles.f90:290)，290–321；本路径要求 WITH_Impurities 对应的实现。

令 a_j=spi_abl_j、c_j=spi_species_j、q=m_i/m_imp、K_j(x)=source_shape_j、V_j 为归一化体积，注入已开启时：

```text
g_j(x) = max(a_j K_j(x) t_norm/(V_j n_norm), 0)   # inj_source 末尾负值归零
S_imp(x) = sum_injector sum_shard g_j(x) c_j/q
A_local = sum_shard a_j c_j / n_cpu
P_norm = A_local t_norm/n_norm
f(x) = real4(max(S_imp(x) dv/P_norm, 0))
p_accept(x) = min(1, f(x))                        # 有限非负 f 的概率语义
```

末行是 uniform 比较推导，并非 Fortran 显式写出的 min。接受还必须满足候选在有限元网格内。实际 birth 的归一化位置分布与圆柱体积测度下 `1_mesh(x)*p_accept(x)` 成正比，不是直接归一化未饱和的 S_imp。权重另由 `phys_source*timesteps/n_particle_asn` 设置为 real*4（539），饱和改变位置分布但不会在这里重算总注入数量。

## 2. 每 shard 到底有没有阈值、平台？

[mod_injection_source.f90:152](/home/ac_desktop/JOREK_code/jorek/models/mod_injection_source.f90:152)，152–173：先取 source_shape，再用数值 source_volume>0 或解析体积归一化；278–291：普通 SPI 分支线性乘 ns_amplitude、时间归一化，未开启时为 0，末尾负值置 0。没有任何正强度上限或平台阈值。

[mod_source_shape.f90:27](/home/ac_desktop/JOREK_code/jorek/models/mod_source_shape.f90:27)，27–43：

```text
K_pol = exp(-((R-R_j)^2+(Z-Z_j)^2)/sigma_j^2)
dphi = abs(phi-phi_j); if dphi>PI: dphi=2*PI-dphi
K_tor = exp(-(dphi/ns_deltaphi)^2)
K = K_pol*K_tor
if ns_delta_minor_rad>0:
    K *= exp(-(abs(psi-psi_j)/ns_grad_psi_j/ns_delta_minor_rad)^2)
```

高斯最大值为 1 是核函数自身的形状，不是把强度 min 到某个上限；其峰值强度仍按 a_j/V_j 改变。云宽 `max(radius_j*ns_radius_ratio,ns_radius_min)`（injection_source 371–375）是宽度下限，不是强度上限。解析圆云 V=PI*R_j*ns_tor_norm*sigma_j²；拉长云用 sigma_j*min(ns_delta_minor_rad,sigma_j)（167），这个 min 是体积公式。数值体积可能因边界变小，反而提高局部归一化强度（源码注释156–160）。这些均不能证明“逐 shard 平台”。

## 3. bg、imp 和 neutral 不是一条被混加的 birth 源

[mod_injection_source.f90:403](/home/ac_desktop/JOREK_code/jorek/models/mod_injection_source.f90:403)，403–411：bg += g_j*(1-c_j)，imp += g_j*c_j/mass_ratio；drift 分支各自同样分配。mass_ratio 在 [mod_initialise_particles.f90:358](/home/ac_desktop/JOREK_code/jorek/particles/mod_initialise_particles.f90:358)，358–368 中按 impurity 设置，Ne 为 central_mass/20。

[mod_initialise_particles.f90:512](/home/ac_desktop/JOREK_code/jorek/particles/mod_initialise_particles.f90:512)，512–531：bg 按 drift_distance 选普通或 drift 数组求和，但 f_source_imp 接收的是 source_tmp，即未 drift 的 impurity 数组之和。source_bg_tmp 没有进入此次 birth 接受判据。不能把 bg+imp 或 drift impurity 当作这里的 P。

[mod_neutral_source.f90:43](/home/ac_desktop/JOREK_code/jorek/models/mod_neutral_source.f90:43)，43–69、151–161：neutral_source 共用同一 source_shape/体积公式；时间未开启时 0；负源只打印警告（159–161），**没有** injection_source 的负值置零。其 total_neutral_source 在 [208](/home/ac_desktop/JOREK_code/jorek/models/mod_neutral_source.f90:208)，208–255 逐 shard 计算后累加，drift 出域时置 0；该函数这里的累加没有 impurity 的 species/mass_ratio 分配。

这不是 marker birth 在 imp_source 后又调用 neutral_source 的串联：WITH_Neutrals 的 use 存在于 initialise 299–301，但实际 neutral 调用块522–529全部被注释。共享形状模块是两条源分支的共同依赖，而不是 birth 中额外的一层阈值。

## 4. 角核与边界

- source_shape 32–34 使用最近环向角距离；对 phi、phi_j 位于同一 [0,2PI] 周期的情况，它等价于 Python 模运算最近角差。Fortran 本身没有对任意超出多周期的输入做 modulo，不能宣称任意角输入都严格相同。
- [model502/initialise_parameters.f90:209](/home/ac_desktop/JOREK_code/jorek/models/model502/initialise_parameters.f90:209)：ns_tor_norm=delta*sqrt(PI)*erf(PI/delta)，即最近周期高斯在一整圈的积分。原生 birth 是逐候选 phi 求角核；n_plane 进入 dv，并不令 source_shape 自动取环向均值。
- marker_spi_source 394–408：R/Z box 先以 pellets(1) 中心初始化，再对 species>0 shard 扩成 ±5sigma 的外包矩形。它不是逐 shard 独立裁切或候选归属判定；高斯核本身没有在5sigma处硬置零。若第一个 shard 为纯 D，初始中心也可能影响外包 box。
- [mod_initialise_particles.f90:391](/home/ac_desktop/JOREK_code/jorek/particles/mod_initialise_particles.f90:391)，391–414：R、Z 各自与全域 bounding box 相交，某个坐标无重叠则该坐标保留全域；phi 默认 [0,2PI]，调用端没有覆盖它。
- [mod_sampling.f90:74](/home/ac_desktop/JOREK_code/jorek/tools/mod_sampling.f90:74)，74–84：R=sqrt(u*(Rhi²-Rlo²)+Rlo²)、Z和phi线性均匀，故提议测度为 R dR dZ dphi。find_RZ 失败不接受（initialise501–502）；非矩形网格不能仅靠 box 判断。
- initialise476、570–573 重试所有未找到/未接受的槽；原生没有这里的有限 proposal budget。原生 drift 源额外在 plasmoid_in_domain 不为1时置0（injection385–394），与 birth 网格接受不同。

## 5. shard 消融率自身 min/max 是上游操作

[petlet 实际文件 pellet_module.f90:307](/home/ac_desktop/JOREK_code/jorek/models/pellet_module.f90:307)，307–313：消融更新半径，负半径置0；332–335：出域消融率置0。Ne Parks 分支 [706](/home/ac_desktop/JOREK_code/jorek/models/pellet_module.f90:706)，706–718：先对 ne_SI<0 置0，计算单 shard 消融率，若 nonlocal_abl 则取 max(local_rate,nonlocal_rate)；732–734随后可磁场缩减。负 spi_abl 在761–764报错停机。另一 NGS 分支485–490也取 local/nonlocal max。

这些操作决定传给 inj_source 的 a_j，发生在空间高斯、shard 源叠加和 birth 比较之前；max(local,nonlocal) 是选较大的物理消融率，不能解释为 min 的平顶接受概率。该 SPI 更新范围内没有发现将 spi_abl 以固定正上限 min 饱和的操作。文件里的其他 max（例如786的半径质量下限、827–832的分母保护）也不能混称为源平台。

对应本仓 [native_shard_dynamics.py:70](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_shard_dynamics.py:70)、77–93：半径下限、场输入非负保护、local/nonlocal rate max、磁场缩减属于消融更新，与 birth 的最终比较分开。

## 6. 小反例：总概率平台不等于逐 shard 平台

设两个 shard 在同一点的无量纲接受贡献均为0.7（使用同一个正 P_norm 和 dv），令 clip(z)=min(1,max(z,0))：

```text
clip(0.7+0.7)=1
clip(0.7)+clip(0.7)=1.4
```

后者甚至不能直接作为概率。若把逐 shard 截断之和再除以其自身积分，其位置分布也一般不同于总源截断之分布。此例只演示非线性次序；不是说原生流体源强度被 min 到1。逐 shard 负值归零也不能从公式中删除；本例用正贡献专门隔离上限饱和争议。

主代理补充：对于非负贡献及相同上限1，若最后仍作总截断，则
`min(1,sum_j min(1,f_j))=min(1,sum_j f_j)`，逐shard平台只是冗余。
本反例针对的是逐shard截断后直接叠加/归一化的替代目标，不应误称两重同阈值截断也不同。

## 7. NativeProductionBirths 当前对应性

| 项目 | 当前 Python 证据 | 评价 |
|---|---|---|
| 全局消融 sum/local/count | [native_source_birth.py:62](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_source_birth.py:62)，62–64 | 对应 A/B，包括1e14最少数量规则 |
| 逐 shard 核再总 sum | [native_source_density.py:94](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_source_density.py:94)，94–118；19–35 | live radius>0；amplitude含 rate/V 与Ne质量换算；高斯后按 shard 累加，没有逐 shard上限 clip |
| 总源 transform/比较 | [native_source_birth.py:119](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_source_birth.py:119)，119–124 | 总 rates 乘 central_mass/20 消去 mass ratio；float32(max(total*dv/norm,0))；ran[3]>=acceptance 拒绝，自然总概率饱和 |
| dv与weight精度 | 同文件20–23、93、130 | dv按default-real乘法和real*4赋值处理；norm=local_rate*t_norm/n_norm；weight为float32 |
| 默认候选与边界 | 同文件70–79、109–124 | R²均匀、Z/phi均匀、±5width外包box和各坐标无交集fallback、网格判据 |
| 显式表格实验 | [native_birth_positions.py:138](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_birth_positions.py:138)，138–159；native_source_birth31–43 | rates先全部累加，再float32下限、min到1，clip顺序正确；但角均值与数值表格有独立语义 |

**默认路径在已支持的纯 Ne、圆云、单周期配置下，与原生“先 sum 再 transform 再 birth 概率饱和”的顺序一致**。无需因“应逐 shard 平台化”的质疑改主代码。这个判断不等同于完整原生逐随机数、逐舍入复现：

1. density validate 43–45、68–69限制纯Ne、非负rate与圆云（elongated不支持），没有混合species/bg/drift和多injector分组结构。纯Ne时bg=0；对合法非负rate，原生inj_source负值归零自然不改变结果。Python线性展开与原生先分组sum的浮点求和及质量乘除顺序不保证bitwise相同。
2. 默认 rates_at(x) 用连续phi核（density31–32）；它没有调用 toroidal_average=True。可选 n0 表格则139–150先取连续角均值，再累加与clip；native_source_birth35–38要求axisymmetric_n0且n_plane=1。一般 mean_phi(clip(source(phi))) != clip(mean_phi(source(phi)))，因此该表格不能视为原生连续phi拒绝采样的严格等价加速。
3. 表格159先clip顶点，160–163积分三线性顶点插值的接受函数；有限网格下不是每个连续候选上直接clip真实高斯。文件1–6与configuration98–100明确实验/未production-qualified。正确的总clip顺序并不能证明表格空间误差为零。
4. Python默认119–124先做源拒绝再定位；原生501–531先定位再计算源。对无副作用且网格内核一致的候选，最终接受是同一合取判据；执行次序不同，不能称完整实际流程逐步骤复刻。
5. Python np.random默认每候选7个uniform（110），原生ran(4)（initialise330）与调用PCG32（marker434）；类docstring与configuration3–5、29已声明MT19937 distribution-only。还有有限proposal/槽容量预算（65–67、103–108），原生while循环没有同样终止预算。
6. Pythonbox对全部shard取外包，不按species>0筛；纯Ne时匹配该筛选，但为空、异常state、混合species不得外推。原生box先纳入pellets(1)中心；正常纯Ne集合中这一中心已被其±5sigma包含。
7. 原生接受表达式无单独mass换回；Python在已按20/central_mass表示的impurity密度上乘central_mass/20。这个全局正比例影响平台宽度，**它不是原生表达式逐字一致**：原生P为S_imp=sum(g_j*c_j/q)，P_norm=A_local*t_norm/n_norm，并未乘q。因此当central_mass!=20时，Python接受值为原生该表达式的q倍（纯Ne、其他条件相同）；count与weights仍一致，但平台与位置分布可能不同。不能因为density代码注释或类名就认定概率尺度完全对应。表格155同样乘q，存在同一差异。此为静态源码发现，本任务不改代码，后续如要修复应单独授权并用原生参考表达式验证。

### 7.1 调用者的 phys_source 是否有质量补偿：复核结果

在本报告指定的 `marker_spi_source → initialise_particles_marker` 路径中，**没有找到把 phys_source 换为 impurity mass 等效速率的补偿**。这不是仅凭参数注释判断：调用者410行累加 `spi_abl*spi_species`，422行只除 `sim%n_cpu`，436行将该值直接作为 `fluid_source`；被调用者350行初始化 `phys_source=0`，377行直接赋 `phys_source=fluid_source`。该子程序中没有随后乘/除 `m_i_over_m_imp` 的 phys_source 赋值。531行分母仍为 `phys_source*t_norm/n_norm`，539行weight也使用该原值。相反，364行定义的 `q=central_mass/20` 只经512行传入 `total_imp_source`，由 injection_source405行作用在空间源分子 `/mass_ratio`。transform 本体 marker_spi_source815行也没有质量补偿。

因此纯Ne、相同源核和归一化条件下，在最终 float32 舍入与概率饱和之前，代数关系为 `f_python = q*f_Fortran`。例如 central_mass=2 时 q=0.1；饱和后的概率不一定仍保持该比例。若所有候选都未饱和，统一正比例在条件接受的位置分布归一化中会抵消；一旦出现部分饱和，平台边界和归一化位置分布通常会不同。此差异没有被调用链中的 phys_source 质量换算消除；保持该静态发现，不改生产代码。

复核全部相关赋值/实参的只读命令：

```bash
rg -n 'phys_source|spi_source_imp_local|spi_source_imp|fluid_source|m_i_over_m_imp' /home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90 /home/ac_desktop/JOREK_code/jorek/particles/mod_initialise_particles.f90
rg -n 'phys_source\s*=|fluid_source\s*=' /home/ac_desktop/JOREK_code/jorek/particles --glob '*.f90'
```

因此最终判定应拆开：**求和/clip的非线性顺序对应；默认birth接受值的Ne质量比例未逐字对应；可选表格还有角均值/离散近似差异。** 上述差异不提供逐shard平台化的证据。

## 8. Source SHA256 与可复核命令

```text
4f6e832b107569d7a8a0155ce5995d04d59cc50771d256cdf99c1e692318991c  /home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90
269ce1cea5d5029c561f8ba93aacf4987f9c24163708f6df1a085cd6ba61f22d  /home/ac_desktop/JOREK_code/jorek/particles/mod_initialise_particles.f90
ebd1e97a2e384b5ed38d1ea15f13f77bee48bd2a377a1ad18f2cb0b5545e5eaf  /home/ac_desktop/JOREK_code/jorek/models/mod_injection_source.f90
e944921f84a90aed4d72a7fccd86fccfc1b9d9ab98d2a314b0f6d0b769fa86d3  /home/ac_desktop/JOREK_code/jorek/models/mod_neutral_source.f90
43db1e384ffc7f098771a2ff19f969eb30293b6226635fb473476fcc9ed205eb  /home/ac_desktop/JOREK_code/jorek/models/mod_source_shape.f90
6b9a9c78ccca3d504fcf2aae70792ea12d4b116738b474a36ffa7da3ef7c0e7a  /home/ac_desktop/JOREK_code/jorek/models/pellet_module.f90
25219d60fba943e3547ddf394b9c19a228970a2855ad3f50bb6ad1693884c7b2  /home/ac_desktop/JOREK_code/jorek/tools/mod_sampling.f90
50ba15d5c1437df44a5e62f26bb1e8c82140ef266f9f35c34bb5a1723119f93d  /home/ac_desktop/JOREK_code/jorek/models/model502/initialise_parameters.f90
8e45a70a0d5c08007383fda95bad31ac7a4c16ccc1f690366a90e1872354a6cc  src/models/native_source_birth.py
5e15982f88956fba13ec1eee585b5dec2fa22d35996eb2ae82a05e1189c1e351  src/models/native_source_density.py
7a947dc4ada1eef5f7fbfbbe57a861c07475d404d7cc58db7469b7cf848d4e39  src/models/native_birth_positions.py
cb3a40b0063cc1747f95ebdb3033982b236702026015df1baf8c2af8918572a5  src/models/native_shard_dynamics.py
```

在 L 中执行以下只读命令，可重算全部证据文件hash：

```bash
cd /home/ac_desktop/LNN-learning-for-JOREK
sha256sum \
  /home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90 \
  /home/ac_desktop/JOREK_code/jorek/particles/mod_initialise_particles.f90 \
  /home/ac_desktop/JOREK_code/jorek/models/mod_injection_source.f90 \
  /home/ac_desktop/JOREK_code/jorek/models/mod_neutral_source.f90 \
  /home/ac_desktop/JOREK_code/jorek/models/mod_source_shape.f90 \
  /home/ac_desktop/JOREK_code/jorek/models/pellet_module.f90 \
  /home/ac_desktop/JOREK_code/jorek/tools/mod_sampling.f90 \
  /home/ac_desktop/JOREK_code/jorek/models/model502/initialise_parameters.f90 \
  src/models/native_source_birth.py \
  src/models/native_source_density.py \
  src/models/native_birth_positions.py \
  src/models/native_shard_dynamics.py
git -C /home/ac_desktop/JOREK_code/jorek rev-parse HEAD
git -C /home/ac_desktop/JOREK_code/jorek status --short --untracked-files=all
git status --short --untracked-files=all
nl -ba /home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90 | sed -n '383,437p;809,817p'
nl -ba /home/ac_desktop/JOREK_code/jorek/particles/mod_initialise_particles.f90 | sed -n '290,321p;391,414p;497,545p'
nl -ba /home/ac_desktop/JOREK_code/jorek/models/mod_injection_source.f90 | sed -n '152,173p;276,291p;347,418p'
nl -ba /home/ac_desktop/JOREK_code/jorek/models/mod_neutral_source.f90 | sed -n '43,69p;149,161p;203,261p'
nl -ba /home/ac_desktop/JOREK_code/jorek/models/mod_source_shape.f90
nl -ba src/models/native_source_birth.py | sed -n '12,43p;62,79p;93,130p'
nl -ba src/models/native_source_density.py | sed -n '19,35p;94,118p'
nl -ba src/models/native_birth_positions.py | sed -n '138,163p'
rg -n 'min\(|max\(|spi_abl' /home/ac_desktop/JOREK_code/jorek/models/pellet_module.f90
```

无需构建JOREK、加载数据或执行训练即可复核。概率公式推导及反例为本报告推论，来源行已给出；不把静态审计包装成生产运行验证。

本次仅写入/修订 `report/native-birth-clipping-order-primary-evidence-2026-10-10.md`，保留用户全部既有变更。

