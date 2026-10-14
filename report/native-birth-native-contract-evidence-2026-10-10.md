# 原生 marker birth 契约：一手核查（2026-10-10）

结论：外层补齐被拒绝的 **marker 槽位**，不是逐 shard 独立生成。n_tor=n_plane=1 仍抽连续 phi 并计算 phi 核；对应原生接受分布的 n0 表应对 **总源完成 gate 后取 phi 平均**，即 meanphi(clip(total))。两 phaseA 的 deltaphi 都为 1000；明确不同的是 radius_min=0.05/0.01 m。半径平台、总烧蚀强度、gate 饱和必须分开。

仅只读 `/home/ac_desktop/JOREK_code/jorek` 源码及生产输入/日志，未执行 JOREK、训练或生成性能试验。按 research skill 启动后台交叉核查；已读 ADR README，未用历史 ADR 作 gate。以下源码事实对应当前文件，不声称生产二进制已由该 commit 重建。

## 1. 源码事实与执行顺序

简称 B=`particles/examples/marker_spi_source.f90`，I=`particles/mod_initialise_particles.f90`，T=`models/mod_injection_source.f90`，K=`models/mod_source_shape.f90`；均相对上述 JOREK 根。

| 环节 | 一手事实与实际行号 |
|---|---|
| 配额与名义 weight | [B:69](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:69) 每 rank 槽位容量 int(n_particles/n_cpu)。B:94–96 以 sum(spi_quantity)/n_particles 定义名义 weight（非 SPI 为 1e21/n_particles）。 |
| clock 与 batch | [B:352](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:352)：start=sim%time−旧 rest，n_steps=int((target−start)/timesteps)，余数留到下一 batch。B:79–86 初始将 timesteps 调整为流体步的整数分割，不能把 namelist 的 5e−9 当作始终精确的实际步长。 |
| rate 与出生数量 | [B:398](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:398) 对 species>0 的 shard 汇总 Q=sum(spi_abl*spi_species)，并扩展合并 RZ box（各方向 ±5 max(radius*ratio,radius_min)）；box 初值含 pellets(1) 中心。B:422–437：Q_r=Q/n_cpu，Δ=n_steps*timesteps，N_r=int(Q_r Δ/weight)；仅 Q_r Δ>1e14 时保底 max(N_r,1)，仅 N_r>0 调一次 I。无逐 shard 数量取整、无余数累加。 |
| 槽位重试 | [I:420](/home/ac_desktop/JOREK_code/jorek/particles/mod_initialise_particles.f90:420) 选 i_elm<=0 的前 N_r 个槽位；不足退出。I:476–573 对尚未找到的位置循环，失败槽位 pack 后重试，成功的不重生；无有限尝试上限。 |
| rank/RNG | [I:388](/home/ac_desktop/JOREK_code/jorek/particles/mod_initialise_particles.f90:388)、416–460：MPI_COMM_WORLD rank/size，rank0 seed 广播；每线程独立流，seq=rank*n_threads+thread+1，n_streams=n_cpu*n_threads，initialize(7,...)；每候选 next(ran(4))。birth 调用传 pcg32_rng()（B:434）。 |
| RZ/phi 提议 | [I:392](/home/ac_desktop/JOREK_code/jorek/particles/mod_initialise_particles.f90:392)、498–501：RZ box 与 domain box 分轴交集（某轴完全不相交会回退整个 domain 该轴），phi 默认 [0,2π]；B 未传 Phibound。 [sampling:81](/home/ac_desktop/JOREK_code/jorek/tools/mod_sampling.f90:81)：R=sqrt(U*(Rmax²−Rmin²)+Rmin²)，Z、phi 线性均匀，故提议测度为 R dR dZ dphi，不是平面均匀。 |
| mesh 判据 | [I:501](/home/ac_desktop/JOREK_code/jorek/particles/mod_initialise_particles.f90:501) 先 find_RZ，ifail==0 才算源。[find_RZ:19](/home/ac_desktop/JOREK_code/jorek/grids/grid_utils/find_RZ.f90:19) 用 i_plane_rtree 的固定几何 phi，不使用候选 phi；:76–97 搜候选元素，第一个成功返回。:128–218 为五初值、最多20次 Newton，每维步长限 ±0.25，坐标压至 [0,1]；errf<=1e−15 **或** errx<=1e−8 返回成功。不能将其简写为严格多边形包含或只有残差门槛。 |
| source inner 循环 | [I:511](/home/ac_desktop/JOREK_code/jorek/particles/mod_initialise_particles.f90:511) 每个 mesh 成功的候选先插值 psi，再调用 T；T:347–416 逐 injector、再逐该组 n_spi(i_inj) shard，radius>0 才调用 inj_source；I:513–520 再逐 injector 汇总 impurity 与 background。因此每候选做 sum(n_spi) 次 shard 检查，基础 inj_source 调用数为有效半径 shard 数；非零 drift 且 plasmoid 在域可另调一次 shifted 源。后台/当前全树检索未见名为 sourceinner 的原生符号，此处按“候选内的源循环”解释。 |
| mass-source gate/截断 | [T:361](/home/ac_desktop/JOREK_code/jorek/models/mod_injection_source.f90:361) radius gate→宽度下限→inj_source。T:183–285 时间 gate（SPI 调用 L_tube=0，因此 t_now>t_ns），T:289–291 对每次 inj_source 输出负值归零，T:404–405 按 species 分配并将 impurity 除 mass_ratio。I:513–531 汇总后 f_source_imp；[B:813](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:813) dv=0.01² R_geo(2π/n_plane)，f=max(P*dv/P_norm,0)，f/dv 为 real*4；P_norm=Q_r*t_norm/n_norm。比较 ran(4)<f，无显式 min(f,1)，但随机数范围给出隐式上限饱和。这里的 mass-source gate 指该接受函数，不是用 atom-number source 替换 P。 |
| newborn 状态 | [I:532](/home/ac_desktop/JOREK_code/jorek/particles/mod_initialise_particles.f90:532) 写 x=[R,Z,phi]、i_elm、st；calc_VBpsiU(time=sim%time) 后 V_prev=V，weight=real(Q_r Δ/N_r,4)，P_imp 全零且 P_imp(0)=1，i_life+=1。不抽出生时刻、不写单独 age；整批出生在子步循环前，随后 [B:482](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:482) 同现存粒子推进全部 n_steps，子步 clock 为 start+(k−1)timesteps。源使用 T 的全局 t_now，不用该子步 t。 |

后台交叉核查补充：粒子基类确有 [t_birth 字段:62](/home/ac_desktop/JOREK_code/jorek/particles/mod_particle_types.f90:62)，默认0，但 I 整个 birth 子程序没有更新它；槽位复用可能保留旧值，不能据其推断本次出生时刻。weight 字段是 real*8（:59），赋值前先降至 real*4。PCG 的 [initialize:26](/home/ac_desktop/JOREK_code/jorek/tools/mod_pcg32_rng.f90:26) 不使用 n_dims/n_streams 生成维数；[next:51](/home/ac_desktop/JOREK_code/jorek/tools/mod_pcg32_rng.f90:51) 与底层 mod_pcg32.f90:44 按 out 大小生成，故传 initialize(7,...) **不代表每候选消耗7个 uniform**，这里实际生成4个。分布推导假定未启用该底层 :46 的固定 RNG 调试覆盖。

clock 补充：[stepper:328](/home/ac_desktop/JOREK_code/jorek/core/mod_jorek_timestepping.f90:328) 初次设置 t_now=sim%time/sim%t_norm；:418 在 t_now>=t_ns 时更新 shard，:426 才推进 t_now。更新 shard 的 >= 与空间源的 > 在精确开启边界不同；不能把 birth rate、源 gate、场查询和子步 clock 视作同一时间字段。

数量端的 species>0 筛选与空间端 radius>0/time gate 不是同一筛选；某 batch 若 Q_r>0 但空间有效源全零，重试可能不终止。出生权重总和约 Q_r Δ（real*4 舍入），不会按 mesh/box 丢失尾部扣除 rate。纯 Ne 的 mass_ratio=central_mass/20（I:363–364）；将 T 返回值乘回 mass_ratio 后再 gate 会改变饱和程度。rank 数还通过 Q_r 改变 gate 尺度，不能只理解为随机流差异。

补充：流体 model502 的 [matrix:1219](/home/ac_desktop/JOREK_code/jorek/models/model502/mod_elt_matrix.f90:1219) 也先汇总 injector，再 max(source_imp,0)（:1237）；FFT 对应 :1234–1253。这是下限归零，不是 birth 的概率上限；不要将二者混称 clip。

## 2. 等分布推导：n0 probability 表应表达什么

**源码事实**：[settings:5/10](/home/ac_desktop/JOREK_code/jorek/models/mod_settings.f90:5) n_tor=n_plane=1；[K:28–42](/home/ac_desktop/JOREK_code/jorek/models/mod_source_shape.f90:28) 仍乘 exp(−(wrapped_dphi/ns_deltaphi)²)，无 n_tor/n_plane 分支；可选 minor-radius 核还依赖 psi。环向归一化 [502 init:209](/home/ac_desktop/JOREK_code/jorek/models/model502/initialise_parameters.f90:209) 为 d sqrt(π) erf(π/d)。wrap 只作一次 dphi>π→2π−dphi，不是任意实数角度的完整 modulo。

**推导，非源码实现**：固定 batch、rank 和源状态，设 G(R,Z) 表示原生 mesh 成功，P 为候选内汇总的 **mass source**，c=dv/P_norm，a=min(1,max(cP,0))（精确原生 gate 还含 real*4 舍入）。连续 RNG 理想化下接受密度 ∝1_box G R a(R,Z,phi)。axisymmetric geometry 下 RZ 边缘 ∝1_box G R meanphi(a)。所以概念对应为 **meanphi(clip(c*total))**，不是 clip(c*meanphi(total))，也不是 sum_shard clip(c*shard)。phi-dependent geometry 则须将 G 放进平均，不能照搬。

对非负源，min(1,x) 为凹函数，meanphi(clip(x))<=clip(meanphi(x))；只有未跨越饱和阈值（全不饱和或全饱和），或 phi 常数等条件才可交换。例：半周 x=0、半周 x=2，前者0.5、后者1。未饱和时，按 **测度积分后的分量质量** 混合 shard 可以等分布；有总源饱和时，独立 shard 生成/逐 shard clip 一般不等分布。这只说明空间分布，不等价于原生 batch 数量、槽位、MPI、RNG 顺序和时间推进。

新 n0 表若只保留 RZ，可精确表达上述边缘（仍需正确 G、柱坐标测度、质量尺度与归一化）。若还要生成原生三维 birth，phi 应按 a/∫a dphi 的条件分布抽样；用独立均匀 phi 是近似。网格 cell/节点积分、phi quadrature、finder 替换和 float64 gate 都应另列数值近似，不能称 bitwise 原生。

## 3. 两 phaseA 与“强度平台”

输入一手证据：两份 `data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400{,_injInner}/jorek_namelist_spitest` 第501行均 deltaphi=1d3，第516行 ratio=1.4，第517行 radius_min 分别0.05/0.01。日志 v400:2874–2877、2909–2913 与 injInner:2706–2709、2741–2745 确认 d=1000、minor_rad=0、spi_num_vol=T、n_inj=1、n_spi=846。因此候选内基础 source 循环检查846个 shard；不是846个 birth batch。日志2940/2772 n_particles=1e6、2929/2761 quantity=7e21，对应名义 weight=7e15。这里只确认现存输入/日志，未外推每段历史重启。

**近似界**：d=1000 时 phi 核在 [exp(−π²/1e6),1]，相对变化最多约9.87e−6；phi 常数化可为小误差近似，但源码仍有 phi 核，有限 quadrature 也不等于解析 mean。两 run 没有 deltaphi 值的差异，phi 中心/空间位置不同可改变点值；若比较 finite-d=1000 与无限宽 n0，这是算法近似差异。

**平台区分**：σ=max(1.4*r_shard,σmin)；r_shard<=0.035714 m（v400）或0.007143 m（injInner）后是 **宽度平台**。Q=spi_abl*species 仍随烧蚀变化；没有由 σmin 固定 Q 的源码规则。圆云、解析体积、同 R_shard/d/Q 且未饱和时，峰值 ∝Q/σ²，故0.01与0.05的理想峰值比25；但生产 spi_num_vol=T，T:161–173 优先用正的 spi_vol，不能未经体积/边界核查就宣称生产峰值25倍。总源重叠、rank、质量系数又可能让 a=1 形成 **概率平台**；大 deltaphi 给的是 **phi 平台**。这些平台都不代表烧蚀强度恒定，也不足以判断源已达稳态。

## 4. SHA 与支持范围

JOREK git HEAD：`54c675bcc39c33b082c0c35046e28122dc53b3aa`。以下 SHA-256 标识实际读取文件（HEAD 不替代文件指纹）：

| 文件 | SHA-256 |
|---|---|
| B | `4f6e832b107569d7a8a0155ce5995d04d59cc50771d256cdf99c1e692318991c` |
| I | `269ce1cea5d5029c561f8ba93aacf4987f9c24163708f6df1a085cd6ba61f22d` |
| T | `ebd1e97a2e384b5ed38d1ea15f13f77bee48bd2a377a1ad18f2cb0b5545e5eaf` |
| K | `43db1e384ffc7f098771a2ff19f969eb30293b6226635fb473476fcc9ed205eb` |
| tools/mod_sampling.f90 | `25219d60fba943e3547ddf394b9c19a228970a2855ad3f50bb6ad1693884c7b2` |
| grids/grid_utils/find_RZ.f90 | `95835d8183228a5540dcaa6ae1249d8e509163de1f462aaa1e1fe2f3e01578cf` |
| models/mod_settings.f90 | `bf7ae97ce087795a401140b9843c2f4d85976b3d68dfa4cc975cc34abceceaf7` |
| models/model502/initialise_parameters.f90 | `50ba15d5c1437df44a5e62f26bb1e8c82140ef266f9f35c34bb5a1723119f93d` |
| models/model502/mod_elt_matrix.f90 | `5a47f3d314cf7251543d999e13109c5e781b31491eb94469fb826777432f814d` |
| models/model502/mod_elt_matrix_fft.f90 | `f99027050d92d623b9d963281ccd65567b79a657a19da0247021e8e860548943` |
| particles/mod_particle_types.f90 | `14fb42d89cf6e0ed55de5bfda4122aa230da4c67c1f4385abc92e1db9197b421` |
| core/mod_jorek_timestepping.f90 | `f45b62fff5165f2c0f728c7a24d14b919628ff455ed4d1497daafaecceecb5fe` |
| tools/mod_pcg32_rng.f90 | `a6009b65f7e95e60b284c273303ac5e5ff053aba22d02fca3f192c6062bb89b1` |
| tools/mod_pcg32.f90 | `8d37a21cbb86ff3d3f567d98a90da955a054287cfb4cbb0f51c505e0c5bf64f8` |
| v400/jorek_namelist_spitest | `0fca2b6a5d0cf94e92eb86a4ef11d961fba50a28d06520708e19dfec3d061811` |
| injInner/jorek_namelist_spitest | `bf3de403f17ab713d54f51883eee72699fa9ff3e5dd0549588bfffbd6c6e2c0e` |

支持范围：WITH_Impurities 的非均匀 marker SPI 路径及当前两个输入/日志；未证明任意混合物、多 injector、elongated/drift 情况的降阶实现正确，未审核任何替代实现源码，也未作统计/速度验收。精确的 native-contract 声称须同时保留 gate 顺序、mass 尺度、mesh 判据、box、rank、数量/weight 与 clock；仅空间等分布不能替代这些契约。
