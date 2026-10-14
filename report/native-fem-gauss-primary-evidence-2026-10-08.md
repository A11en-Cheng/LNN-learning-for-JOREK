# 原生自由度→高斯点→局部两项载荷：一手源码证据（2026-10-08）

本报告使用 research 技能核查本地只读 JOREK；依据现行 ADR-0018/0019/0020，roadmap_v2 未修改。本文只记录一手源码与生产元数据，数值实现和测试结果由对应实现报告登记。验收边界只能是“完整自由度求值与局部两项载荷参考链”，不是完整 FEM 装配或九场耦合。

## 来源和版本

根目录 `/home/ac_desktop/JOREK_code/jorek`，`git rev-parse HEAD` 为 `54c675bcc39c33b082c0c35046e28122dc53b3aa`。只读核查时该工作树存在既有 `communication/eqdsk2jorek.f90` 修改及未跟踪 `poincare`；本报告所核查源码文件无工作树修改。没有修改 JOREK。

| 文件（相对源码根） | SHA-256 |
|---|---|
| elements/mod_basisfunctions.f90 | ccb43c0e9eb18db876029ec07dec1ffaa2e6a9700218f2b7f48a9320092f0788 |
| elements/basis_at_gaussian.f90 | 6c430594f289d2cb13667372b9c91731469575228c84033cdffdf6a89d44c953 |
| elements/gauss.f90 | b6467693d9e544e49376eed64cd2528c1075ee3ba33dd359e854ca1384bcd707 |
| models/model502/mod_elt_matrix.f90 | 5a47f3d314cf7251543d999e13109c5e781b31491eb94469fb826777432f814d |
| models/corr_neg_include.f90 | 5be1a4cb976426438068aaddb4c4aec52eb2157ed775b6cd53f4514203270c73 |

## 实际生产分支

`matrix/construct_matrix_mod.f90:63–82`：完整环向块且 `n_tor>=n_tor_fft_thresh`，或 unified 为真，走 FFT；否则普通 `element_matrix`。`models/preset_parameters.f90:14` 阈值为2；`models/model502/mod_model_settings.f90:55` unified 为假。原始 phaseA v400 `jorek004110.h5` 为 model502、n_order=3、n_degrees=4、n_tor=n_plane=n_period=1，因此本次核查普通路径，绝不混入 FFT 中不同的投影下限或归一化。

生产日志 `data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400/logfile_spitest:2243` 明确记录 `GAUSS_ORDER : Preprocessor flag not set. Thus, n_gauss=4`。这是该生产运行使用4点规则的直接日志证据，不只是根据当前源码默认值猜测。8点规则是源码支持的另一个显式配置，不能自动声称该 run 使用8点。

## 基函数、布局和几何

令 `A0(s)=(1-s)^2(1+2s)`，`B0(s)=3s(1-s)^2`，`A1(s)=s^2(3-2s)`，`B1(s)=3(1-s)s^2`。实际 cubic 是 mixed Bezier/cubic 基，不是忽略3因子的标准 Hermite 导数插值。`B0'(0)=3`，`B1'(1)=-3`，因此原生 degree 系数不能直接当成局部坐标导数。`elements/mod_basisfunctions.f90:40–51,76–86,141–164` 给出值、导数和完整2D展开。

四顶点顺序 `(0,0),(1,0),(1,1),(0,1)`；每顶点 degree 为 `(A_s A_t, B_s A_t, A_s B_t, B_s B_t)`，其中 A/B 按顶点端点取0或1。包括全部4个 degree、全部 harmonic 的线性求值才是原生求值；dof0或双线性替代不是无损变换。源码另有 quintic 分支，若接口仅支持生产 cubic，必须对 n_order≠3 明确失败而非悄悄截断。

| 输入→公式 | 消费位置 | 单位/语义 | 源码 |
|---|---|---|---|
| 节点坐标 `x(1,j,1/2)`→`Σ x*size*H` | R,Z及一阶/二阶几何 | R,Z为m；s,t无量纲 | M263–277 |
| 节点全 values→`Σ values(in,j,k)*size(i,j)*H*HZ` | eq_g、eq_s、eq_t、eq_p及二阶极向导数 | 各流体原生归一化量 | M285–292 |
| 全投影 values→同一基与size、HZ | eq_aux_g/s/t/p | 五通道原生单位，不先裁dof | M294–298 |
| 全投影 deltas→同一基与size、HZ | delta_aux_g/s/t | 原始投影增量，先求值后局部缩放 | M301–303 |
| 全流体 deltas→同一基与size、HZ | delta_g/s/t | 独立于投影增量 | M306–308 |
| `(R_s Z_t-R_t Z_s)`→J | 物理导数、两项载荷 | 有符号m²，不自动abs | M353 |
| `f_R=(Z_t f_s-Z_s f_t)/J`；`f_Z=(-R_t f_s+R_s f_t)/J` | 流体/反馈物理导数 | 每m | M410–411,999–1000等 |

此表 M 指 `models/model502/mod_elt_matrix.f90`。该普通 model502几何只消费坐标 harmonic1（Fortran索引），没有使用 HZ_coord；非轴对称几何不能套此公式冒充完整三维几何。

`grids/grid_utils/finish_grid.f90:85–89` 先将节点切向量归一化，`:115–128` 根据弦长除 n_order 和切向朝向生成**带符号**的 size；`:134–135` size1=1、size4=size2*size3。原生检查点已保存 size，不应在读取时根据四角重算或取绝对值；轴等位置可有零size（同文件549–550）。`temporary_element_sizes.f90` 明确是临时绘图用，不能替代 finish_grid 或已保存原生size。

导出布局由 `communication/mod_export_restart.f90:404–407,491–493,608–617,687–692` 确定。Fortran t_x为 `[node,coord_harmonic,degree,component]`，t_values/t_deltas为 `[node,harmonic,degree,field]`；h5py看到反向维度。真实004110：`x=(2,4,1,4125)`、`values=deltas=(9,4,1,4125)`、`size=(4,4,3986)`（degree,vertex,element）、`vertex=(4,3986)`（原生节点编号1起）。不能把 key x误记为coords；坐标字段必须完整保留。

## Gauss和环向因子

`elements/gauss.f90:4–8,22–27` 的生产4点规则在 `[0,1]`，权重和1；点为 `.0694318442029735,.3300094782075720,.6699905217924280,.9305681557970265`，权重 `.173927422568727,.326072577431273,.326072577431273,.173927422568727`。8点规则在该文件11–20显式给出。参考复现可使用这些常数，避免隐式映射额外乘1/2。

`elements/basis_at_gaussian.f90:76–91`：`phi_p=2π(p-1)/(n_plane*n_period)`，物理 harmonic 为常数1、cos(mode*phi)、sin(mode*phi)，phi导数为0、`-mode*sin`、`+mode*cos`。坐标 sine 使用相反号（94–100），但本次普通model502几何根本不消费非轴对称坐标谐波。

M347–368逐s/t及mp求和；M351仅 `wst=wgauss(ms)*wgauss(mt)`，M1427测试函数 `v=H*size*HZ`，M1854 `RHS(ij9)+=rhs_ij_9*wst`。从函数开始到3120返回，普通路径没有环向 `1/n_plane` 或 `2π/n_plane` 因子。这是**源码局部向量约定**，不是物理全环体积分；下游若需要全环积分须另行定义，不能在此载荷偷偷加因子。

## 原始与修正流体量

M409/426/471分别取原始 rho/rho_imp/Te。M419/436调用密度修正：`L1=1e-3*1e-9=1e-12`，`L2=1e-3*1e-5=1e-8`；若 val<L1+L2，修正为 `L1+L2*exp((val-L1-L2)/L2)`，否则为原始val。具体实现 `models/corr_neg_include.f90:309–329`。

M492–500电子温度修正 floor为 `max(T_min,Te_1)`（JU），L1=L2=floor/2，同样指数修正（include78–98）。M937规定 `Te_eV=Te_JU/(e*mu0*central_density*1e20)`。不要依据 phys_module中旧注释或namelist旁注反推错误的温度单位。

`models/model502/initialise_parameters.f90:182,189–195,277` 依次preset、读namelist、读数值profiles；`models/read_num_profiles.f90:40–46`用电子温度profile最后一个点覆盖Te_1。生产namelist:411为Te_file=jorek_e_temperature，:442为T_min=3.2085e-4；生产log:2653 Te_1=1e-5、:2769 T_min=3.2085e-4。preset_parameters:514默认Te_1=5e-9不能覆盖实际profile结果。HDF5不保存上述全部阈值，适配器必须要求显式有来源的配置，不能从缺失字段猜测。

闭合电子密度M1081–1082使用rho_corr+alpha_e*rho_imp_corr；下面历史载荷使用的却是**原始**Te0和rn0。这两者不能交换。

## 局部两项载荷与一次性因子

1. 直接电子能量项（M1728）：`L_energy[a]=Σ_q -v_a(q)*R(q)*J(q)*dt_JU*(C1(q)+max(C2(q),0))*w_s*w_t`。C1保留符号，C2截零（M989–990）；dt为JOREK归一化时间。没有theta、zeta、额外gamma−1或全环2π因子。五通道原始能量率已归一化；若dt输入为SI只转换一次 `dt_JU=dt_SI/t_norm`。
2. marker投影历史项（M1788）：`L_history[a]=Σ_q -v_a*R*J*Te_raw*rho_imp_raw*delta_alpha_e*w_s*w_t`。没有额外dt、theta、zeta因子。M324–326先把投影delta局部乘dt/dt_prev；M1005–1007在截断/下限后的Q,N基础上算 `delta_Q=(scaled_delta_C4-Q*scaled_delta_C5)/N`；M1079令delta_alpha_e=mass_ratio*delta_Q。一次缩放只在消费组件执行，原生保存数组保持原始值。M1021–1028电荷触顶清空间导数而不清delta_Q；低温后处理同样不能凭数学平滑截断导数重写代码顺序。

M1784–1786另外三项为**流体**历史增量时间项，有zeta；它们不是本次marker投影历史载荷，不能混合。M1789–1790带zeta的投影交叉历史项被注释，不能算入实际生产公式。

局部测试函数包含原生 signed size和HZ；此处只算电子方程9的这两项，不计算所有反馈闭合引起的对流、输运、LHS和联立影响。全局约束、边界/磁轴、矩阵装配、求解器及更新映射不由本报告核查结果自动完成。

## 证据边界

已确认：本地源码的完整cubic自由度公式、signed size、轴对称几何映射、一阶/二阶极向基及物理导数、phi基、4/8点规则和实际生产4点证据、原生HDF5维序、原始/修正流体、两项局部载荷精确因子。

尚待实现报告登记：独立解析/Fortran oracle数值比对、真实phaseA求值与注册restart零反馈贯通测试。只读原生文件布局核查不是数值求值验证。源树HEAD不等于生产二进制构建指纹；虽然日志确认Gauss4与实际参数，未获得完整生产二进制哈希与全部编译选项。quintic和非轴对称几何、FFT装配、全局矩阵与九场求解均不得作为本轮已验收能力。
