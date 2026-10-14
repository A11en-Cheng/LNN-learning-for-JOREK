# JOREK marker 源、model502 消费与 restart 时序：一手证据（2026-10-07）

## 范围与证据等级

源码根目录为 `/home/ac_desktop/JOREK_code/jorek`，检查时 HEAD 为 `54c675bcc39c33b082c0c35046e28122dc53b3aa`。工作树另有 `communication/eqdsk2jorek.f90` 修改和未跟踪的 `poincare`；本报告重点读取的 `models/model502/mod_elt_matrix.f90`、`particles/examples/marker_spi_source.f90` 未显示本地修改。两文件本机 SHA-256 分别为 `5a47f3d314cf7251543d999e13109c5e781b31491eb94469fb826777432f814d`、`4f6e832b107569d7a8a0155ce5995d04d59cc50771d256cdf99c1e692318991c`。下文「源码」表示可由所列路径和行号直接复核；「实测」表示主代理已在本地通过仓库符号链接 [`data/EHL-2`](../data/EHL-2)（指向 `/mnt/d/EHL-2/`）直接检查两组生产 run 原始文件。`data/restart_events.json` 不作为原始证据。

## 1. 五通道怎样生成并投影

`marker_spi_source` 先从 `jorek_restart.h5` 读取流体场；`restart_particles` 为真时另从 `part_restart.h5` 读粒子，否则创建粒子。时间换算为 `t_norm=sqrt(μ₀ρ_norm)`、`tstep_si=tstep*t_norm`。随后创建带滤波参数的 `jorek_feedback` 投影，`use_marker` 分支分配第五维为 5 的 RHS，并将其节点列表赋给全局 `aux_node_list`。[marker_spi_source.f90:75](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:75)、[marker_spi_source.f90:100](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:100)、[marker_spi_source.f90:145](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:145)

粒子子步使用旧位置/旧单元 `i_elm_old`，在 `particle_start_time+(k−1)*timesteps` 采样流体场。电离能变化从电荷态概率变化乘逐级电离能及粒子权重求和，再乘 `EL_CHG`；辐射损失按权重、辐射系数及子步时长累计。三个矩分别是 `weight*⟨q²⟩*timesteps`、`weight*⟨q⟩*timesteps`、`weight*timesteps`。[marker_spi_source.f90:482](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:482)、[marker_spi_source.f90:531](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:531)、[marker_spi_source.f90:543](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:543)、[marker_spi_source.f90:598](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:598)

五通道 RHS 顺序及源码中的显式归一化如下；每项还乘极向基函数 `HH`、单元 `size` 和环向模基 `HZ` 后累加到 `feedback_rhs`。这里列的是进入 FEM 求解之前的载荷，不是导出的节点场。[marker_spi_source.f90:614](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:614)、[marker_spi_source.f90:628](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:628)、[marker_spi_source.f90:637](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:637)

| 通道 | 粒子侧量 | RHS 因子（另乘 `HH*size*HZ`） | model502 解释 |
|---|---|---|---|
| 1 | `dEion_dt` | `t_norm/E_norm` | 电离能损失率 |
| 2 | `rad_sink` | `t_norm/E_norm` | 辐射功率密度 |
| 3 | `Z_eff`（加权 `q²` 矩） | `1/n_norm` | 除以通道 5 后的 impurity `q²` 均值 |
| 4 | `Z_imp`（加权 `q` 矩） | `1/n_norm` | 除以通道 5 后的平均电荷态 |
| 5 | `N_imp`（加权粒子数） | `1/n_norm` | 时间平均 impurity 数密度 |

主循环令 `rhs_gather_time=n_steps*timesteps`，清空当前 `feedback_rhs`，完成粒子子步后在 `use_marker` 分支将**全部五通道**除以该时长，再将 gather time 清零。故 3–5 通道也按此窗口做时间平均；1–2 通道是窗口能量损失换算后的速率。`mod_uncoupled_projection` 中 `/rhs_gather_time` 只保留为注释，不能再除一次；其类型说明亦称 RHS 在采样时累积、投影前做 MPI 合并。[marker_spi_source.f90:442](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:442)、[marker_spi_source.f90:669](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:669)、[mod_uncoupled_projection.f90:144](/home/ac_desktop/JOREK_code/jorek/particles/diagnostics/mod_uncoupled_projection.f90:144)、[mod_projection_type.f90:35](/home/ac_desktop/JOREK_code/jorek/datatypes/mod_projection_type.f90:35)

投影事件先采样 `rhs_f`、调用 `project_only`，再按 `nout_projection` 决定是否写 HDF5。后者把单元 RHS 按全局 dof 索引装配、MPI `Reduce`，求解稀疏系统 `A x=b`，将解存入节点 `values`、解的差存入 `deltas`，再广播节点。`A` 由带质量项及可选垂直、超扩散、平行滤波项的 FEM 矩阵构成；所以 `projections*.h5` 的 `values` 是滤波 FEM 解，不是粒子核直接相加的 RHS。写出器从节点 `values` 存 `values`、从 `sim%time` 存 `t_now`。[mod_project_particles.f90:232](/home/ac_desktop/JOREK_code/jorek/particles/diagnostics/mod_project_particles.f90:232)、[mod_uncoupled_projection.f90:134](/home/ac_desktop/JOREK_code/jorek/particles/diagnostics/mod_uncoupled_projection.f90:134)、[mod_uncoupled_projection.f90:202](/home/ac_desktop/JOREK_code/jorek/particles/diagnostics/mod_uncoupled_projection.f90:202)、[mod_uncoupled_projection.f90:224](/home/ac_desktop/JOREK_code/jorek/particles/diagnostics/mod_uncoupled_projection.f90:224)、[mod_uncoupled_projection.f90:258](/home/ac_desktop/JOREK_code/jorek/particles/diagnostics/mod_uncoupled_projection.f90:258)、[mod_uncoupled_projection.f90:588](/home/ac_desktop/JOREK_code/jorek/particles/diagnostics/mod_uncoupled_projection.f90:588)、[mod_project_particles.f90:544](/home/ac_desktop/JOREK_code/jorek/particles/diagnostics/mod_project_particles.f90:544)、[mod_project_particles.f90:583](/home/ac_desktop/JOREK_code/jorek/particles/diagnostics/mod_project_particles.f90:583)

本地原始样本 [`injInner/projections002704.h5`](../data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner/projections002704.h5) 含 `t_now` 和 `values`；`values.shape=(5,4,1,4125)`（以 `h5py` 所见的轴顺序），文件根及 `values` 数据集均无属性来说明通道名称或单位。生产者写出数组的 Fortran 轴序为 `(n_nodes,n_tor,n_degrees,n_fields)`；`n_dof` 另存为标量，通道意义只能由上面的生产者和消费端源码建立，不能由 HDF5 自描述推出。若取 `values[:,0,0,:]`，所得是**五通道的第 0 个多项式 degree、环向索引 0 切片** `(5,4125)`，俗称 dof0 切片；它不是包含四个 degree 的全 `values`，也不能以此切片代替全 dof 场（文件里的 `n_dof` 是全局自由度计数，不是此轴名）。当前源码 `n_order=3,n_tor=1` 与样本的中间两轴一致。[mod_project_particles.f90:535](/home/ac_desktop/JOREK_code/jorek/particles/diagnostics/mod_project_particles.f90:535)、[mod_project_particles.f90:544](/home/ac_desktop/JOREK_code/jorek/particles/diagnostics/mod_project_particles.f90:544)、[mod_project_particles.f90:578](/home/ac_desktop/JOREK_code/jorek/particles/diagnostics/mod_project_particles.f90:578)、[mod_project_particles.f90:583](/home/ac_desktop/JOREK_code/jorek/particles/diagnostics/mod_project_particles.f90:583)、[mod_settings.f90:5](/home/ac_desktop/JOREK_code/jorek/models/mod_settings.f90:5)

## 2. model502 如何消费：RHS 源与系数分开

`aux_node_list => jorek_feedback%node_list` 使矩阵装配读取投影节点。`construct_matrix` 将全局 `aux_node_list` 节点拷贝给 `elementary_matrix_build`；model502 在高斯点以 `aux_nodes%values` 构造 `eq_aux_g/s/t/p`，并读 `deltas` 构造增量。[marker_spi_source.f90:151](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:151)、[construct_matrix_mod.f90:522](/home/ac_desktop/JOREK_code/jorek/matrix/construct_matrix_mod.f90:522)、[mod_elt_matrix.f90:294](/home/ac_desktop/JOREK_code/jorek/models/model502/mod_elt_matrix.f90:294)

marker 分支中，`eq_aux_g` 的通道 1、2 分别赋给 `aux_dEion_dt`、非负截断后的 `aux_rad`；电子能量方程 `rhs_ij_9` 明确减去二者之和，这是五通道中的直接 RHS 源项。通道 3–5 则先取非负、给 `n_imp` 下限 `0.5`、以通道 5 分母恢复 `Z_imp`/`Z_eff`，再用于 `alpha_e`、其空间导数/增量、库仑对数与电阻率等系数计算；不能把这三项整体称作加到某个 RHS 的独立外源。注意另一条 FFT 装配路径的 `n_imp` 下限是 `1e−2`，因此本报告的 `0.5` 只适用于这里核对的 `mod_elt_matrix.f90` 路径。[mod_elt_matrix.f90:983](/home/ac_desktop/JOREK_code/jorek/models/model502/mod_elt_matrix.f90:983)、[mod_elt_matrix.f90:998](/home/ac_desktop/JOREK_code/jorek/models/model502/mod_elt_matrix.f90:998)、[mod_elt_matrix.f90:1065](/home/ac_desktop/JOREK_code/jorek/models/model502/mod_elt_matrix.f90:1065)、[mod_elt_matrix.f90:1079](/home/ac_desktop/JOREK_code/jorek/models/model502/mod_elt_matrix.f90:1079)、[mod_elt_matrix.f90:1128](/home/ac_desktop/JOREK_code/jorek/models/model502/mod_elt_matrix.f90:1128)、[mod_elt_matrix.f90:1159](/home/ac_desktop/JOREK_code/jorek/models/model502/mod_elt_matrix.f90:1159)、[mod_elt_matrix.f90:1723](/home/ac_desktop/JOREK_code/jorek/models/model502/mod_elt_matrix.f90:1723)、[mod_elt_matrix_fft.f90:999](/home/ac_desktop/JOREK_code/jorek/models/model502/mod_elt_matrix_fft.f90:999)

同一 model502 还有 `total_imp_source` 产生的 `source_bg/source_imp`，以及 `particle_source`、`source_pellet`，它们在密度等方程 RHS 中出现，属于另一条源链，不应与 marker 五通道混同。在 `use_marker` 分支，流体侧的辐射 `Lrad`、`dLrad_dT` 被置零，辐射损失改由投影通道 2 提供。[mod_elt_matrix.f90:1214](/home/ac_desktop/JOREK_code/jorek/models/model502/mod_elt_matrix.f90:1214)、[mod_elt_matrix.f90:1542](/home/ac_desktop/JOREK_code/jorek/models/model502/mod_elt_matrix.f90:1542)、[mod_elt_matrix.f90:1179](/home/ac_desktop/JOREK_code/jorek/models/model502/mod_elt_matrix.f90:1179)、[mod_elt_matrix.f90:1196](/home/ac_desktop/JOREK_code/jorek/models/model502/mod_elt_matrix.f90:1196)

## 3. 写/读时序及 restart 可见分支

事件数组把投影放在 JOREK timestep 前；同一时刻 `with_event_1D_at` 按数组顺序执行。主循环先用 `[particle_start_time,target_time]` 推算子步，算完反馈后设 `sim%time=target_time` 再运行事件。正常连续步按这条源码调用链，投影先落入全局 `aux_node_list`，随后 JOREK 矩阵装配读取该列表；`do_jorek_timestep` 先增 `index_now`，求解并更新 `t_now`，在 `index_now mod nout=0` 时写 `jorek{n:06d}`，退出时另写 `jorek_restart`。projection 写出条件是 `index_now mod nout_projection=0`，文件按 `index_now` 命名，`t_now=sim%time`（SI 秒）。这支持正常步 `projections{n}` 对应已采样的 `[T_{n−1},T_n]` 窗口、按调用链供下一流体求解读取；对 restart 后的具体一步不能只凭时间戳作同样断言。[marker_spi_source.f90:175](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:175)、[mod_event.f90:200](/home/ac_desktop/JOREK_code/jorek/particles/mod_event.f90:200)、[marker_spi_source.f90:342](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:342)、[marker_spi_source.f90:749](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:749)、[mod_jorek_timestepping.f90:327](/home/ac_desktop/JOREK_code/jorek/core/mod_jorek_timestepping.f90:327)、[mod_jorek_timestepping.f90:424](/home/ac_desktop/JOREK_code/jorek/core/mod_jorek_timestepping.f90:424)、[mod_jorek_timestepping.f90:532](/home/ac_desktop/JOREK_code/jorek/core/mod_jorek_timestepping.f90:532)、[mod_project_particles.f90:263](/home/ac_desktop/JOREK_code/jorek/particles/diagnostics/mod_project_particles.f90:263)、[mod_project_particles.f90:471](/home/ac_desktop/JOREK_code/jorek/particles/diagnostics/mod_project_particles.f90:471)

粒子 `part{n:05d}.h5` 则在循环末、上述 `call with(sim, events, at=sim%time)` **之后**按 `index_now mod nout=0` 写；`/time` 直接写 `sim%time`（SI 秒）。故 `part` 的编号取自已步进的 `index_now`，其粒子时间可落在前一流体时刻，不能用编号同名配对时间代替 HDF5 `/time`。[marker_spi_source.f90:749](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:749)、[marker_spi_source.f90:797](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:797)、[mod_particle_io.f90:119](/home/ac_desktop/JOREK_code/jorek/particles/mod_particle_io.f90:119)

restart 可见路径：流体 reader 从 `jorek_restart.h5` 读取并据 `t_start` 设置 `sim%time`，随后 `restart_particles=.t.` 的分支读 `part_restart.h5`，其 reader 又用粒子文件的 `/time` **覆盖 `sim%time`**。两组生产 run 的 namelist 均启用此分支。新建投影对象时其节点 `values/deltas` 被清零、五通道 RHS 也归零，然后将节点列表赋给全局 `aux_node_list`。主粒子循环**进入 `do while` 之前**就在当前（已被粒子 restart 回拨的）`sim%time` 调用事件；投影事件排在 JOREK 事件之前，先投影零 RHS、再由 JOREK 步的 `construct_matrix` 经 `aux_node_list` 读取节点。因此按这条源码路径，重启首个流体步使用新生成的零投影；它不是恢复的 marker 历史投影。随后才发生正时长的粒子采样，并生成下一编号投影。这个结论结合下文日志和文件数值验证；这里的“读取”是静态调用链结论，并非另有矩阵输入运行时 dump。[mod_fields_linear.f90:393](/home/ac_desktop/JOREK_code/jorek/particles/mod_fields_linear.f90:393)、[marker_spi_source.f90:100](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:100)、[mod_particle_io.f90:457](/home/ac_desktop/JOREK_code/jorek/particles/mod_particle_io.f90:457)、[mod_project_particles.f90:127](/home/ac_desktop/JOREK_code/jorek/particles/diagnostics/mod_project_particles.f90:127)、[marker_spi_source.f90:145](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:145)、[marker_spi_source.f90:175](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:175)、[marker_spi_source.f90:311](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:311)、[mod_event.f90:200](/home/ac_desktop/JOREK_code/jorek/particles/mod_event.f90:200)、[mod_jorek_timestepping.f90:327](/home/ac_desktop/JOREK_code/jorek/core/mod_jorek_timestepping.f90:327)、[construct_matrix_mod.f90:522](/home/ac_desktop/JOREK_code/jorek/matrix/construct_matrix_mod.f90:522)

### 重启输入与步进参数（补查）

重启**不是**在 `marker_spi_source.f90` 中硬编码一张事件时刻表：`jorek_restart.h5` 的 `index_now/t_now` 被导入为 `index_start/t_start`，粒子 `part_restart.h5` 的 `/time` 再设置粒子侧起点。以 injInner 当前保存的 [`jorek_namelist_spitest`](../data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner/jorek_namelist_spitest) 第 10–26、528–530 行为例，`restart=.t.`、`restart_particles=.t.`、`restart_spi=.t.`，`tstep_n=(0.2,0.1,0.2,0.5,1,2,5)`、`nstep_n=(1,400,300,800,300,300,1500)`，`time_evol_scheme="Gears"`，`nout=nout_projection=1`，粒子名义子步 `tstep_particles=5e−9 s`。v400 保存的 namelist 第 17–18 行使用另一张步长表 `tstep_n=(0.3,0.3,0.5,1,2)`、`nstep_n=(1,800,500,300,1500)`；不能把两 run 的参数混用。[mod_import_restart.f90:1542](/home/ac_desktop/JOREK_code/jorek/communication/mod_import_restart.f90:1542)、[mod_particle_io.f90:457](/home/ac_desktop/JOREK_code/jorek/particles/mod_particle_io.f90:457)

源码中的 `get_tstep_n(i)` 按 `nstep_n` 累计次数选择对应 `tstep_n`，**每次进程启动** `jorek_stepper%istep=1`，所以重启首个 JOREK 步按当前进程输入表的第一项取步长，而不是以 `index_start` 直接索引全历史步长表；`index_now` 则从导入的 `index_start` 加 1。Gears 在 `update_time_evol_params` 中设置 `theta=1,zeta=0.5`，并非“零步长”的演化方法。粒子侧先将 JOREK 归一化步长乘 `t_norm` 换秒，再调名义子步使其整除该步；循环按 `target_time−particle_start_time` 算 `n_steps`，采样结束后才设置 `sim%time=target_time` 并运行下一组事件。[mod_jorek_timestepping.f90:74](/home/ac_desktop/JOREK_code/jorek/core/mod_jorek_timestepping.f90:74)、[mod_jorek_timestepping.f90:309](/home/ac_desktop/JOREK_code/jorek/core/mod_jorek_timestepping.f90:309)、[mod_jorek_timestepping.f90:327](/home/ac_desktop/JOREK_code/jorek/core/mod_jorek_timestepping.f90:327)、[mod_jorek_timestepping.f90:574](/home/ac_desktop/JOREK_code/jorek/core/mod_jorek_timestepping.f90:574)、[update_time_evol_params.f90:8](/home/ac_desktop/JOREK_code/jorek/models/update_time_evol_params.f90:8)、[marker_spi_source.f90:79](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:79)、[marker_spi_source.f90:350](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:350)

injInner 的现存日志第 1776 行实录 restart 索引/时间 `9290/11551.8`，第 2055 行实录重启首步 `dt_jorek=0.2`（`8.17155e−8 s`），第 2105、2110–2111、2771、2775 行实录上述方法、步长表、粒子重启和子步输入。[logfile_spitest](../data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner/logfile_spitest)

补查每个边界的 `jorek{n−1}.h5` 和 `jorek{n}.h5`：两组 run 共 15 对，HDF5 的 `index_now`、`t_now`、`tstep` 均可直接读取；各对的 `tstep` 相同，v400 按边界顺序为 `5,0.5,1,0.5,2,1,0.5,0.5,0.3`，injInner 为 `5,1,1,1,1,0.2`（JOREK 单位）。比较所有 `jorek{n−1}` 的 40 个单值数据集，**仅** `index_now/t_now/tstep` 随边界变化；其余 37 个（包括 `jorek_model=502`、`n_order=3`、`n_tor=1`、`central_density=0.8`、`central_mass=1`、`t_norm=4.085775225951833e−7`、`eta=1.443e−10`、`visco=0.0024`、`visco_par=0.0005`）在两 run 各自内部不变，所列共同参数在两 run 间也一致。这是现存 namelist 对固定设置可信度的独立校验，也让历史各段实际的**当前步长**可由文件直接重建。[mod_export_restart.f90:706](/home/ac_desktop/JOREK_code/jorek/communication/mod_export_restart.f90:706)、[mod_import_restart.f90:1542](/home/ac_desktop/JOREK_code/jorek/communication/mod_import_restart.f90:1542)

**精确边界**：HDF5 的 `tstep` 是当前步长，不是完整 `tstep_n/nstep_n` 计划表；40 个单值字段也不覆盖 namelist 的全部输入。因此用户提供的“这些 phaseA 运行除步长表外其余配置未变”与所查文件相容、且固定的已保存参数得到实证支持；但若要逐项断言每段历史的完整计划表或未保存参数，仍需该段的原始输入/日志，不能把这个更窄的限制误写成“重启输入和步进方法都无法确认”。

## 4. 生产 run 原始数据实测

原始 run 目录为 [`data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400`](../data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400) 和 [`data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner`](../data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner)。两者的 [`jorek_namelist_spitest:528`](../data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400/jorek_namelist_spitest) 均为 `use_marker=.t.`。文件模式分别为 `<run>/macroscopic_vars.dat`、`<run>/jorek{n:06d}.h5`、`<run>/projections{n:06d}.h5`、`<run>/part{n:05d}.h5`。`macroscopic_vars.dat` 的 `@times` 由 [`parse_macroscopic_times`](../src/data/output_contract.py) 解析；`jorek` HDF5 的 `t_now` 是 JOREK 时间单位，projection HDF5 的 `t_now` 和 part HDF5 的 `/time` 是 SI 秒。主代理从 run 的 `macroscopic_vars.dat` 头读取 `sqrt_mu0_rho0=4.085775226e-7` 秒/归一化单位；换算时**只对 projection/part 除以此值**。两 run 的 `macroscopic_vars.dat` SHA-256（本地复核）分别为 v400 `dc80c40a01af6d50c8487fb99d74fd870b1aa1abd2e67998572d991337085e3c`、injInner `b9545154067037b8472a3ae5cf6b729e75713a61c2cc76316589674aa5c99aca`。源码分别在 [mod_export_restart.f90:710](/home/ac_desktop/JOREK_code/jorek/communication/mod_export_restart.f90:710)、[mod_project_particles.f90:594](/home/ac_desktop/JOREK_code/jorek/particles/diagnostics/mod_project_particles.f90:594)、[mod_particle_io.f90:122](/home/ac_desktop/JOREK_code/jorek/particles/mod_particle_io.f90:122) 写出三个时间字段。

| run（`<run>` basename） | `@times` 数量 | 零步长事件 `n` | `projections{n−1}.t_now/t_norm − jorek{n−1}.t_now`（JOREK 单位，按事件顺序） |
|---|---:|---|---|
| `phaseA_dataset_1d22_ang15_massloss_neon_v400` | 12470 | 2702, 4111, 5871, 6871, 8371, 9421, 10421, 11421, 11711 | −5, −0.5, −1, −0.5, −2, −1, −0.5, −0.5, −0.3 |
| `phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner` | 11105 | 2704, 5681, 6441, 7221, 8611, 9291 | −5, −1, −1, −1, −1, −0.2 |

全部 15 个事件满足 `T[n]=T[n−1]`；上列偏差等于 `−(T[n−1]−T[n−2])`，二者的最大绝对误差为 `1.364e−7` JOREK 单位。同一事件的 `projections{n}.t_now/t_norm−jorek{n}.t_now` 的最大绝对误差为 `1.362e−7`。这是零步长边界的**直接文件时间证据**，说明前一编号的投影文件时间仍滞后一个来源步；不能用重复的 `T[n]` 强行把 `projections{n−1}` 时间对齐到 `jorek{n−1}`。`data/restart_events.json` 只是先前登记注释，证据优先级低于这些 `@times` 和 HDF5 时间字段。

对两 run 全部 15 个事件直接读取 `values` 的完整四维数组：每个 `projections{n−1}` 均为**全零**，不是仅 dof0 为零；13 个已进入有源阶段的 `projections{n}` 均非零，两个初始无源事件（v400 `2702`、injInner `2704`）的 `projections{n}` 也为零。这个模式与上述“启动投影先执行、粒子后采样”的源码路径一致。零值本身不说明投影未被求解器消费；源码恰恰显示这个零投影先进入重启首步的矩阵装配。由此，`T[n]=T[n−1]` 是流体输出重复时间，并**不等于** `projections{n}` 的粒子采样窗口长度为零。

可逐行核对的实例是 injInner 最后一次 restart 的 [`logfile_spitest`](../data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner/logfile_spitest)：第 1775–1780 行记录 `restart time: 9290 11551.800000` 和流体 restart 读入；第 2034 行先写 `projections009290.h5`，第 6968 行记录完成 `After step 009291`；第 10940–10942 行才记录粒子从 `4.7197241100107399e−3 s` 开始、`particle_step_time=8.1715504519214166e−8 s`、`n_steps=16`；第 10957 行 `GATHER TIME=8.1715504519036660e−8 s`，第 10974 行写出 `projections009291.h5`。同目录 HDF5 的 `projections009290` 全 `values` 最大绝对值为 `0`，`projections009291` 则为 `2.3280016967558367e6`；两者 `t_now` 分别是 `0.00471972411001074` 和 `0.004719805825515259` 秒。故 9291 这次的 `projections{n}` 是**正时长采样所得**，不是所谓“退化 S_n”。

全量扫描 `part[0-9]*.h5`：v400 为 `977/977` 个、injInner 为 `840/840` 个文件，其 `/time/t_norm` 均匹配对应 `T[n−1]`，且没有缺少的前一 `@times`；最大绝对误差分别为 `1.36217e−7`、`1.45860e−7` JOREK 单位。injInner 的具体抽查 `n=2710,5680,5690,6440,6450,9290,9300` 亦满足此式。例：`part02710.h5` 为 8407，对应 `T[2710]=8412`；`part05680.h5` 为 10068，对应 `T[5680]=10069`；`part09300.h5` 为 11552.6，对应 `T[9300]=11552.7`。这些是文件时间配对证据；不包含对粒子内容或采样窗口物理长度的验证。

另对 restart 边界的两个 JOREK HDF5 读取 `xtime_spi_ablation[-1,0]`：injInner `5680→5681` 从 `4.24136397e20` 降到 `6.2920e17` atoms，`6440→6441` 从 `1.8045e20` 降到 `5.6677e17`；v400 `4110→4111` 从 `1.9644e19` 降到 `2.0501e17`。本地核对的其余**非零预算**事件也呈急降；初始事件 v400 `2702`、injInner `2704` 两侧均为零，不能以预算下降证明这两处的分段。预算突变与 restart 分段相符，是与时间重复帧独立的原始 HDF5 现象；它仍不标识实际被求解器消费的投影对象。JOREK 写出该历史数组的源码见 [mod_export_restart.f90:851](/home/ac_desktop/JOREK_code/jorek/communication/mod_export_restart.f90:851)。

**剩余证据边界：**上述源码调用链、一次有编号的运行日志和全部 15 个边界的投影数值，足以否定“`projections{n}` 是零长度采样窗口”，也支持“重启首步装配使用新建的零投影”。但目前没有在每个矩阵装配调用处额外 dump 实际 `aux_nodes`，也没有逐段运行日志复核全部 15 次的求解器内部状态；若要证明二进制执行在每次事件中逐项消费了哪个节点值，需要添加这种运行时插桩。不要把这项更窄的缺口混同于已经可由源码与现有日志确定的事件顺序。

可复核方法（从本仓库根目录运行；分别切换两个 `RUN` 值）：

```bash
RUN=data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner
python3 - "$RUN" <<'PY'
from pathlib import Path
import re
import sys
import h5py
import numpy as np
from src.data.output_contract import parse_macroscopic_times

run = Path(sys.argv[1])
T = parse_macroscopic_times(run / "macroscopic_vars.dat")
t_norm = 4.085775226e-7  # 从该 run 的 macroscopic_vars.dat 头复核

def scalar(prefix, n, key, width=6):
    with h5py.File(run / f"{prefix}{n:0{width}d}.h5") as f:
        return float(np.ravel(f[key][()])[0])

events = [n for n in sorted(T) if n-1 in T and T[n] == T[n-1]]
print("n_times", len(T), "zero_dt", events)
for n in events:
    lag = scalar("projections", n-1, "t_now") / t_norm - scalar("jorek", n-1, "t_now")
    aligned = scalar("projections", n, "t_now") / t_norm - scalar("jorek", n, "t_now")
    with h5py.File(run / f"jorek{n-1:06d}.h5") as before, \
         h5py.File(run / f"jorek{n:06d}.h5") as after:
        budget_pair = (float(before["xtime_spi_ablation"][-1, 0]),
                       float(after["xtime_spi_ablation"][-1, 0]))
        step_pair = (float(np.ravel(before["tstep"][()])[0]),
                     float(np.ravel(after["tstep"][()])[0]))
    print(n, "lag", lag, "expected", -(T[n-1]-T[n-2]),
          "same_index", aligned, "budget_before_after", budget_pair,
          "tstep_before_after", step_pair)
part_indices = sorted(int(m.group(1)) for p in run.glob("part[0-9]*.h5")
                      if (m := re.fullmatch(r"part(\d{5})\.h5", p.name)))
missing = [n for n in part_indices if n-1 not in T]
errors = [abs(scalar("part", n, "time", width=5)/t_norm - T[n-1])
          for n in part_indices if n-1 in T]
print("part_count", len(part_indices), "missing_previous_T", missing,
      "max_abs_error", max(errors, default=None))
if run.name.endswith("_injInner"):
    with h5py.File(run / "projections002704.h5") as f:
        print("projection_keys", list(f), "shape", f["values"].shape,
              "root_attrs", list(f.attrs), "values_attrs", list(f["values"].attrs))
    for n in (2710, 5680, 5690, 6440, 6450, 9290, 9300):
        print("part", n, scalar("part", n, "time", width=5)/t_norm,
              "T_prev", T[n-1], "T_same", T[n])
for n in events:
    maxima = []
    for k in (n-1, n):
        with h5py.File(run / f"projections{k:06d}.h5") as f:
            maxima.append(float(np.max(np.abs(f["values"][()]))))
    print("projection_values_maxabs", n, maxima)
PY
```

对 v400 run 把 `RUN` 改为 `data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400`，脚本会跳过 injInner 专属样本。上述统计来自本地原始文件；复核环境需要可访问仓库符号链接的目标。

## 5. 对 ADR-0020 的证据边界

- **决定 1 的核心方向有源码支持**：marker run 的投影通道 1–2 是被 model502 消费的电子能量 RHS 项；通道 3–5 是被消费的闭合量/系数输入。建议将“五通道真实场源”的统称细分为这两种作用，避免把后三项误当成独立的加性 RHS 源。
- **决定 4 的文件对象应说清楚**：记录的是 FEM 投影解的完整 `values`；当前 `[nodes,5]` 接口仅取 degree 0、环向索引 0。该降维选择可用于当前模型接口，但不能宣称它与完整 JOREK 节点自由度对象逐项相同。
- **决定 5 的登记策略有数据支持，但物理解释需要更正**：15 个零步长流体事件、前一编号投影的时间错位和全零节点场均可从原始文件重现；非初始事件还显示累计烧蚀量重置。按源码与 injInner 的重启日志，首个重复时间的流体步装配新生成的零投影，之后的 `projections{n}` 来自正时长粒子采样，不应称作退化窗口。可选择跳过重复流体帧用于训练，但这属于数据集策略，不能写成“该投影在运行时未被消费”或“`S_n` 的物理窗口为零”。

## 5. 持久状态实现时再次核查（2026-10-07）

`mod_initialise_particles.f90:421-450,529-563` 确认：只使用 `i_elm<=0` 的空槽；
池满终止；每次成功分配令 `i_life+=1`、中性化全部 `P_imp`、将 `V_prev` 初始化为出生处
流体速度。该例程没有设置 `t_birth`，所以不能从此字段零值推断实际出生时刻。
`mod_particle_types.f90:663,722,864` 确认 marker 的 `V_prev` 通过 `v_2d_arr`
写读，`mod_particle_io.f90` 将该数组保存为 `v`。

`mod_fieldline_euler.f90:56-78` 的推进是当前速度模乘当前/旧速度**单位方向**组合，
不是向量速度的普通 AB2；矩和辐射在旧位置、旧 P_imp 计算，沉积后才推进位置
（`marker_spi_source.f90:535-601,614-663`）。
`mod_fields_linear.f90:76-85,164-176` 不截断线性时间比例；余留时间可以引入
至多一个粒子子步的已有场历史外推。实现、状态分类、独立公式测试和真实 4110
读取记录见 [持久状态实现报告](marker-persistent-state-implementation-2026-10-07.md)。
