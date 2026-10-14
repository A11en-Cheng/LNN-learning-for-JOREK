# Marker 初始化、持久状态与续跑（2026-10-07）

依据：现行 ADR-0018/0019/0020，重点 0020 决定 3/5/9；`roadmap_v2.md` 未改动。
本次交付是可核查的小规模参考组件、原生检查点读取器、持久 marker 接口与事件驱动器。
未训练；未验证自生成源与流体的完整闭环；未扩展 FieldPropagator 的五通道到九方程映射。

## 源码复核

只读源码根：`/home/ac_desktop/JOREK_code/jorek`。本次重新读取行号和调用链：

| 事实 | 源码位置 |
|---|---|
| 流体 reader 先设置时间；粒子 reader 再用 `/time` 覆盖事件时间 | `particles/mod_fields_linear.f90:393-417`；`particles/mod_particle_io.f90:457` |
| 导入的流体 `tstep` 先调整固定粒子子步；不能从它恢复完整流体步长表 | `particles/examples/marker_spi_source.f90:82-86` |
| restart 读粒子之后新建投影、清 RHS；投影事件排在流体事件前 | `marker_spi_source.f90:100-175`；`particles/diagnostics/mod_project_particles.f90:127` |
| 启动前置事件、余留时间初始化和跨窗口累积 | `marker_spi_source.f90:311-355`：`real_particle_time=sim_time-step_rest_time`；`n_steps=INT((target-real_particle_time)/particle_dt)` |
| 一次在整个粒子窗口前分配新粒子；所有活粒子继续推进 | `marker_spi_source.f90:426-442,474-486` |
| 旧位置、旧电荷概率先计算矩和辐射；显式邻电荷态转移，截断后归一化；再沉积、再推位置 | `marker_spi_source.f90:496-601,614-663` |
| 旧反馈通道 4/5 参与平均电荷、电子密度和可选差分扩散通量 | `marker_spi_source.f90:496-529` |
| 背景数密度 `(rho-rhoimp)*n_norm` 下限 `1e16`；Ne 杂质数密度 `rhoimp*n_norm*central_mass/20` | `particles/mod_fields.f90:335-418`；梯度也带质量比 |
| 速度 pusher 使用当前速度模乘 `1.5*当前单位方向-0.5*旧单位方向`；不是普通向量 AB2 | `particles/pushers/mod_fieldline_euler.f90:56-78`；phi 增量除以旧 R |
| 新投影 deltas = 新 values − 旧 values，完整 degree/harmonic 保留 | `particles/diagnostics/mod_uncoupled_projection.f90:258-294` |
| 只复用 `i_elm<=0` 槽；池满失败；新粒子 `P_imp=(1,0,...)`、`V_prev=出生处流体速度`、`i_life+=1` | `particles/mod_initialise_particles.f90:421-450,529-563` |
| `t_birth` 是 real4 字段，但此初始化例程没有给它赋出生时间；不能把零值解释成真实出生时刻 | `particles/mod_particle_types.f90:53-79`；`mod_initialise_particles.f90:529-563` |
| 流体事件顺序：安装反馈 → 求解 → 用更新前场更新 shard → 提交新场/时间 | `core/mod_jorek_timestepping.f90:327-328,392-426` |
| 字段线性插值没有 clamp 时间比例；粒子余留可造成窗口起点之前至多一个粒子子步的外推 | `particles/mod_fields_linear.f90:76-85,164-176` |

核对 SHA-256：

- `marker_spi_source.f90`: `4f6e832b107569d7a8a0155ce5995d04d59cc50771d256cdf99c1e692318991c`
- `mod_initialise_particles.f90`: `269ce1cea5d5029c561f8ba93aacf4987f9c24163708f6df1a085cd6ba61f22d`
- `mod_fieldline_euler.f90`: `95b0635c67e9499c563547fefa74a247d21dc6feec7fd06a6141dfe3d72467a9`

## 状态和入口

| 类别 | 必须保存 | 可重建/限制 |
|---|---|---|
| `MarkerCloud` 物理槽位 | x、st、i_elm、weight、完整 P_imp、V_prev、i_life、t_birth、group | 身份为 group/slot/lifetime；不从均值电荷猜测 P_imp；不丢失失活槽的身份 |
| `ProjectionFeedback` | 全 values、deltas、时间、布局 | deltas 若已有相邻完整投影可差分无损重建；单帧投影不能恢复 deltas；dof-0 截取不是无损压缩 |
| `MarkerClock` | SI sim_time、step_rest_time、固定 particle_dt、场端点时间、窗口计数、启动事件标志 | 原生 part `/time` 只有事件时间，不保存余留；不能冒充真实粒子时间 |
| `ReferenceRollout` | 两个完整场端点、完整原生 `node%deltas`、时间积分方案、shard、流体时间/编号、步长计划及游标、前步长 | 场端点差不能无损重建原生增量；当前 HDF5 tstep 不能替代计划；程序重启从提供的 segment 计划开头执行 |
| 学习残差 | h_residual、模型权重/缓冲区、模型运行模式 | `AugmentedState.h_marker` 仅保留为兼容名称；不是 marker 物理状态 |

- `src/data/marker_checkpoint.py::initialize_native_checkpoint` 一次读取明确指定的流体、marker、时间表和 namelist，校验内部时间、事件编号、Alfvén 时间归一化、完整快照字段及 P_imp 归一化。文件名不作为同步依据，不搜索后续帧。
- `continuous` 初始化另需保存的 clock、完整前反馈/deltas、前场；缺失时明确失败。初态缺 marker 只在调用者明确确认不存在活粒子时允许空云。
- `jorek_restart` 恢复原生粒子槽、投影置零、余留归零、登记待执行启动事件。粒子子步由导入 HDF5 当前 tstep 调整；流体 segment 计划来自明确提供的 namelist，两者分别记录。
- `reference_rollout_from_checkpoint` 接入完整参考事件状态；连续初始化还需明确的剩余计划，不猜测历史计划。`reference_fluid_event` 执行启动/流体/shard 顺序；`reference_particle_then_fluid` 只使用已持有的场端点，调用粒子组件后再做下一流体事件。
- `advance_window` 的场、ADAS 系数、几何、沉积和投影由明确回调提供，可独立核对公式。纯状态转移不写输入，不读取 HDF5。
- `save_model_rollout/load_model_rollout` 保存物理状态、反馈、余留时间、场历史、计划游标、学习记忆和模型 state_dict；校验 mesh/ADAS/配置指纹，恢复模式；不会附加 JOREK restart。只用于可信本地 inference checkpoint，没有训练优化器语义。

## 接入现有降阶模块及近似边界

`SourceModule(marker_state=...)` / `MarkerChain` 现在接受并返回持久状态。缺状态不再静默生成空云。
每窗口只增加本窗口新生中性粒子；旧粒子继续 ADAS 更新与推进。关闭烧蚀或 shard mask 不删除旧云。
前投影参与电子密度和可选差分扩散；新反馈按新旧投影构造 deltas。有界学习修正后的实际输出
也写回反馈状态，保证下次粒子使用的是交给场模块的反馈。

marker 分支使用前一个流体/shard 事件已经给出的 shard 位置及 `spi_abl*species`，窗口内保持 shard
不变，避免粒子子步重复更新 shard。独立 analytic_shard 分支保留原来的碎片预测器。
`advance_augmented_source` 是两个**已经可用**场端点之间的源组件便利接口；它不负责下一流体求解。
完整事件顺序使用 ReferenceRollout，不能把一个名为 u_next 的参数解释为可读未来真值。

`reduce_checkpoint_marker` 是显式适配：保留槽位/P_imp/v/身份，将 st 在双线性网格重定位，并取反馈 dof-0。
它**没有**把百万粒子摘要称作充分状态。参考路径限制最多 10,000 个活 marker、每窗口最多 10,000 个子步；
超出直接失败，待标定的降阶压缩另行接入。真实 4110 的 11,168 个活 marker 仅做检查点读取验证，未送入训练回放。

已登记近似：固定 golden-angle 包数/出生位置（域外候选不占活槽）、可增长槽池、二维极向速度、
常数 D_perp、位移 cap、零速度时的平稳延拓、PLT-only 辐射（非完整 PRB/CX）、双线性场、
无滤波 lumped 投影、numpy detached 参考物理。这些都不是逐值等价的 JOREK 全自由度复刻，
也没有宣称输运/辐射已标定。参考默认 pusher 对上游零速度 0/0 明确失败，降阶适配器才使用登记延拓。

## 原生 restart 窗口修正

`native_run.py` 的 registered 事件现在从前编号零投影的内部时间推导正时长粒子窗口，
验证该投影全 values 为零；仍跳过零流体跨界配对和滞后投影的监督配对。
这不等于运行时跳过零投影：reference driver 先用零投影执行首个流体事件。
`restart_degenerate_source_windows` 保留兼容报告字段，但正确 registered 事件不再填入。
`list` 仅保留历史诊断的流体差分解释，不能用作物理粒子窗口。

## 真实 phaseA 读取

run：`data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400`。
明确输入 `jorek004110.h5`、`part04110.h5`、`macroscopic_vars.dat`、`jorek_namelist_spitest`。

- 流体内部时间：`8602.000000000284` JOREK；粒子事件时间：`8601.500000000251` JOREK。
- `t_norm=4.085775225951833e-7 s`；按导入 HDF5 tstep 调整的 `particle_dt=5.107219032439791e-9 s`。
- 1,000,000 槽、11,168 活 marker、P_imp `(1000000,11)`，完整反馈布局 `(5,4,1,4125)`。
- 独立原生 HDF5 比对 `/time` 与 `v→V_prev`；读取后保持粒子电荷态，程序 restart 状态为零反馈且待启动。
- 当前留存 namelist 的步长表是显式提供的 segment 输入，不声称它复原了 4110 之前的历史步长表。

## 验证记录

全部使用 `.torch-cuda-env/bin/python`，无正式训练。小型测试使用已知场/系数驱动，属于组件验证。
独立基准手算邻电荷态转移、旧电荷矩/辐射和方向式柱坐标推进，未只比较同一实现两遍。
序列化续跑在相同平台/dtype 下要求逐位一致；手算公式采用 float64 `atol=1e-15`；
内部时间匹配采用约 `1e-7` JOREK 时间及最小 `1e-14 s` 的舍入容差。

| 命令（均加 `-q`） | 结果 |
|---|---|
| `-m pytest tests/test_source_module.py tests/test_step_dataset.py tests/test_marker_projection.py`（改动前） | 75 passed, 1 skipped |
| `-m pytest tests/test_marker_state.py`（早期参考组件） | 7 passed；补齐事件驱动/连续读取后 9 passed |
| `-m pytest tests/test_source_module.py tests/test_marker_state.py`（接入过程） | 初次 2 failed / 30 passed：出生槽位计数与 SI 时间消去舍入；已修复。一次后续出生计数断言失败：域外候选不分配活槽，已改成物理计数断言 |
| `-m pytest tests/test_step_dataset.py tests/test_marker_state.py tests/test_source_module.py` | 80 passed, 1 skipped |
| `-m pytest tests/test_source_module.py tests/test_marker_state.py tests/test_step_dataset.py tests/test_output_contract.py tests/test_check_export_v3.py` | 103 passed, 1 skipped（后续新增测试另见下行最终记录） |
| `-m pytest -q`（首轮） | 287 passed, 1 skipped, 2 warnings / 188.33 s |
| `-m pytest tests/test_source_module.py tests/test_marker_state.py -q`（新增反馈/配置校验） | 38 passed；补余留时间下出生权重公式后 39 passed / 5.79 s |
| `-m pytest -q`（一次无效复测） | 6 failed, 283 passed, 1 skipped：运行中重构共享步进函数导致进程加载新旧模块混用的 ImportError；新进程相关测试全通过，冻结代码后再次全量验证，见最终记录 |
| `tools/check_export_v3.py --run-dir data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner --zero-dt-policy registered --quiet --no-budget` | exit 0，整个真实 run 登记 restart 通过 |
| `-m compileall -q src/models/marker_state.py src/models/reduced_marker.py src/models/source_module.py src/data/marker_checkpoint.py src/data/native_run.py`；`git diff --check` | exit 0 |

| `-m pytest -q`（冻结代码后的最终全量） | **290 passed, 1 skipped, 2 warnings / 188.46 s**；两条 warning 为既有 torch.jit.script 弃用提示 |

补充：出生原子权重使用实际粒子 `n_steps*particle_dt`，不是单纯事件跨度；余留时间跨窗口造成二者不等时也有独立权重断言。浮点 SI 时间消去仅在距整数小于 `1e-10` 个子步时吸附，其他情况保留 Fortran INT 行为。loader 的正窗口起止是事件时间，不能在缺少余留历史时声称已精确恢复 gather time。

## 流体历史增量传递补齐（2026-10-07）

核查发现先前原生适配器虽读取 `deltas`，参考事件驱动未将它传给流体求解器；此前场端点保存不能代替这项历史。本次补齐 `NativeMarkerCheckpoint.fluid_deltas → ReferenceRollout.fluid_deltas → reference_fluid_event → solve_fluid`，保存的是完整原生布局、原生场归一化下的未缩放节点增量。模型检查点保存并恢复该数组、时间积分方案、前步长及计划游标；缺失历史、布局不匹配或非有限值明确失败。即使存在两个场端点也不自动相减：浮点加法舍入以及约束/冻结谐波的处理使该操作不能保证无损。

重新核查的本地 JOREK 依据（路径相对只读源码根）：

- `communication/mod_export_restart.f90:493,616–617` 导出节点增量；`communication/mod_import_restart.f90:1306–1311,1410–1426` 读回并映射节点增量。本次使用 model502；不套用 model180 的特殊清零分支。
- `models/model502/mod_elt_matrix.f90:225,301–326` 分别插值场 `nodes%deltas` 与 marker 辅助投影 deltas，然后在装配局部量中乘 `tstep/tstep_prev`，不覆盖持久数组。Gears 的有效 `zeta=tstep/(tstep+tstep_prev)`；`theta=1`。`1563,2168` 分别显示标量密度历史 RHS 项和 `1+zeta` 时间矩阵项。
- `models/update_time_evol_params.f90` 定义 Gears、implicit Euler、Crank-Nicholson 系数；`models/preset_parameters.f90:12` 默认是 Crank-Nicholson。方案来源（显式 namelist 或此默认）写入检查点 provenance。
- `core/mod_jorek_timestepping.f90:310–335,392–429`：程序首步令 `tstep_prev=tstep`，求解后先在旧场更新 shard，再 `update_values`、`update_deltas`、推进流体时间并保存当前步长为下步前步长。程序 restart 保留导入的流体节点增量，只有 marker 投影遵循已登记的清零事件。
- `communication/update_values.f90:89–94` 将求解增量加入节点值并保存节点增量；`communication/update_deltas.f90:17–39` 将更新后的节点增量打包回求解向量，而非用场端点相减计算增量。

求解回调现在显式接收 `fluid_deltas` 和 `FluidTimeCoefficients(theta,zeta,delta_scale)`，另行接收完整 marker feedback。两类 deltas 互不替换。回调返回 `FluidSolveResult(values, fluid_deltas)`，由具体后端负责原生节点映射、约束和冻结谐波；驱动在旧场 shard 更新后同时提交这两个数组，并更新前步长。连续边界使用 HDF5 当前 `tstep*t_norm` 作为前步长，不能从可能重复的输出时间推导；JOREK 程序首步则使用当前计划步长。模型续跑不触发程序首步覆盖规则。

新增 `tests/test_fluid_history.py` 使用独立不等步长 BDF2 标量公式核对 Gears：令 `r=h/k`，单位强迫下 `Δu_new=[h(1+r)+r²Δu_old]/(1+2r)`。覆盖相同当前场/不同历史、两次变步长、逐位一致的模型序列化续跑、舍入后端点相同但节点增量非零、缺失/坏历史失败、重复输出时间及三种方案系数。它是时间历史与事件驱动的组件验证，未实现或验证九场耦合，也未开展训练。

本次验证命令和结果（使用仓库既有环境）：

- `.torch-cuda-env/bin/python -m pytest tests/test_marker_state.py tests/test_source_module.py -q`：改动前 39 passed / 6.88 s。
- `.torch-cuda-env/bin/python -m pytest tests/test_fluid_history.py tests/test_marker_state.py tests/test_source_module.py -q`：50 passed / 7.18 s，包含真实 phaseA 检查点读取，未跳过。
- `.torch-cuda-env/bin/python -m pytest tests/test_fluid_history.py tests/test_marker_state.py tests/test_source_module.py tests/test_step_dataset.py tests/test_output_contract.py -q`：111 passed, 1 skipped / 11.42 s；补加真实 phaseA 原始 HDF5 `deltas` 与参考状态逐值比对后复测：111 passed, 1 skipped / 10.30 s。真实检查点读取测试通过；跳过项为既有其他组件测试。
- `.torch-cuda-env/bin/python -m compileall -q src/models/marker_state.py src/data/marker_checkpoint.py tests/test_fluid_history.py`；`git diff --check`：exit 0。
