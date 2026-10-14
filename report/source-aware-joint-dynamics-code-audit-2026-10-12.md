# Source-aware 联合 LTC/CfC：当前源码审计（2026-10-12）

## 范围、版本与结论

本报告是新方案的一手代码 sidecar，不重复论文研究。已读 AGENTS.md 与 ADR 索引；仅新增此文件，未修改代码、运行测试、训练、安装依赖或执行 GPU benchmark。
L 指 /home/ac_desktop/LNN-learning-for-JOREK，J 指 /home/ac_desktop/JOREK_code/jorek；下文链接均给出本次磁盘文件的精确行号。
只读 git -C J rev-parse --show-toplevel HEAD 核实 JOREK HEAD=54c675bcc39c33b082c0c35046e28122dc53b3aa；status 为修改 communication/eqdsk2jorek.f90、未跟踪 poincare。L HEAD=0611ce473c632b5d256e4d93c719c3017f01a079，工作树含大量既有改动；HEAD 不等于全部受审字节。
现行规则是 ADR-0018/0019/0020 加 0021/0022 的定向修订；旧 ADR-0001..0017 仅属（历史）。当前规则明确逐 base 耦合、不实现宏步跳步，并允许源/shard 停止梯度：[ADR-0021:5](/home/ac_desktop/LNN-learning-for-JOREK/docs/adr/0021-native-state-cfc-window-curriculum.md:5)。

**结论：当前代码不是数学上无法处理含源动力学。** 双源已进入 CfC，完整持久状态和因果事件闭环已经存在；缺少的是显式学习的宏观联合流，以及穿过源/shard 重算的训练梯度，不是缺少一切闭合机制。
本报告将现有逐 base 因果闭环称为 causalbase；这是行为简称，不是源码类名。新宏步方案属于研究提议，尚未实施，也不自动改变现行事件契约。

## 1. 哪些源特征实际进入 CfC

主路径由 CoupledNativeField 默认 joint_learning 注册，排斥 SmallNativeResponse 冻结容量混入；真实链构造器已经实例化 CfC：[native_coupling.py:825](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_coupling.py:825)、[native_coupling.py:843](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_coupling.py:843)、[native_real_chain_builder.py:179](/home/ac_desktop/LNN-learning-for-JOREK/tools/native_real_chain_builder.py:179)。
联合源编码的实际维数为 101=48+50+3，经 Linear(101,hidden) 与 tanh 输出节点 latent：[native_source_features.py:137](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_source_features.py:137)、[native_source_features.py:155](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_source_features.py:155)。

| 编码块 | 已验证的当前输入 | 一手位置 |
|---|---|---|
| 完整源系数 48 | 五通道 marker 投影、五通道 raw projection deltas、独立背景/杂质密度源，各四 degree，H=1 | [native_source_features.py:166](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_source_features.py:166) |
| Gauss 源观察 36 | 上述 12 通道的值 12 + R/Z 一阶梯度 24，经正体积权重汇聚 | [native_source_features.py:168](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_source_features.py:168) |
| 闭合观察 14 | 平均电荷/电荷平方、alpha_e、ne、Zeff、电子压力/容量、两种梯度、charge history、能量率、投影历史压力增量 | [native_source_features.py:15](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_source_features.py:15)、[native_source_features.py:170](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_source_features.py:170) |
| 事件时间 3 | 实际 gather、相对持久时间原点的事件时刻、登记 restart 标志 | [native_source_features.py:176](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_source_features.py:176) |

场支路另输入完整 U/rawD 的 36+36 系数、五个时间格式量、同场 Gauss 值/梯度及分组条件，并有边界参数/适用性特征支路；NativeFeaturePath 的分组参数 schema 仅 n_spi、pellet_density、eta，不能称全集物理参数已接通：[native_field_step.py:199](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_field_step.py:199)、[native_field_step.py:205](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_field_step.py:205)、[native_features.py:83](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_features.py:83)、[native_features.py:252](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_features.py:252)。

实际顺序为 source encoding → 加入场 latent → stencil → CfC → 九场 rate 解码。故源影响 CfC 的输入、记忆及场增量，并非只在 CfC 后补一个源项：[native_coupling.py:883](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_coupling.py:883)、[native_field_step.py:219](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_field_step.py:219)、[native_field_step.py:229](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_field_step.py:229)。
这些闭合量是学习特征；joint_learning 没有调用 frozen99 的物理 increment。压力、电阻率、交换算子及完整 JOREK 联立矩阵不能由特征接通推定已执行：[native_coupling.py:887](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_coupling.py:887)、[native_coupling.py:893](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_coupling.py:893)、[native_coupling.py:719](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_coupling.py:719)。

## 2. 场头没有直接接收什么

coupled_inputs 虽接收 shards，实际只提取 native_density_source、feedback values/deltas 和源时间；没有把完整 shards 或 cloud 传给场网络：[native_coupling.py:900](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_coupling.py:900)。
因此位置 R/Z/phi、注入坐标系速度、radius、ablation_atoms_s、species、injector_phi、ids 不直接进入场头；它们参与解析密度/birth/shard 更新后才间接影响源条件：[native_shard_dynamics.py:35](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_shard_dynamics.py:35)、[native_source_density.py:104](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_source_density.py:104)。
source_volume_m3 及其时刻/来源、nonlocal_diagnostic 表、old_equilibrium/ES 也保存在 runtime 的 shards 中，由物理消费者使用，不直接进入该 101 维编码：[native_source_density.py:79](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_source_density.py:79)、[native_shard_dynamics.py:54](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_shard_dynamics.py:54)。
cloud 的 x、st/i_elm、weight、P_imp 全 11 电荷态、V_prev、group/slot/lifetime、t_birth，以及 opt-in transport_history，均不直接进入场头：[marker_state.py:53](/home/ac_desktop/LNN-learning-for-JOREK/src/models/marker_state.py:53)、[marker_state.py:148](/home/ac_desktop/LNN-learning-for-JOREK/src/models/marker_state.py:148)。
粒子 h、step_rest_time、field_start/end、window_index、控制游标及 RNG 状态没有成为完整显式联合输入；当前源事件时间/gather 特征不能代替这些推进历史：[marker_state.py:133](/home/ac_desktop/LNN-learning-for-JOREK/src/models/marker_state.py:133)、[marker_state.py:563](/home/ac_desktop/LNN-learning-for-JOREK/src/models/marker_state.py:563)、[native_source_features.py:176](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_source_features.py:176)。

**解释性推断**：投影与 latent 压缩一般不能唯一恢复完整云；只保留 c3/c4/c5 的电荷矩不能逐电荷态计算 ADAS 转移和辐射。当前 runtime 保留 P_imp，所以整个解析闭环仍可闭合；这不是对 CfC 表达能力的不可能性证明。
（历史）SourceFieldHead 接收 shard 集合，但文件已标 v1 LEGACY，使用 kNN 和五通道消融代理；不能据此宣称当前原生联合头已读入完整云：[source_head.py:1](/home/ac_desktop/LNN-learning-for-JOREK/src/models/source_head.py:1)、[source_head.py:108](/home/ac_desktop/LNN-learning-for-JOREK/src/models/source_head.py:108)。

## 3. 状态、事件与训练梯度的边界

ReferenceRollout 保存当前/前场、raw fluid deltas、前 dt、field_memory、marker runtime、shards 与步长计划；field_memory 与物理历史分别保存：[marker_state.py:570](/home/ac_desktop/LNN-learning-for-JOREK/src/models/marker_state.py:570)。
普通事件执行可用场历史上的 birth/marker 窗口与投影，再求场候选，在 OLD 场更新 shard，最后提交 U/rawD/memory/时钟；诊断刷新读取提交后的预测场：[marker_state.py:800](/home/ac_desktop/LNN-learning-for-JOREK/src/models/marker_state.py:800)、[marker_state.py:764](/home/ac_desktop/LNN-learning-for-JOREK/src/models/marker_state.py:764)、[native_coupling.py:1035](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_coupling.py:1035)。
源时刻必须绑定当前场事件与实际 gather；源码主动拒绝 stale/future 源。登记 restart 清空投影/history/residual，先执行启动流体事件；正常续跑不调用 restart：[native_source_features.py:33](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_source_features.py:33)、[marker_state.py:166](/home/ac_desktop/LNN-learning-for-JOREK/src/models/marker_state.py:166)、[native_coupling.py:1076](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_coupling.py:1076)。

纯参考 solver 使用 no_grad 与 NumPy 输出；TorchRecomputedSourceEvents 则把 state.values/raw_deltas/memory 原计算图直接交给场模型，并另存 detached NumPy mirrors 给源/shard：[native_field_step.py:321](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_field_step.py:321)、[base_coupled.py:201](/home/ac_desktop/LNN-learning-for-JOREK/src/training/base_coupled.py:201)、[base_coupled.py:213](/home/ac_desktop/LNN-learning-for-JOREK/src/training/base_coupled.py:213)。
**已验证的梯度范围**：场/记忆 BPTT 和 Torch 闭合公式可微；field → NumPy birth/atomic/transport/projection/shard → 后续 source 的依赖梯度被切断。数值上反馈仍存在，优化时没有这一链式导数：[base_coupled.py:132](/home/ac_desktop/LNN-learning-for-JOREK/src/training/base_coupled.py:132)、[native_coupling.py:724](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_coupling.py:724)。
记录节点只决定监督 mask；循环每次消费一个实际 StepControl，未在 loss 节点自动重置场/云。TBPTT 显式 detach 保留状态值；checkpoint 绑定完整 session、配置、游标和 RNG：[base_coupled.py:281](/home/ac_desktop/LNN-learning-for-JOREK/src/training/base_coupled.py:281)、[base_coupled.py:323](/home/ac_desktop/LNN-learning-for-JOREK/src/training/base_coupled.py:323)、[base_coupled.py:26](/home/ac_desktop/LNN-learning-for-JOREK/src/training/base_coupled.py:26)。

**causalbase 能闭合执行**：当前事件代码具备上述完整链；现存 v400/injInner 各两事件机器产物为 success=true、future_truth_feedback=false：[v400 result.json:2](/home/ac_desktop/LNN-learning-for-JOREK/output/native-autonomous-smoke-2026-10-09/v400-run2/result.json:2)、[injInner result.json:2](/home/ac_desktop/LNN-learning-for-JOREK/output/native-autonomous-smoke-2026-10-09/injInner-run2/result.json:2)。
这些是已有运行记录，本次只读、未重跑；smoke 构造器将 decoder 权重设为 1e-8，且提交后才读取 labels，证明范围是数值链可执行，不是当前字节的再认证或长程精度：[native_real_chain_builder.py:188](/home/ac_desktop/LNN-learning-for-JOREK/tools/native_real_chain_builder.py:188)、[smoke_native_autonomous.py:59](/home/ac_desktop/LNN-learning-for-JOREK/tools/smoke_native_autonomous.py:59)。

## 4. CfC、外层 rate 与可复用 LTC

当前 CfC Adapter 锁定 ncps 1.0.1 default 模式与源码哈希，输入节点特征、显式 memory、dt_JU/time_scale；它本身没有 MHD/守恒保证：[native_temporal.py:18](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_temporal.py:18)、[native_temporal.py:37](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_temporal.py:37)。
已读本地库实现：default 分支对两个 tanh 候选用 sigmoid(time_a*ts+time_b) 混合，返回相同 output/new_hidden；不是原生系数的物理解算：[ncps/torch/cfc_cell.py:136](/home/ac_desktop/LNN-learning-for-JOREK/.torch-cuda-env/lib/python3.12/site-packages/ncps/torch/cfc_cell.py:136)。
外层设 h=dt_SI/t_norm、τ=time_scale_JU，执行 m_next=m+(1−exp(−h/τ))*(m_CfC−m)，ΔU=increment_scale*rate*(h/τ)，再共享 degree/约束映射得到 U_next：[native_temporal.py:70](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_temporal.py:70)、[native_field_step.py:233](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_field_step.py:233)、[native_field_step.py:58](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_field_step.py:58)。
该时间乘子只积分网络学习的 coefficient rate；源特征中的 energy/history 已有物理时间解释，不能当作额外直接强迫再重复乘 dt。零时间恒等由场 wrapper 早返回实现：[native_source_features.py:4](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_source_features.py:4)、[native_field_step.py:185](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_field_step.py:185)。
可复用 LTCCell 冻结局部 rate/target 后执行 m_next=m*exp(−aΔt)+b*(1−exp(−aΔt))，a=softplus+eps；允许 state_dependent 改变系数输入：[ltc_cell.py:108](/home/ac_desktop/LNN-learning-for-JOREK/src/models/ltc_cell.py:108)。

**未保证**：这种局部精确衰减不证明更新源、位置、参数后的联合非线性推进有严格半群、任意大步稳定、守恒或精度。已有 core/外层 adapter 可作同契约 LTC/CfC 比较接缝，当前 native 主路径尚无 LTC Adapter。

## 5. JOREK 一手时序、ADAS 与 birth clipping

J/particles/examples/marker_spi_source.f90:314 先调启动事件；:352 保留 residual，N=int((target−start)/h_particle)，:443 设置实际 gather=N*h_particle；birth 在子步之前，:486 从 start+(k−1)h 采样：[marker_spi_source.f90:352](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:352)、[marker_spi_source.f90:433](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:433)。
场候选求解成功后先 update_spi，再 update_values/update_deltas；所以 shard 使用 OLD 场，不可改成新候选场：[mod_jorek_timestepping.f90:393](/home/ac_desktop/JOREK_code/jorek/core/mod_jorek_timestepping.f90:393)、[mod_jorek_timestepping.f90:418](/home/ac_desktop/JOREK_code/jorek/core/mod_jorek_timestepping.f90:418)、[mod_jorek_timestepping.f90:424](/home/ac_desktop/JOREK_code/jorek/core/mod_jorek_timestepping.f90:424)。
shard 先旧速度推进位置、旧消融侵蚀半径并截零，在新位置采 OLD 场求新消融，再 rocket6 更新速度；marker 分支局部 Z_imp=0 placeholder，不能换成当前 marker 电荷：[pellet_module.f90:269](/home/ac_desktop/JOREK_code/jorek/models/pellet_module.f90:269)、[pellet_module.f90:307](/home/ac_desktop/JOREK_code/jorek/models/pellet_module.f90:307)、[pellet_module.f90:689](/home/ac_desktop/JOREK_code/jorek/models/pellet_module.f90:689)、[pellet_module.f90:887](/home/ac_desktop/JOREK_code/jorek/models/pellet_module.f90:887)。

一阶采样为 U(t)=U_now−rawD*(time_now−t)/(time_now−time_prev)，需 t_jorek>0、abs(跨度)>1e-10、非 static；否则保持 U_now，static 仍可有 rawD 时间导数。Python literal 接缝已存在：[mod_fields_linear.f90:70](/home/ac_desktop/JOREK_code/jorek/particles/mod_fields_linear.f90:70)、[native_source_time.py:42](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_source_time.py:42)。
本地 marker example 的 reader i=−1，成功读取后置 static=true；故不能不看实际配置就声称所有生产窗口都随时间插值。birth 初速度使用 sim%time，而非 residual 的 particle_start：[marker_spi_source.f90:76](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:76)、[mod_fields_linear.f90:405](/home/ac_desktop/JOREK_code/jorek/particles/mod_fields_linear.f90:405)、[mod_initialise_particles.f90:537](/home/ac_desktop/JOREK_code/jorek/particles/mod_initialise_particles.f90:537)。
ADAS 非 CE 分支用旧 P_imp 做逐相邻电荷态 SCD/ACD 显式转移，逐项 max(.,0) 后归一化；电离能使用新旧概率差，辐射和电荷矩使用旧概率，再在旧位置沉积、移动并重新定位：[marker_spi_source.f90:535](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:535)、[marker_spi_source.f90:555](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:555)、[marker_spi_source.f90:598](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:598)、[marker_spi_source.f90:628](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:628)。
JOREK ADF11 读入时把 cm⁻³/eV/Wcm³ 转为 m⁻³/K/Wm³；get_Lz=(PLT+PRB)*ne。Python 支持显式 PLT 或 PLT+PRB 模式，但具体等价性须绑定表/配置：[mod_openadas.f90:104](/home/ac_desktop/JOREK_code/jorek/tools/mod_openadas.f90:104)、[mod_radiation.f90:73](/home/ac_desktop/JOREK_code/jorek/particles/projection_functions/mod_radiation.f90:73)、[native_coupling.py:648](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_coupling.py:648)。
V_prev 存上一采样的总速度 V+V_ext；pusher 外推的是方向 1.5*v_hat−0.5*prev_hat，再乘当前速率和 dt，不是普通向量 AB2。丢掉 V_prev 会改下一位置：[mod_fieldline_euler.f90:67](/home/ac_desktop/JOREK_code/jorek/particles/pushers/mod_fieldline_euler.f90:67)、[marker_state.py:224](/home/ac_desktop/LNN-learning-for-JOREK/src/models/marker_state.py:224)。
birth 数量由 Σ旧 ablation*species/n_cpu 的 atoms/s 与 gather 决定；候选空间的 P 则是逐 shard 核、species/mass_ratio 分配后跨 injector 累加的 impurity MASS 源，两种 sum 不可混淆：[marker_spi_source.f90:410](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:410)、[mod_injection_source.f90:403](/home/ac_desktop/JOREK_code/jorek/models/mod_injection_source.f90:403)、[mod_initialise_particles.f90:512](/home/ac_desktop/JOREK_code/jorek/particles/mod_initialise_particles.f90:512)。
接受判据在总 P 求和之后才执行 ran(4)<real4(max(P*dv/P_norm,0))，源码没有显式 min(.,1)；均匀随机比较产生有效概率饱和。不可先逐 shard cap 后相加，也不可把密度源本身截成平台：[mod_initialise_particles.f90:530](/home/ac_desktop/JOREK_code/jorek/particles/mod_initialise_particles.f90:530)、[marker_spi_source.f90:809](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:809)。
Python 当前按同样总 mass 源与 real4 阈值接受，默认 MT19937 与可选表格位置实验均声明不复现生产 PCG32 拆流；表格实验亦非连续原生接受分布的精确复刻：[native_birth_acceptance.py:7](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_birth_acceptance.py:7)、[native_source_birth.py:30](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_source_birth.py:30)、[native_source_birth.py:124](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_source_birth.py:124)。

## 6. 共用 FEM、投影与 nearby 的可复用接缝

NativeSourceMesh.point/points_many 用相同 cubic basis、size、曲面 Jacobian 求场值/梯度；TorchNativeGauss 将静态同核基函数注册为 buffer，动态 contractions 保留梯度：[native_coupling.py:140](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_coupling.py:140)、[native_coupling.py:162](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_coupling.py:162)、[native_features.py:62](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_features.py:62)。
粒子旧位置沉积按 shared DOF 累加，除实际 gather 后求投影；raw projection delta 是新投影减持久旧投影，与 fluid rawD 分开：[native_coupling.py:400](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_coupling.py:400)、[native_coupling.py:409](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_coupling.py:409)、[marker_state.py:396](/home/ac_desktop/LNN-learning-for-JOREK/src/models/marker_state.py:396)、[mod_uncoupled_projection.f90:280](/home/ac_desktop/JOREK_code/jorek/particles/diagnostics/mod_uncoupled_projection.f90:280)。
marker 可用 inspected n0 滤波投影；独立 shard 密度强制用未滤波的独立表示。共用几何不代表两个 projector 相同：[native_projection_n0.py:11](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_projection_n0.py:11)、[native_source_density.py:143](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_source_density.py:143)。
JOREK row5/8 独立使用 total_imp_source；五通道 c1/c2 是 row9 能量项，c3/c4/c5 参与电荷、压力/电子容量与 Zeff 闭合，c5 不能替代密度源或变为固定 5→9 RHS：[mod_elt_matrix.f90:989](/home/ac_desktop/JOREK_code/jorek/models/model502/mod_elt_matrix.f90:989)、[mod_elt_matrix.f90:1542](/home/ac_desktop/JOREK_code/jorek/models/model502/mod_elt_matrix.f90:1542)、[mod_elt_matrix.f90:1717](/home/ac_desktop/JOREK_code/jorek/models/model502/mod_elt_matrix.f90:1717)、[mod_elt_matrix.f90:1728](/home/ac_desktop/JOREK_code/jorek/models/model502/mod_elt_matrix.f90:1728)。
source 特征的正体积 pooling 是有损观察汇聚；当前 node-P1 latent stencil 也不是完整 cubic 弱算子。共用原生场表示不意味着学习层已等同 FEM 装配：[native_source_features.py:120](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_source_features.py:120)、[native_field_step.py:121](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_field_step.py:121)。
NativeNearbyLocator 用保存的 native neighbours、8 次边限 Newton、跨边坐标转移及零邻居判失；全局 fallback 仍是 Python locator，已声明 production_qualified=false：[native_nearby_locator.py:62](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_nearby_locator.py:62)、[native_nearby_locator.py:124](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_nearby_locator.py:124)、[mod_find_rz_nearby.f90:117](/home/ac_desktop/JOREK_code/jorek/grids/grid_utils/mod_find_rz_nearby.f90:117)。
它是 opt-in source Adapter，可接现有 locate/deposit/project 回调；离散单元选择、birth 数量、拒绝采样、失活、槽重用仍是事件边界，不因局部插值平滑而自动全局可微：[native_nearby_locator.py:186](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_nearby_locator.py:186)、[native_coupling.py:606](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_coupling.py:606)、[marker_state.py:91](/home/ac_desktop/LNN-learning-for-JOREK/src/models/marker_state.py:91)。

另有显式 ctypes/C++ 原子表内插值与元素采样选项，可复用作执行内核；它们不提供 Torch backward，也不是新联合流。本次仅查看源码，未构建/装载内核：[native_compiled_atomic.py:32](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_compiled_atomic.py:32)、[native_element_functions.py:35](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_element_functions.py:35)。

## 7. 新联合 LTC/CfC 提议：设计推断，未实现

目标应定义为可恢复联合状态 X=(U,rawD,p_shard,m_cloud,P,D_P,h_U,h_p,h_m,clocks,caches,RNG)，而非单个 U 的含源拟合；依托现有完整状态与共享 S 入口，新增显式 joint-flow Adapter。

新增 shard 编码保持身份/局部位置、坐标系速度、radius、旧消融、species，并局部 FEM gather OLD 场值/梯度；nonlocal/volume/ES 作为带时刻的独立依赖。按物理消费者分组，不把所有全局参数拼进场头。

新增 cloud 编码首轮保持完整 P_imp、weight、位置、V_prev、活性/寿命与残余时钟；若压缩为云包或电荷态密度，必须给出可推进状态和重建/投影误差，不能只留下五通道快照。

共用空间核执行 field→shard/cloud gather 与 cloud→FEM scatter；云观察向节点做局部汇聚，场对粒子/云的作用由相同几何回采样。继续保留完整 U/rawD 恒等支路和独立密度源。

建议先让 LTC 承担云/局部闭包的耗散记忆、有界输运或速率残差，CfC 承担场及较慢联合记忆；原子转移保留 ADAS 确定性骨干。LTC 用于 latent 不等于用独立指数衰减替代电荷态耦合矩阵。

两种时间核心共用显式输入、归一化、状态布局、时间单位、约束和监督；亦应允许交换核心作归因比较。现有 CfCTimeCore、TimeConsistentResidualAdapter、LTCCell 是组件接缝，不预先宣称 LTC 或 CfC 胜出。

第一阶段联合化仍按 base 事件推进：可微连续核更新、共享双源编码、场候选、OLD 场 shard、统一提交；在已声明固定事件分支内保留场→源/shard→后续场的梯度，延续现有 source-event 时刻断言。

具体可迁移核包括 ADAS 表内插值/概率更新、固定单元的 cubic gather/scatter、解析密度核、shard 消融/rocket 和静态投影线性解；当前对应源核多为 NumPy，属于待实现，而非已经端到端。

**可行性推断**：静态投影矩阵 A 的解 C=A⁻¹b 可对 b 反传；若以后 A 随状态变，必须同时处理 dA。TorchMarkerLoads 已提供可微原生求值/weak_load，但不包含可微云 birth/transport/project 全链。

离散事件先保留精确前向规则，显式声明分支条件梯度/不可微范围；随机事件若使用梯度估计或连续近似，需独立协议与误差归因，不能用 straight-through 的便利冒充原生 ADAS/JOREK 导数。

第二阶段才研究宏区间联合流 Φ_joint(X,ΔT,controls)，区间内随状态更新源并处理 birth、跨单元、边界和诊断事件，同时输出可续跑的全部历史；它不同于把当前 S 冻结后把场 dt 放大。

macro 接口要分别返回区间源积分/平均、场增量、shard/cloud 新态和各对象时刻；当前 gather 已来自上一可用场窗，不能静默改成同一未来区间的源，也不能把 projection history 和区间积分互换。

宏步跳跃涉及现行 ADR-0021 的明确边界，需另立近似/时间接口契约；本报告仅提出设计，不修改 roadmap/ADR，不实施跳步。现有 causalbase 可作为相同初态、事件计划、源对象下的参考链。

后续应分别核验连续梯度覆盖、事件/预算/概率约束、逐场 degree 与兼容梯度、云投影/位置/shard 误差、同终点拆步误差和整链成本；本次无新测试或性能数字，不能预设宏步收益。

## 8. 已验证与未保证的最终边界

已验证：含源 CfC 输入、九场 rate 解码、完整持久 runtime、OLD 场顺序、独立双源、NumPy stop-gradient、共用 FEM 求值/沉积接缝；已有机器记录支持 causalbase 数值闭环能够执行。
未保证：完整物理 BC、生产 dirty 构建/宏/namelist 等价、逐 marker RNG 重放、长程精度/刚性稳定、宏步联合流、源/shard 端到端梯度或加速收益；生产 Adapter 本身标 not_certified：[native_source_production.py:23](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_source_production.py:23)、[native_field_boundary.py:60](/home/ac_desktop/LNN-learning-for-JOREK/src/models/native_field_boundary.py:60)。
本次关键文件 SHA256：L/src/models/native_source_features.py = 4a30ac34ea4181f1bb60a42716c2f2434b0f5a0e8d7be93cabf14b53a0d1ab19；L/src/models/native_field_step.py = 1fa0a2447ff8d73472f9da29fb4695e2cfe157cf8cdb517a6ffb4c69a4d00cb6。
L/src/models/native_temporal.py = 472218b0f9d8cdfc1e618c67b448e47123e5a2afce9bfe0faf6d3f45986cc49c；L/src/training/base_coupled.py = af188905232c662957621ce381c43a114a2b4a75ca17a5c3891b7bdd352771a2。
J/particles/examples/marker_spi_source.f90 = 4f6e832b107569d7a8a0155ce5995d04d59cc50771d256cdf99c1e692318991c；J/core/mod_jorek_timestepping.f90 = f45b62fff5165f2c0f728c7a24d14b919628ff455ed4d1497daafaecceecb5fe。
