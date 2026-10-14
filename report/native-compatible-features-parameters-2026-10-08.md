# G/P：兼容值/梯度与分组参数的可微学习路径

日期：2026-10-08。遵循 ADR-0018/0019/0020/0021 的定向覆盖关系；不变更 roadmap、ADR、JOREK。验收仅为**兼容值/梯度与分组参数已进入可微学习路径**，不是预测精度、五通道物理耦合或 C0 闭环通过。本轮没有训练、CfC、完整求解器或新反馈消费者。

## 1. 源码事实与边界

只读 JOREK 版本：54c675bcc39c33b082c0c35046e28122dc53b3aa。完整基函数、signed size、顶点次序和几何事实沿用并重新对照[原生 Gauss 证据](native-fem-gauss-primary-evidence-2026-10-08.md)及[局部载荷报告](native-gauss-local-marker-loads-2026-10-08.md)：

- elements/mod_basisfunctions.f90:40–51、76–86、141–164：cubic 基函数及一阶导数；degree 为值、两个方向控制系数及混合控制系数。后面三项不是可随意当作物理导数的数组。
- finish_grid.f90:115–135；model502/mod_elt_matrix.f90:263–308：signed element size 作用于各 degree，顶点顺序为 (0,0)、(1,0)、(1,1)、(0,1)。几何也使用完整 cubic 系数。
- model502/mod_elt_matrix.f90:353 的 signed Jacobian 用于物理导数；gauss.f90:22–27 的四点规则，两方向合计 16 点。
- preset_parameters.f90:555：pellet_density 单位为 10^20 m^-3；:799–800 为 n_spi 数量配置。
- models/mod_plasma_functions.f90:46 与 model502/mod_elt_matrix.f90:1474、1947：实际电阻消费者含 eta_T、电流及其他闭合。本轮 eta·∇ψ 只是声明的学习观测，不能替代该 RHS/LHS。结合现有归一化约定，eta 的磁扩散系数单位为 m²/JOREK time；不再重复转换为 SI 电阻率。

支持 H=1、norder=3、n_plane=1、n_coord_tor=1 的轴对称几何，R>0、非奇异映射。不是任意 harmonic 的实现。JOREK 文件仅被读取，未修改。

## 2. 单一 Interface 与逐层 shape

实现：
- src/models/native_features.py：TorchNativeGauss、NativeFeaturePath、固定参数 schema 与适配校验。
- src/data/native_features.py：从当前检查点读取静态全轴几何，选择当前输入的参数；不搜索后续帧。
- src/models/native_field_step.py：现有场步的可选 feature_path 实验分支、原有 ReferenceRollout solver Adapter。
- tools/diagnose_native_features.py：可复现组件成本及梯度诊断。

| 层 | 输入/输出 | 含义 |
|---|---|---|
| 当前状态 | U、rawD 各 [9,4,1,N] | 完整原生系数及原始流体历史，分别保留 |
| 静态求值器 | x [2,4,1,N]、size [4,4,E]、vertex [4,E] | 文件顶点转为显式零基索引；完整几何绑定版本 |
| Torch 收缩 | values [E,16,9]、gradients [E,16,9,2] | 实际 U 定义同一份场值和 R/Z 一阶导数 |
| 分组缩放 | 同 shape | 先求物理量，再缩放；没有独立梯度预测 |
| 共享节点池化 | [N,27] | 9 值 + 18 梯度，空间学习观测 |
| 场景分支 | 2 → 4 | n_spi、pellet_density 的独立 tanh 编码 |
| 物理特征分支 | eta·∇ψ → [N,2] | 只进入声明的 magnetic_encoder |
| 原场编码 | [N,77] → [N,hidden] | 36 U + 36 rawD + 原有 5 个时间特征 |
| 输出 | 增量、下一 U、下一 rawD 各 [9,4,1,N] | 完整残差更新，保留现有约束映射 |

主调用：NativeFeaturePath(U, rawD, units=..., parameters=...) 返回 NativeFeatures，其中包含原始 Gauss 量、缩放值/梯度、节点观测、可选场景/物理特征及原 rawD 引用。restore_values / restore_gradients 可逆。不存在隐式未来标签或时间计划输入。原有 dt、prev_dt、格式、theta/zeta 接口继续由场步处理。

静态 B/G 通过已验证 NumPy 求值器对 16 个独立单位系数编译为 Torch buffer；这等价于固定线性算子的完整矩阵求值，不是在动态场上调用 NumPy。几何、单位、基函数/聚合版本共同绑定 digest。动态 U/rawD、参数及输出均走 Torch；真实测试在前向/反向期间禁止 Tensor.detach、Tensor.numpy 与 HDF5 打开。静态几何不支持求导。

## 3. 数学、单位与缩放

每个 Gauss 点执行完整收缩：u = Σ(vertex,degree) B u_native，梯度 = Σ G u_native。signed size 已包含在 B/G 中，只消费一次。若几何切向量为 (R_s,Z_s)、(R_t,Z_t)，J=R_s Z_t−R_t Z_s：

∂R u = (Z_t u_s−Z_s u_t)/J；∂Z u = (−R_t u_s+R_s u_t)/J。

使用 signed J 变换导数，不能用 abs(J) 替代。值保留各原生场单位，梯度为相应单位/m。原生系数的现有分 degree 缩放与物理值/梯度统计互相独立；物理梯度不除以系数 scale。

统计只访问 split=train 的窗口完整 support_records，以 (run,内部 step_index) 去重。验证/测试窗口被排除。训练统计包含该训练支撑的初始化、warm-up、目标记录，这是离线拟合而非预测时读取标签。统计持久化版本 native-gauss-train-statistics-v1，记录完整支撑、几何 digest、组开关及尺度。

令 W_eq = w_s w_t |J| R。对训练唯一记录和全部 Gauss 点：
- 值：训练加权均值 μ、标准差 σ；特征 (u−μ)/σ。
- 梯度：每场每方向的加权 RMS g；特征 ∂u/g，不平移梯度。
- 场景参数：每项跨唯一记录 RMS，不平移，保留恒定条件的非零输入。
- eta·∇ψ：各方向的加权 RMS。
- 精确零尺度采用单位尺度 1 的显式可逆约定；非有限/非正持久尺度拒绝。没有 log 或其他需要链式法则的物理量变换；可学习 tanh 位于后续编码，Torch 自动求导。

值的单位依照 NativeUnits/原生场契约；ψ 为 T·m²，∇ψ 为 T·m，eta·∇ψ 为 T·m³/JOREK time。后者不是磁通量、电阻热源或电阻 RHS；这里只以该量构造可解释的系数相关学习特征。

## 4. 聚合近似与完整状态支路

每个顶点的正权 hat 为 (1−s)(1−t)、s(1−t)、st、(1−s)t。节点观测：

A_i(f) = Σ(e,q incident i) W_eq h_eqi f_eq / Σ(e,q incident i) W_eq h_eqi。

共享顶点跨单元累加后统一归一化；没有覆盖的节点明确拒绝。这里的 bilinear hat **仅用于特征池化**，完整场与梯度此前已由 cubic B/G 求得。池化丢失单元/Gauss 分辨率，不是原生 FEM 试函数投影、弱式载荷、容量矩阵逆或唯一 DOF 求解。聚合特征绝不代替 U/rawD。

新增 feature_encoder 与 magnetic_encoder 加到原 77 维编码支路；场景通过独立 FiLM 调制。空间层为 hidden + tanh(spatial(hidden)−hidden)。最终仍为完整 U + 本步原生增量，原有共享 DOF/约束处理不变。decoder 归零时完整 U 恒等支路保留。节点 stencil 复用到全部 degree 是明确的学习近似，物理 BC 尚未验收。

## 5. 固定参数路由

参数 schema native-feature-parameters-v1 固定语义排序；调用者字典/行顺序不决定特征排列。取值复用已有 native_parameters 路由；来源、单位、状态、available_at、声明 consumer 均校验。provenance、文件路径、run ID 不进入数值编码。

| 参数 | 分组/单位 | 合格来源 | 本轮唯一消费位置 |
|---|---|---|---|
| n_spi | scenario / 1，正整数 | 已声明 namelist/HDF5、初始化可用 | 场景编码第一项及 FiLM |
| pellet_density | scenario / 10^20 m^-3，正数 | 已声明 namelist/HDF5、初始化可用 | 场景编码第二项及 FiLM |
| eta | physical_coefficient / m²/JOREK time，非负 | 已声明 namelist/HDF5、初始化可用 | eta·∇ψ 观测及独立 magnetic_encoder |

必要参数缺失、重复、来源不明、默认/未确认、单位/consumer 不匹配、未知名字、非有限及非法范围均失败。更大路由表中已知但未消费的项目保留 excluded_names，不把 61 维全集 concat。场景组、物理组须用显式 boolean 开关；禁用时返回 None 并登记 disabled_groups，不能默默补零。时间仍走已有时间接口，不混进参数编码。schema、训练统计、全部模型权重/几何 buffer 进入模型 checkpoint 配置和 state_dict。

真实所测片段这些参数基本恒定。梯度/响应测试只能证明路径有效，不能证明可辨识、跨参数泛化或物理正确性。

## 6. 场步、事件与恢复

NativeFieldStep(feature_path=...) 保持既有 solver 调用、完整原生增量输出和原始历史更新。ReferenceRollout 继续使用原有事件驱动，不新增调度框架。Adapter 校验 context 完整几何/单位与求值器绑定一致，并要求 physics.feature_parameter_rows；模型续跑恢复配置和权重，不引入 JOREK restart。ReferenceRollout NumPy/无梯度 Adapter 是推理接口；可微核心直接调用 NativeFieldStep，不宣称该 Adapter 是 BPTT。

原有 zero-only 反馈限制保留；feedback=None 明确失败，不能解释为零。真实检查点缺 projection deltas 的事实没有被本轮“补齐”。两个真实片段只验收特征与学习结构；没有把教师/缺反馈状态称为自生成源闭环。source_proj 的未物理对齐标识保留。

## 7. 测试证据、命令与成本

新增 tests/test_native_features.py，按公开求值/特征/场步/恢复接缝做 red→green；包括：
- 独立常数、矩形线性和曲边解析场，非单位 signed size；相同 dof0 改变高阶系数会改变值/梯度。
- 已有 NumPy 参考和编译未修改 JOREK 基函数的 Fortran 参考。Fortran 每列归一化误差 <2e-11；独立解析/池化 float64 容差约 2e-14～3e-14。
- gradcheck eps=1e-6、atol=2e-8、rtol=2e-6；全部九场四 degree 的 U 梯度覆盖，rawD 保持独立梯度。参数 eta 与密度有叶张量梯度。
- 独立加权统计 oracle：矩形线性场均值 107/30、σ=sqrt(971/30)、梯度 RMS [2,3]；逆变换、验证数据不污染训练统计。
- 正权和常数保持、非均匀双单元共享节点 oracle 52/3、单元内非恒定特征 oracle 5/14 与 11/16。
- 未来标签改写不影响当前输入；替换预测 U 后重新求值；未知/缺失/不合法参数、单位、奇异几何、缺 rawD 均拒绝。改变路径元数据不改变数值特征。
- 现有 ReferenceRollout 的变步长执行与中间模型保存/恢复一致；失败不修改输入；旧状态、窗口、场步、历史、源模块回归保留。

真实组件 fixture：v400 4111–4114、injInner 2704–2707，各自在显式小训练块内拟合自己的诊断统计，不借用上一轮 validation 统计。每例 E=3986、N=4125；完整 U/rawD [9,4,1,4125]，值 [3986,16,9]，梯度 [3986,16,9,2]，池化 [4125,27]。当前输入分别为 4112 / 2705，未来帧仅用于窗口标签和离线训练支撑统计，不进入模型步。登记 restart 由窗口生成器分段，普通变步长保留；没有额外 restart。

复现：

    .torch-cuda-env/bin/python -m pytest tests/test_native_features.py -q
    .torch-cuda-env/bin/python -m pytest tests/test_native_features.py tests/test_native_windows.py tests/test_native_field_step.py tests/test_native_state.py tests/test_fluid_history.py tests/test_marker_state.py tests/test_native_gauss.py tests/test_marker_feedback.py tests/test_source_module.py tests/test_field_propagator.py tests/test_step_dataset.py tests/test_output_contract.py -q
    .torch-cuda-env/bin/python tools/diagnose_native_features.py --output-dir /tmp/native-gp-diagnostics-2026-10-08 --repeats 3

测试及诊断最终结果见本节末。诊断不是训练：固定种子、一次预热、三次前向/反向，无 optimizer；CPU 单线程、float64、hidden=12、1 层 stencil、2 次迭代。参数总数 **1932**：原 encoder 936、decoder 468、stencil 36；场景编码 12、特征编码 336、FiLM 120、物理特征编码 24。全部参数梯度有限且非零，两个真实例 U/rawD 的全部 36 个场/degree 梯度覆盖。

机器可读结果：[诊断 JSON](native-compatible-features-diagnostics-2026-10-08.json)。时间包含完整特征求值与场步，初始化另列；内存为进程峰值，包含 Python/Torch、HDF5、窗口状态和静态矩阵，不是分支增量内存或 GPU 测量。此小片段 materialized 参考路径不代表大规模吞吐性能。

最终结果：新增测试 **14 passed / 5.51s**；上述 12 文件回归 **276 passed、1 skipped / 30.35s**。跳过的是既有 export_v3 真文件测试（尚无该导出格式），两个新增真实 phaseA 特征案例及 Fortran 对照均执行。单独检查跳过原因时曾误写测试节点名 test_real_export_v3_run，pytest 返回“not found”、exit 4；更正为 test_real_export_v3_run_if_present 后确认原因。compileall 检查本轮五个 Python 文件通过；git diff --check 通过。

| 组件 fixture | setup（含 I/O/静态矩阵/统计）s | forward 中位数 s | backward 中位数 s | 进程峰值 KiB |
|---|---:|---:|---:|---:|
| v400 4111–4114 | 2.146743 | 0.036080 | 0.016175 | 777156 |
| injInner 2704–2707 | 1.951834 | 0.033821 | 0.015006 | 785756 |

诊断训练支撑和版本化尺度另保存为[统计 JSON](native-compatible-features-statistics-2026-10-08.json)，仅供这两个独立组件 fixture 复现，不是全 run 生产训练统计。无需重新拟合即可恢复；模型 checkpoint 同时保存自身统计 buffer 与配置。

## 8. 剩余近似与证据缺口

只有 H=1；几何固定，未实现网格形变求导。Gauss→节点池化和节点 stencil 均为登记的空间学习近似。没有全局 FEM 装配、容量矩阵逆、完整九场求解器、闭合消费者、非零 marker 反馈、物理 BC 验收或 CfC。没有预测误差、长时间稳定性或参数泛化证据。真实测试覆盖上述两个片段，不推广到全部 run。完整状态保真、因果窗口、特征兼容性与梯度可达，不能替代物理推进有效性。
