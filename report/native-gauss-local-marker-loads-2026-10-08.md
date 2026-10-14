# 完整原生自由度求值与 marker 局部两项载荷参考链（2026-10-08）

依据现行 ADR-0018/0019/0020。使用 research 核查只读本地源码，使用 tdd 按公共求值/局部载荷边界实现；未修订领域定义或 ADR，因此未使用 domain-modeling。未训练、未修改 roadmap 或 JOREK，保留其他用户修改。

验收结论限定为：**完整自由度求值与局部两项载荷参考链**。这里的完整自由度是生产 model502 cubic 的全部4个 degree 和全部给定场 harmonic，不是节点 dof0。不是完整 FEM 装配，不是九场耦合通过，不是 `source_proj` 已完成替换。

## 源码确认事实

详细一手映射、源码版本、hash 和生产日志见 [源码证据报告](native-fem-gauss-primary-evidence-2026-10-08.md)。只读源码 HEAD 为 `54c675bcc39c33b082c0c35046e28122dc53b3aa`。

| 确认事实 | 一手位置（相对 JOREK 根） | 本轮处理 |
|---|---|---|
| 实际 cubic 为 mixed Bézier/cubic，B0/B1 含3因子，mixed含9；右端导数为−3 | `elements/mod_basisfunctions.f90:40–51,76–86,141–164` | 保留全部 degree，不按标准 Hermite 导数系数直接解释 |
| vertex 依次(0,0)/(1,0)/(1,1)/(0,1)，degree依次A_sA_t/B_sA_t/A_sB_t/B_sB_t | 同上二维展开 | 本地数组沿原生顺序求值 |
| 完整 values、投影 deltas 和坐标都乘 signed element size | `models/model502/mod_elt_matrix.f90:263–308` | 不从四角重算 size，不 abs，不重复乘 |
| size 在网格完成时包含弦长、n_order 和方向符号 | `grids/grid_utils/finish_grid.f90:115–135` | 直接读取检查点原生 size；允许原生零 size，奇异几何明确失败 |
| native h5py values 为[field,degree,harmonic,node]，x 为[R/Z,degree,coord-harmonic,node]，size 为[degree,vertex,element] | `communication/mod_export_restart.f90:404–407,491–493,608–617,687–692` | 本地边界保持相同轴序，仅显式选4节点 |
| 普通路径几何仅坐标 harmonic1；J是有符号det | `models/model502/mod_elt_matrix.f90:263–277,353` | 只支持轴对称几何；不静默截断非轴对称坐标 |
| Gauss4为[0,1]规则；生产日志确认未启用GAUSS_ORDER8 | `elements/gauss.f90:22–27`；phaseA v400 `logfile_spitest:2243` | 显式 `GaussRule(4)`，不额外做[-1,1]映射 |
| HZ为1/cos/sin，phi=2π*plane/(n_plane*n_period) | `elements/basis_at_gaussian.f90:76–91` | 全部给定场 harmonic 和phi导数参与求值；phi不除R |
| 测试函数v=H*size*HZ，局部RHS只乘w_s*w_t并逐plane求和 | `models/model502/mod_elt_matrix.f90:1427,1854` | 不加2π、delta_phi或1/n_plane |
| 直接源负号并乘一次dt_JU；投影历史负号但不额外乘dt/theta/zeta | 同文件 `1728,1788` | 接用已验证的 marker_feedback 消费层；其内部只缩放一次投影 deltas |
| rho/rhoimp指数修正，Te阈值max(T_min,Te_1)；历史项使用原始Te/rhoimp | 同文件 `419,436,492–500,1788`；`models/corr_neg_include.f90:78–98,309–329` | 原始与修正量同时返回；必须明确提供JU温度阈值 |

`Te_1` 可能被数值 profile 覆盖，不能套 preset。此 run `logfile_spitest:2653,2769` 确认 Te_1=1e-5、T_min=3.2085e-4（JU）；完整取值链见源码证据报告。

## 实现与明确输入输出契约

实现：[native_gauss.py](../src/models/native_gauss.py)、[native_element.py](../src/data/native_element.py)。沿用 [marker_feedback.py](../src/models/marker_feedback.py) 的公式消费接口。

| 公共入口/对象 | 输入 | 输出与规则 |
|---|---|---|
| `read_native_element(path, element_index=...)` | 明确指定一个原生文件；零起始 element_index | `NativeElement`：完整坐标/size、保留原生vertex顺序的零起始节点IDs、显式采样元数据及路径 provenance；只读几何/元数据，不搜索任何场/投影/未来文件 |
| `NativeElement.gather(array)` | 调用者已有的完整原生[field,4,harmonic,node]数组 | 本地[field,4,harmonic,4]拷贝；流体、投影 values 和各自 deltas 分别调用，不丢degree或harmonic |
| `NativeElementGeometry` | coordinates=[2,4,1,4]，size=[degree,vertex]，n_order=3 | 明确原生布局；不是四角坐标插值 |
| `evaluate_native_element(geometry, values, rule, toroidal)` | 完整本地原生数组，显式Gauss4与mode/plane/period | 全场 values、s/t/phi/R/Z一阶导数、R/Z几何、GaussMetric、signed J、带size/HZ的完整测试基和极向权重 |
| `FluidCorrectionParameters(t_min,te_1)` | 显式解析来源后的JU阈值 | rho/rhoimp用源码L1=1e-12,L2=1e-8；Te用L1=L2=max阈值/2，先完整求值再修正 |
| `marker_electron_energy_local_loads(...)` | 几何、完整本地9场、完整5通道 values、原始完整 projection deltas、units/parameters、rule/toroidal/correction、FluidTimeCoefficients及dt_si | `NativeMarkerLocalLoads`：raw场求值、raw/修正流体、GaussMarkerFeedback、消费结果、direct与projection_history两个局部向量及原生方程9局部RHS索引 |

`q` 展平顺序是 `(plane,s_index,t_index)`；一切高斯点数组沿同一q顺序。`test_basis=[q,vertex,degree,harmonic]`，`weights=[q]` 只含 `w_s*w_t`。局部向量为 `[4,4,n_harmonics]`；`rhs_indices` 是完整环向块、九场原生局部RHS中的**零起始**索引：`((vertex*4+degree)*9+8)*n_harmonics+harmonic`。不声称完成受约束自由度映射或全局稀疏装配。

纯数组链不读取文件、不改变输入。一次性因子分工：

1. 求值器：完整原生系数乘size和HZ；测试基也各含一次size/HZ。原始投影 deltas只做空间求值。
2. 现有消费层：`dt_si/t_norm`只进入直接源；`FluidTimeCoefficients.delta_scale`只给投影历史增量乘一次步长比。
3. 现有 `electron_energy_weak_terms`：测试基、R、signed J和极向Gauss权重各消费一次。
4. 局部驱动：只求和，绝不再乘dt、zeta、theta、size或环向积分因子。

提供给未来 FieldPropagator 的是原生物理空间数组与这两个方程9局部向量，不能把它们当latent forcing。现有 `source_proj` 的“未物理对齐”标识保留；本轮没有修改其行为。剩余方程闭合/LHS后端仍需明确接入。

缺少完整degree/harmonic/坐标/size、投影原始 deltas、修正阈值、单位/物理配置，或有非法布局、NaN、求值溢出、奇异J、非正R、未知规则时明确失败。没有用零值补状态。多harmonic原生文件若无显式mode计划也失败，不能从编号猜测mode。

## 独立测试证据

测试：[test_native_gauss.py](../tests/test_native_gauss.py)；独立参考驱动：[native_gauss_reference.f90](../tests/native_gauss_reference.f90)。

- 单一非dof0原生系数，以**展开多项式**独立检验3倍基函数及s/t/R/Z导数。
- 曲边几何选R=2+s+0.1s²t、Z=t+0.2st²，场f=R+2Z+RZ。由解析端点值/导数换成原生系数，全部4个degree和带符号、非单位size参与；物理梯度的独立oracle为f_R=1+Z、f_Z=2+R。
- 矩形R=2+s、Z=t上，解析积分 `∫R*Aleft=23/20`、`∫R*Aright=27/20`、`∫R*Bleft=3/5`、`∫R*Bright=13/20`；t向 `∫A=1/2,∫B=1/4`。逐16个局部测试函数核对两项载荷，不只检查总和。
- 多harmonic解析场与phi导数；8个环向采样面的cos²和为4，直接证明测试基HZ和plane求和因子只消费一次，未混入FFT归一化。
- 非零投影 deltas、4种步长比、负J、原始负流体与正修正量、输入不变、禁止任何运行时文件读取，以及缺失/非法输入失败。
- **真实非零 phaseA 检查点**：`jorek004111.h5`、当前 `projections004111.h5` 和过去的零 `projections004110.h5`。检查内部时间匹配；投影文件不保存 deltas，本例仅因已保存前投影是完整零场，才能从当前减去过去零场无损重建本例原始增量。没有默认补零、没有后续帧。选当前投影信号最大的一个元素，完整9场、5投影值/导数和原始增量，以及两个载荷，与Fortran独立驱动对照。
- Fortran oracle 在pytest临时目录以 `gfortran -cpp -ffree-line-length-none -O0` 编译**只读 JOREK 原始** `elements/mod_basisfunctions.f90` 和 `elements/gauss.f90`，配最小n_order=3参数模块与独立驱动。未复制Python基函数生成期望值，未修改JOREK，未编译或运行完整流体求解器。
- **注册 restart 4111**：使用初始化适配器读取 `jorek004110.h5 + part04110.h5`，验证其新建零投影与原始 `projections004110.h5` 全degree一致；待启动标志保持。另一元素的完整求值产生两项精确零载荷，原始非零流体 `node%deltas` 逐位不变。登记文件只取事件IDs，不采用其旧“退化粒子窗口”描述；零流体输出时间差不意味着粒子采样窗口为零。

解析场/手算载荷float64容差约 `rtol=2e-14, atol=2e-14`（不同断言按单位范围更严）；原始数组保持不变和零载荷用逐位比较。真实Fortran对照逐场/导数列用 `max(|reference_column|,1e-13)` 独立缩放，最大相对缩放误差要求 `<2e-11`，避免大能量通道掩盖小密度/几何误差；载荷 `rtol=2e-11, atol=1e-18`。容差涵盖Python contraction与Fortran显式求和的消去和舍入顺序差异，不能据此宣称逐位等价。

## 运行命令和结果

使用仓库已有 `.torch-cuda-env/bin/python`；Fortran测试仅临时编译组件，无训练。

| 命令 | 结果 |
|---|---|
| `.torch-cuda-env/bin/python -m pytest tests/test_marker_feedback.py tests/test_marker_state.py tests/test_fluid_history.py -q`（改动前） | 69 passed / 3.84s |
| `.torch-cuda-env/bin/python -m pytest tests/test_native_gauss.py -q`（tdd逐切片） | 求值入口缺失先1 failed→1 passed；Fourier切片1 failed/2 passed→3 passed；两项载荷入口1 failed/3 passed→4 passed；原生reader入口1 failed/4 passed→5 passed |
| 同上（真实Fortran、restart、修正量） | 8 passed / 1.78s；无skip |
| 同上（溢出失败切片） | 先1 failed/8 passed，补明确失败后9 passed / 1.67s |
| 同上（补步长/环向/非法输入覆盖） | 26 passed / 1.71s；真实读取与Fortran都执行，未skip |
| `.torch-cuda-env/bin/python -m pytest tests/test_native_gauss.py tests/test_marker_feedback.py tests/test_fluid_history.py tests/test_marker_state.py tests/test_source_module.py tests/test_field_propagator.py tests/test_step_dataset.py tests/test_output_contract.py -q` | 209 passed, 1 skipped / 17.23s；新增26项全部执行 |
| `.torch-cuda-env/bin/python -m compileall -q src/models/native_gauss.py src/data/native_element.py tests/test_native_gauss.py`；`git diff --check` | exit0 |

## 剩余近似和证据缺口

本次完整cubic求值和上述两项载荷**没有使用**dof0、双线性、lumped质量替代；已登记的旧降阶近似仍留在其他模块，不能借此报告消除。

- 仅实现生产cubic、Gauss4和普通轴对称几何；quintic、源码另有的Gauss8配置、非轴对称坐标/FFT路径明确未实现或未验收。
- 对流体只返回本链需要的一阶导数；完整九场装配需要的二阶导数、全部闭合、非对角矩阵、流体历史时间项、全局装配/约束/轴/边界处理及求解更新映射不在本轮交付内。
- 真实对照覆盖一个非零元素和一个注册零反馈元素，不是全网格扫描或多段生产轨迹验证；独立Fortran驱动仍是组件，并非实际生产二进制装配处的运行时dump。
- 本地HEAD与生产日志支持本次公式/参数核查，但未获得完整生产二进制哈希和全部编译选项。NumPy float64参考路径不提供自动微分、GPU驻留或训练成本结论。
- 当前 FieldPropagator 仍接node-dof0/latent；需要保存/生成可推进的完整原生自由度或明确登记降阶近似，再接局部载荷到方程装配后端。不能把本参考链输出直接塞进自由学习5→hidden矩阵，并宣称其物理替换完成。

结论仅为“完整自由度求值与局部两项载荷参考链”；完整 FEM 装配与九场耦合仍未通过验证。
