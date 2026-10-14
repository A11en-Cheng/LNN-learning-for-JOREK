# 目标流体边界与完整云有界执行（agent B，2026-10-09）

本轮完成网络输出固定行的字段映射修正、Mach1/natural-open 适用关系与物理参数的实际网络路由，以及保留全部槽位的显式预算分块推进核。小规模两次完整 source→含源九场→OLD-field shard→提交及中途恢复通过。**真实 phaseA 完整事件仍为 0**：协调文件的预算接线尚未完成，且真实云单点定位已经暴露现有全域 Newton 搜索的失败。没有把接口拒绝算成完整云推进，没有给真实锚点授予资格。

禁止训练约束已遵守：无训练、optimizer 更新、超参搜索、未来标签、真值重置、完整 JOREK 矩阵移植或冻结容量回归。JOREK 与原始数据只读；保留工作区已有修改。本轮不写 `native_coupling.py`、`src/training/base_coupled.py`、`native_shard_dynamics.py` 或 A 的新模块。

## 一手证据及目标配置

只读源码根 `/home/ac_desktop/JOREK_code/jorek`，检查 HEAD 为 `54c675bcc39c33b082c0c35046e28122dc53b3aa`；工作树已有 `communication/eqdsk2jorek.f90` 修改及未跟踪 `poincare`。生产记录声明 `v2.25.12.0-1042-gb4165893f-dirty`。以下是**所检查源码版本的机制**和**实际记录的参数证据**，不声称当前 HEAD 等价于生产 dirty 构建。现行依据为 ADR-0018～0022；旧 ADR 不作 gate。

两个实际目标目录为 `data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400` 和其 `_injInner` 目录。两份 `jorek_namelist_spitest:133,400` 明确 `freeboundary=.f., bc_natural_open=.true.`。下面其余值来自实际生产 `logfile_spitest`，不是从源码默认值补出的生产配置；日志浮点数有输出舍入。

| 参数 | v400 日志行 | inner 日志行 | 记录值 |
|---|---:|---:|---|
| n_wall_blocks | 2351 | 2183 | 14 |
| use_simple_bnd_types | 2616 | 2448 | F |
| grid_to_wall | 2845 | 2677 | T |
| mach_one_bnd_integral | 2849 | 2681 | F |
| no_mach1_bc | 2853 | 2685 | F |
| RMP_on | 2864 | 2696 | F |
| density_reflection | 2810 | 2642 | 0 |
| imp_reflection | 2899 | 2731 | 0 |
| gamma_sheath_i / e | 2813–2814 | 2645–2646 | 3.5556 / 4.6667 |
| vpar_smoothing | 2817 | 2649 | F |
| min_sheath_angle | 2818 | 2650 | 1 degree |
| visco_par_heating | 2295 | 2127 | 0 |
| gamma | 2691 | 2523 | 1.6667 |
| T_min | 2769 | 2601 | 3.2085e-4 |

实际读取 v400 `jorek004110.h5` 与 inner `jorek005680.h5`：两者 `F0=-3.30874513`（此处显示舍入）、4125 节点，boundary 计数均为 `0:3927, 1:32, 2:94, 5:10, 9:6, 11:32, 15:10, 19:4, 20:4, 21:6`。这些是所读快照的证据，不认证生产初始化磁场或完整有效配置。

| 检查源码位置 | 对输出约束/适用关系的结论 |
|---|---|
| `models/model502/mod_model_settings.f90:32–45` | `var_rhon=8` 且 `var_rhoimp=8`，即本项目 rho_imp；`var_T=0` 不是第六场，Ti/Te 分别为6/9。必须先解析别名再映射九场。 |
| `models/model502/mod_boundary_conditions.f90:201–248,316–364` | 每条相邻非零 boundary 边固定 value 与对应切向 derivative；ψ/Phi/j/omega 固定，codes2/3/12/20/21 或 no_mach1 时固定全部九场，包含 rho_imp。法向/混合高阶系数不能一起清零。 |
| 同文件 `:156,356–358` | 实际 JOREK 加有限 1e12 惩罚，并非无限强度精确零增量。网络的理想零增量行是已有登记近似，不能宣称有限惩罚解逐位等价。 |
| 同文件 `:236–247,367–370` | 节点 Mach1 对1/3/4/5/9/11/15/19适用；3→2 的有向相邻关系跳过该 Mach1 项。target 的 integral=F 走节点分支。 |
| 同文件 `:400–448,490–528,554–625` | 固定旧磁场与几何后，对 Vpar、Ti、Te、Phi 的 value/切向导数建立耦合线性化惩罚；不是简单 vpar=±声速。 |
| `matrix/construct_matrix_mod.f90:85–185` | bc_natural_open=T 才装配开放积分。wall 网格 side2优先映射degree2、side3映射degree3（Fortran）；无已知side的corner-pair跳过，两side冲突也跳过。不能从节点 Mach1 mask 推出自然积分 mask。 |
| `models/model502/mod_boundary_matrix_open.f90:251–275,288–319,325–455` | rho/Ti/rho_imp/Te 的边界通量同时产生弱式 RHS 与矩阵耦合。方程7积分项另由 mach_one_bnd_integral 控制；target为F时该积分项乘0，节点 Mach1仍存在。 |

节点公式 `:517` 是

`Mach1BC = -Vpar0 + direction/Btot * factor * sqrt(gamma*(Ti0+Te0)) + factor/Btot * R^2 * Phi_b/psi_b`。

Ti0/Te0 在该分支先取 `max(T,T_min)`。即使 target 关闭 smoothing，也有磁场归一化、E×B 相关 Phi 项和耦合温度增量。把网络结果硬设为 `vpar=±sqrt(gamma*T)`、硬设零法向梯度、将弱式载荷直接相加到系数、或者对所有 cubic 系数正值裁剪，都不等价于这些源码行。本轮未实施这些转换。

## 已实施的边界切片

`NativeDirichletRows` 原实现遗漏 rho_imp，本轮先以完整场步的 code20 样例复现错误，再修正 `apply_dirichlet_all` 为九场全部 value/切向增量固定。现有边方向、共享 solver DOF alias 传播与自由高阶系数保持；旧测试中“rho_imp不在分支”的错误预期一并勘误。版本为 `model502-h1-dirichlet-rows-v2_rhon8`，旧错误配置不能静默作为精确续跑。此处纠正了 2026-10-09 base-step integration 报告中的同一错误；不覆盖其历史验收记录。

新增 `NativeBoundaryLearningRows` 是同一固定行 Module 上的学习 Adapter：分别构建节点 Mach1 行与自然积分 trace 行，执行上述两套源码适用规则及角点优先级。它提供56列输入：36个每场/degree固定mask、4个节点Mach1 mask、4个自然trace mask、9个边界物理参数和3个分支flag。参数为 `gamma, gamma_sheath_i, gamma_sheath_e, density_reflection, imp_reflection, min_sheath_angle, F0, T_min, visco_par_heating`；flag为 `no_mach1_bc, mach_one_bnd_integral, vpar_smoothing`。参数来源、wall配置、mask及schema进入模型检查点配置。

`NativeFieldStep` 将这些输入经专用编码器实际送入含源联合九场网络，输出仍通过共享DOF映射和固定行mask。policy为 `dirichlet_rows_open_learned_v1`；`TorchFieldStepResult.boundary_features` 返回所用输入，便于诊断。新编码器在主干里，不只是报告 metadata；手设权重的独立 tanh 数值例子验证了参数到实际 rho 增量的路由。没有把几何量冒充边界物理解；同场值/梯度、rawD、噪声观察、外层速率积分与Torch图沿既有路径保留。

支持范围必须分开声明：固定行的理想输出投影及其适用映射已实现；Mach1/natural 耦合响应由联合网络学习，**未显式执行这些弱式矩阵，未验证其物理预测精度**。一般 smoothing=True 时的系数路由/响应也未认证，target记录为F。新策略不叫“完整物理BC已通过”。与代码原有 `learned_unverified` 或固定行子集策略区分，旧模型未自动升级。没有恢复旧节点模型的简化 sonic 硬赋值路径。

## 有界完整云推进

新增不可变 `MarkerExecutionBudget(max_slots,max_live,chunk_size,max_particle_steps,max_seconds,max_rss_bytes)`，公开源推进 seam 为 `advance_window(..., execution_budget=...)`。没有显式预算时仍保留10000活粒子防线；只有显式预算允许更多活粒子，并要求正的槽位/活粒子/子步/时间/RSS上限。

先检查容量及复制/诊断数组的内存下限，再创建私有候选；birth仍在全部子步之前插入，空槽复用/寿命/权重/出生时间语义不变。每个粒子子步按原升序槽位分块，全部活粒子均处理；OLD位置沉积、电荷与辐射次序、gather和余留时间、前反馈、投影rawDP保持不变。没有裁剪云、压缩身份或把marker子步换成base事件。`sample`须只在指定ids上执行粒子局部计算，`deposit`须线性可加；当前原生源满足这一依赖结构。跨块浮点归约可能有舍入差异，等价验证使用显式数值容差，不能声称所有chunk配置都bitwise相同。

核心在callback前后检查耗时/RSS，失败抛出而不提交私有候选。RSS包含整个进程的既有高水位，内存下限不是整个源/场/shard事件的峰值预测。单个长callback不能靠这些检查中断，因此诊断工具再由父进程独立监控wall/RSS并终止超限子进程；不得以核心检查代替外部硬截止。

专项测试包含百万槽位、完整11电荷态、11168及52604活粒子的解析资源样例，保持全部identity/dead slots，分别实际处理对应数量的粒子步。它们不是把真实Ne云替换成解析云后的生产验收；真实云成本与生产事件另列如下。

## 真实云有限成本诊断及未进入的完整尝试

工具 `tools/diagnose_native_cloud_cost.py` 只读现有anchors和原始文件，拒绝覆盖证据。每个进程限20秒、1536MiB RSS；已知粒子点值的少量成本采样不是截断云rollout。完整尝试模式保留整个原始云，只在源的公开预算桥存在时调用；它明确是 frozen-field resource reference，不认证restart/birth/nonlocal，也不冒充含源场完整事件。

证据保存为 `output/native-boundary-cloud-execution-2026-10-09/cloud-cost.json` 和 `cloud-attempt-interface.json`。

| 实读云 | 槽位/活粒子 | cloud数组payload | point样本成本 | 真实locator结果 |
|---|---|---|---|---|
| v400 part04110 | 1000000 / 11168 | 172000000 B | 4点0.000413s | 首个活粒子即抛 invalid native point geometry |
| inner part05680 | 1000000 / 52604 | 172000000 B | 4点0.000418s | 同上 |

在保存的有效 `i_elm/st` 上全系数求值正常；现有 `NativeSourceMesh.locate` 对远处候选单元做无保护Newton，试探出非法Jacobian/R时从 `native_coupling.py:145` 经 `point:132` 全部退出。错误发生在几何候选搜索，不能据此宣布该真实marker已越界。必须允许候选失败并转查其它单元，且需要候选筛选/已有单元优先来避免活云×全网格的无限成本。本轮未在协调文件中自行改写定位。

cost进程实际耗时2.18/1.98s，观测峰值RSS约726/728MB。四点求值外推约1.15/5.50s只估计**一次场point求值**，不是sample+ADAS+deposit+locate+projection成本，更不是完整事件加速结果。

attempt-interface 两进程实际2.42/2.40s，峰值约883/884MB，已加载完整云并构造16419 unique DOF的稀疏参考投影。随后明确返回 `attempt_unavailable: NativeMarkerSource execution_budget bridge not wired`；**真实源窗口完成0、完整耦合事件完成0**。没有启动无限推进，没有对已知会失败的locator进行盲目大云重放，也没有将此接口检查记为完整云推进成功。

## 无训练测试与两事件/恢复证据

采用 codebase-design/tdd：把规则集中在现有 Module 的小 Interface 内，在用户已确认的源推进、完整场步、完整事件与checkpoint seam验证。新增预算/边界Interface先红；rho_imp错误先以源码别名对应的独立case复现，再改mask。未用私有物理实现生成oracle。

- 单步手算 `q=[.8,.2]`，OLD中性云电荷矩0，能量 `.4*10*elementary_charge`、辐射0.6J、number积分0.2；验证分块与未分块及余留时间、无输入污染。
- 百万槽位完整11态解析云实际推进11168/52604活粒子；默认guard仍拒绝，显式预算处理全部活粒子，dead slots/identity保留。容量、时间、RSS失败保留输入。
- 底边及corner19/20手算适用关系：两corner没有已知wall side时跳过自然积分，但19仍有节点Mach1；20固定全部九场；法向/混合自由degree仍有梯度。
- 完整场步的边界参数路由由手设权重得到独立 `.25*tanh(32/9)` oracle。它是依赖与算术验证，**不是物理响应预测的正确性gate**。
- 两次完整事件用 `jorek_raw_deltas_v1, field_static=True`，第一事件SCD对应q1=.04；手设场速率使rho1→1.1，第二事件消费预测rho，独立预期q1=.08608。中途保存，在新模型恢复后继续，当前/前场、rawD、场记忆、完整云所有数组、投影/rawDP和clock与连续运行bitwise相同。此例为解析源/小网格参考fixture，未授予phaseA资格；一般OLD-field shard及三阶段回滚由既有相关回归覆盖。

实际命令与结果：

```bash
.torch-cuda-env/bin/python -m pytest -q tests/test_native_boundary_learning.py tests/test_native_bounded_cloud.py tests/test_native_field_boundary.py tests/test_native_field_step.py tests/test_native_temporal.py tests/test_native_joint_source.py tests/test_native_coupling.py tests/test_native_source_time.py tests/test_native_source_time_integration.py tests/test_marker_state.py tests/test_native_state.py tests/test_fluid_history.py tests/test_native_execution_features.py tests/test_base_coupled_noise.py tests/test_native_context_guard.py --junitxml=/tmp/native-B-boundary-cloud-regression.xml
# 144 passed in 17.46s（最后的参数路由样例及T_min/visco参数扩展之前）
.torch-cuda-env/bin/python -m pytest -q tests/test_native_boundary_learning.py tests/test_native_bounded_cloud.py tests/test_native_field_boundary.py --junitxml=/tmp/native-B-boundary-cloud-focused.xml
# 最终10 passed in 1.28s；百万槽位样例使用完整11电荷态。
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .torch-cuda-env/bin/python tools/diagnose_native_cloud_cost.py --anchors output/native-closure-certification-2026-10-08/certified/anchors.json --output /tmp/native-B-cloud-cost.json --seconds 20 --memory-mib 1536
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .torch-cuda-env/bin/python tools/diagnose_native_cloud_cost.py --anchors output/native-closure-certification-2026-10-08/certified/anchors.json --output /tmp/native-B-cloud-attempt-interface.json --mode attempt --seconds 20 --memory-mib 1536
```

没有运行含optimizer的curriculum测试。上述机制没有精度、长期稳定或训练资格结论；Torch边界路径保留计算图，NumPy源/shard仍声明停止梯度。

## 主线程最小协调修改与剩余项

1. `native_coupling.py` 的 `NativeMarkerSource` 显式接收并保存 `execution_budget`，将其完整配置纳入source.configuration，再透传至 `advance_window`。默认None不改变既有guard；chunk配置关系到归约舍入及精确续跑，不静默变更。
2. 同文件的原生locator需要候选Newton失败容忍和保守候选筛选/旧单元优先。优化后先以小网格和真实保存st独立验证定位，再用本工具的外部wall/RSS上限作完整云有限尝试；不能只去掉guard。
3. 在真实模型构造时实例化 `NativeBoundaryLearningRows` 并绑定实际boundary/index/vertex及有来源的参数；context boundary policy采用其policy。不要把本报告的舍入日志值或fixture常数伪装为完整有效配置认证。
4. A的生产nonlocal/cache、birth/RNG、独立密度核由主线程按其接口接线；B未写A文件。完整真实事件还需合法startup/ordinary历史、明确控制、source/shard/场配置与dirty构建证据。普通快照不能补零历史；本轮不改数据资格。
5. 在这些条件完成前，“一个真实合法完整事件→第二事件吃预测场→恢复”仍未完成。本轮只有小fixture满足该事件/恢复结构；56列边界输入不等于弱式边界物理已学会。源预算桥、locator以及生产状态绑定的限制不能由随机未训练网络误差gate掩盖。

修改路径：`src/models/native_field_boundary.py`、`src/models/native_field_step.py`、`src/models/marker_state.py`、`tests/test_native_boundary_learning.py`、`tests/test_native_bounded_cloud.py`、`tests/test_native_field_boundary.py`、`tools/diagnose_native_cloud_cost.py`、本报告及独占output证据目录。`reduced_marker.py`保持原来的显式参考支持范围，没有借本次原生预算切片自动升级其近似模型。
