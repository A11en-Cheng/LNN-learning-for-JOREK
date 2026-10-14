# 模型与训练集闭环缺口审计（2026-10-08）

## 结论与范围

**不启动训练。当前小规模含源联合学习链已接通，但两个 phaseA 配置尚不能按现有入口获得真实闭环运行/训练资格。** 本轮仅审计代码、有限原始文件和已有接口测试；不修改模型、数据、JOREK、roadmap 或 ADR，不重新训练。使用 research；模型侧并行只读核查，最终证据汇总到本文。

依据为 [ADR 索引](../docs/adr/README.md)、[ADR-0022](../docs/adr/0022-source-conditioned-joint-native-advancement.md)及现行源码。联合网络替代含源九场推进映射；**不把完整九场矩阵、冻结容量、显式完整输运算子或全源端到端可微设为完成条件**。允许近似，但必须对应目标配置、明确接受范围并经测试，不能用缺项冒充近似。

结论分级：`已证实`=代码或本轮实读/运行；`缺实现`=入口明确不支持；`缺证据`=尚不能认证，不等于永久不可恢复；`近似待验收`=有替代实现但不等价生产配置。

## 1. 原始文件有限审计（本轮实读）

根目录：`data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400{,_injInner}`。

| run | 实读 jorek/projections 编号 | 结果 |
|---|---|---|
| v400 | 4109,4110,4111,4112,4142,4150,4161,4169,4180 | 9个完整场/投影对 |
| injInner | 2702,2703,2704,2705,2713,2714 | 6个完整场/投影对 |

15个场文件均有 `values[9,4,1,4125]`、原始 `deltas`；投影均为 `[5,4,1,4125]`，根键没有 `deltas`、gather 或 residual。两个网格 `len(unique(index))=16419`，并非小参考允许的128自由度。两个锚点4110/2703均有 shard 原始位置、速度、radius、abl等数组，`treat_axis=F`；这不意味着边界处理已经实现。

所选编号中 v400 的 `part04110/04150/04180.h5` 存在，`P_imp.shape=(1000000,11)`，含 `x,st,i_elm,i_life,t_birth,v,weight` 等；主审计读取元数据/shape，**百万是槽位数，不是百万存活粒子**。并行审计另按4110的`i_elm`读得11168活marker，超过参考10000上限；未推进该云。所选 injInner 编号没有同号 part 文件；不据此宣称整个 run 或其他位置不存在可用快照/注入前空云证明。

调用 [audit_native_segment](../src/data/native_state.py:76) 复核各run前4个编号：登记restart4111/2704为重复流体时刻，前一投影为合法零投影；普通时间配对通过。v4004112起所抽查步长为0.05 JU；injInner2705/2713为5 JU，2714已变为0.05 JU。不能把100倍步长差外推到整个run，更不能从重复流体时刻推定粒子零gather。

**资格结论**：实读范围有完整流体监督量，但普通源历史/时钟没有被这些文件直接完整保存；存在合法restart锚点线索，不等于两组闭环初态已经组装完毕。

## 2. 高优先级发现

### D1：步长数组被解析成首值（已证实，优先修复）

[parse_namelist](../src/data/namelist.py:66)按逗号分割后仅保留含等号片段，裸数组后续元素被丢弃。本轮最小复现：

```bash
.torch-cuda-env/bin/python - <<'PY'
from src.data.namelist import parse_namelist
print(parse_namelist('tstep_n = 0.3,0.3,0.5,1,2\nnstep_n = 1,800,500,300,1500'))
PY
# 实际：{'tstep_n': 0.3, 'nstep_n': 1.0}
```

[检查点初始化](../src/data/marker_checkpoint.py:116)直接消费这一结果，`reference_rollout_from_checkpoint`据其展开restart计划。因此真实namelist路径可能静默变成单步计划。已有显式JSON窗口计划不受同一解析路径影响；不能据此否定此前所有训练，但也不能凭小fixture恢复测试认证真实namelist续跑。

修复验收：完整数组、跨行/注释/多赋值的明确支持或拒绝；真实文件与独立预期对照；完整计划/游标恢复。即使解析修复，现存namelist也不自动等于每次restart所用时间表：用户确认非时间参数不变，并不能补出各段时间控制。需独立注册分段计划，端点只作核验、不充当未来输入。

### D2：数据资格系统仍是 field-only（缺实现）

[WindowConfig](../src/data/native_windows.py:30)拒绝非零replay及非`field_supervised`资格；[FieldRecord](../src/data/native_windows.py:16)/CurrentFieldInput没有完整shard、marker初始化、双源及统一源事件上下文。`read_segment`只在投影含deltas时读feedback，且没有完成源时钟/闭环恢复。

[audit_spec](../src/data/native_window_io.py:128)的`coupled_rollout_windows=0`是代码常量，并非全库扫描后测得的物理不可恢复窗口数；其gaps还保留“梯度/CfC未连接”的旧文案。现阶段应表述为**该构建器不能授予闭环资格**，不是“整个数据集不存在任何可用闭环窗口”。需将报告升级为由实际资格检查导出，区分文件缺失、可恢复未证、未实现、验证通过。

### M1：真实网格无法进入当前源/特征路径（已证实）

[NativeSourceMesh](../src/models/native_coupling.py:15)限制128自由度，构造逐元素全局稠密矩阵及全局basis并求一致投影；真实网格16419自由度超限。不能仅提高max_dofs：一个16419² float64矩阵约2.0 GiB，当前还逐元素保留矩阵，复杂度不可接受。formula-only消费者亦依赖该mesh，故影响不仅是粒子回放，还涉及真实网格含源特征构造。

修复：将局部cubic求值/聚合与全局源投影求解分开；源投影用可扩展共享索引稀疏装配/求解或经过明确验证的降阶方案。小网格与现有稠密oracle对照，真实网格只做资源受控的构造/投影验证。**不需要稀疏电子容量或完整九场求解器。**

### M2：生产配置与小源模型存在可定位差异（近似待验收）

两组实际 [v400 namelist](../data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400/jorek_namelist_spitest:487) / [injInner namelist](../data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner/jorek_namelist_spitest:487)均声明：`nonlocal_abl=.t.`、`spi_abl_model=3`、`n_spi=846`、`pellet_density=4.30d8`；531–536行投影滤波参数非零。

[SmallShardModel](../src/models/native_coupling.py:436)也是model3，但无nonlocal、采用归一化Gaussian密度核和每窗中心出生；[NativeNeonAtomicRates](../src/models/native_coupling.py:167)为PLT-only；投影为unfiltered。JOREK [投影矩阵](../../JOREK_code/jorek/particles/diagnostics/mod_uncoupled_projection.f90:592)明确有普通/高阶/平行滤波项。

**更正此前泛化风险**：两文件507行是`spi_abl_model=3`，508行另有`spi_rkt_model=6`。烧蚀与rocket是不同开关，不能笼统称“生产烧蚀model6”，也不能因此排除rocket缺口。应对实际model3烧蚀、model6 rocket、nonlocal、出生/密度核、滤波、原子辐射逐项核查。滤波系数非零不意味着H=1下每项必定非零，需验证其实际贡献。PLT之外实际启用哪些辐射过程仍需配置/源码对照，不能仅凭“不含PRB/CX”断言全部必需。

### D3：可恢复初态及历史的证据链未贯通（缺证据＋缺实现）

[initialize_native_checkpoint](../src/data/marker_checkpoint.py:77)明确区分continuous与restart：continuous需要保存的clock、前场、反馈与deltas；restart有合法零反馈/余留初始化。不能把普通窗口初始化为restart，不能凭缺part推定空云。

JOREK [project_only](../../JOREK_code/jorek/particles/diagnostics/mod_uncoupled_projection.f90:280)的deltas来自投影解减旧值。因此“文件无deltas”不等于“原则上永远不能重建”：只有证明相邻文件正是该更新前后状态、输出未改写/漏步/过滤阶段一致、精度与事件相位正确时，才可认证重建。此前past-zero特例不能推广。应逐窗口提供 saved / verified_reconstructed / causal_replayed / unavailable 的来源；不从未来状态反推缺失历史。

建议先选v400已存在快照附近的合法restart锚点，以及injInner的可证初态锚点，注册源时钟、上一场、rawD、shard、计划与游标、物种/表格/配置。缺证据时才列精确补采清单，不预先默认必须补导出所有量。

### M3：边界只有身份保存，尚无物理执行资格（缺实现）

[read_native_update_mapping](../src/data/native_state.py:130)只允许`learned_unverified`，物理边界策略会被拒绝。共享DOF合并不等于执行Dirichlet/Neumann/混合边界。修复范围是两个目标网格的实际约束，不要求实现全部JOREK边界配置；验收包括各场/degree边界响应、共享一致性及未知配置拒绝。

### M4：源时钟缩放需要真实范围验收（风险，未证实数值故障）

[JointSourceEncoding](../src/models/native_source_features.py:116)把绝对eventJU作为输入；[physical_reference](../src/models/native_source_features.py:49)的两个时间尺度均为1，而本轮真实时刻约8377–8605 JU。与小fixture不同，线性编码再tanh可能饱和或过度依赖绝对时刻。不是“显式时间必然错误”，也未运行真实训练证明故障。

无训练验收：合法真实输入的逐组幅值、预激活/饱和率、源扰动灵敏度，明确时间原点与尺度；若使用事件相对时间，保持物理事件与检查点语义。变更需版本化，不能悄悄改旧模型解释。

### M5：耦合入口绕过context校验（静态确认，优先修复）

[CoupledNativeField.reference_solver](../src/models/native_coupling.py:415)直接构造NativeReferenceSolver，未复用[原场入口](../src/models/native_field_step.py:156)的units/modes、boundary policy、网格及完整几何核验。后者构造仅提取参数，调用时检查model摘要，不能替代context与模型的几何/边界一致性。构造耦合模型时存在内部几何匹配检查，但不覆盖后传入context。

本轮确认静态防线缺口，未构造错配实例证明具体运行结果。修复验收应以公开入口拒绝错units、错mesh、错误boundary声明、配置漂移；同时保留正确耦合配置和恢复路径。

### M6：直接Torch场步缺少源—场时间绑定（接口缺口）

[CoupledNativeField.forward](../src/models/native_coupling.py:394)接收裸U与source_time，没有当前场时刻；[SourceEventTime.validate](../src/models/native_source_features.py:27)只验证有限时间、正gather和特殊零投影条件，无法判断源是否来自未来或已经过期。参考事件驱动正常路径因果生成源，不能据此声称已经发生未来泄漏；待完善的是直接Torch窗口入口的强制配对契约。应传入可信的带时刻状态/事件上下文，在同一公开入口拒绝错配，不以“时间已作为特征”代替验证。

## 3. 非阻塞近似与禁止扩大范围

- ADR-0022的联合九场网络、同场值/梯度、源闭合特征继续使用；不要求先补完整压力/电阻率/交换显式算子。
- NumPy源事件可作为闭环参考，但不能声称源端到端可微；[advance_coupled_reference](../src/models/native_coupling.py:507)当前为CPU推理事务路径，与给定源Torch场步区分。
- 小云上限/槽位量、定位与投影成本要实测。百万槽位不等于百万存活，也不能直接裁剪而不登记降阶误差。
- 节点latent stencil、Gauss池化保留为已登记学习近似，不当作完整FEM算子。
- 当前[训练器](../src/training/native_curriculum.py:86)仍为field-only，不运行优化器或用训练损失掩盖上述缺口。

## 4. 修复顺序与无训练完成条件

| 顺序 | 切片 | 可复核完成条件 |
|---|---|---|
| P0 | 修复配置解析、context核验、源场时间绑定＋目标支持矩阵 | 两run数组无静默截断；时间表来源/游标明确；错mesh/BC/源时间拒绝；烧蚀3/rocket6/nonlocal/filter分别登记 |
| P1a | 可恢复锚点与窗口资格 | 至少每run一个已证初态；完整clock/history/shard/双源字段；重建证据逐项标记；split包含回放/预热全部支撑 |
| P1b | 可扩展源空间路径（可与P1a并行） | 小网格对照＋真实网格资源受控求值/投影；不靠调高稠密上限 |
| P2 | 目标源配置与边界约束 | 非局域/滤波/出生/辐射差异逐项实现或接受并验证的近似；实际边界独立oracle；源尺度真实范围检查 |
| P3 | 两run无训练完整事件验收 | 从合格初态连续至少两事件；下一源读预测场；不读未来；中途恢复一致；故障回滚；证据覆盖真实配置而非仅fixture |
| P4 | 闭环训练集发布前审计 | 机器可读资格、来源、全支撑隔离、源/场教师语义、单位和scaler来源完备；仍不启动训练 |

P3使用未训练网络仅检验执行/约束/依赖，**不要求随机网络长期稳定或准确**；若预测离开物理支持域，应明确失败并区分模型状态无效与接口故障，不靠真值重置冒充自回归。后续是否启动训练，待这些范围明确完成后另行决定。

## 5. 本轮验证与复现

```bash
.torch-cuda-env/bin/python -m pytest tests/test_native_joint_source.py tests/test_native_coupling.py -q
# 本轮：27 passed in 4.51s；未训练、未运行课程训练测试或优化器。
```

原始文件审计为只读h5py：对§1列出的每个编号检查`values/deltas`存在与shape、`t_now/tstep`、`unique(index)`、projection根键、part存在性及`P_imp.shape`；调用`audit_native_segment`检查各run前4个编号。未扫描全库、未推进真实完整云、未新生成训练窗口，不声称全量265项独立复跑。

本轮只增加审计报告及索引。现有源码/数据缺口未在本轮修复；报告中优先级是实施建议，不是已完成工作。

## 后续实施状态（2026-10-08，保留以上历史审计）

接口防线及显式步长控制见 [入口守卫报告](native-closure-entry-guards-2026-10-08.md)。后续稀疏空间路径、实际资格入口、两个真实锚点盘点与剩余生产缺口见 [稀疏与资格续篇](native-closure-sparse-qualification-2026-10-08.md)。此前“仅审计、未修复”的描述仅指原审计轮次；目前已有实现与 227 项回归证据，但不能将未滤波/无约束参考投影或小 fixture 资格视为目标生产闭环通过。
