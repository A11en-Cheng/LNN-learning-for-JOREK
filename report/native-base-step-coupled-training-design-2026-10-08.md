# base 连续耦合与真实节点监督：公共训练契约（2026-10-08）

状态：本轮用户选择的设计及已实现的无训练接口；验收范围见文末。依据：AGENTS、ADR README、roadmap_v2、ADR-0018～0022，以及 large-step-feedback-timing、sparse-qualification、anchor-certification 三份报告。旧 ADR 仅为（历史）。本轮禁止训练、optimizer.step、拟合与超参搜索，不选择最优窗口数字。

## 公共 Interface 与文件所有权

数据/训练方新增 `src/data/base_coupled.py`、`src/training/base_coupled.py` 与 `src/training/coupled_metrics.py`。源、含源场、完整事件、检查点四个已确认的公开 Interface 为测试 seam。模型方继续拥有 `src/models/`；本轮不同时修改这些文件，不改变现有公共模型类型。

训练 Adapter 使用完整事件依赖 `advance(state, control)`，内部所有者负责真实的 source → joint nine-field candidate → shard(OLD field) → atomic commit。数据不实现另一套粒子调度。NumPy 参考 Adapter 调用现有 `advance_coupled_reference`；可微事件 Adapter 必须由模型方显式声明，不能由训练方把 NumPy 包装为可微。

公共消息：完整跨段状态只由事件所有者持有；数据只提供已隔离的初始化证据、控制和只读标签。新增类型只存在数据/训练层，不要求模型 agent 同步修改已有类型。控制序列、mask、教师策略、cut cursor 和 checkpoint 外层 envelope 见下文。若模型方新增可微完整事件，只须提供同一事件 seam 的 Adapter 及其真实梯度能力声明。

## 学习对象与梯度

主学习对象是含源联合九场完整原生增量和时间记忆；不先求冻结容量解，不重新打开完整 JOREK 矩阵移植前置。源中的 shard 烧蚀/出生、marker 电荷态与辐射、输运、FEM 投影/gather，以及 OLD 场 shard 更新是声明版本的确定性路径。既有可训练 source encoding 属于场输入编码；允许未来显式有界源残差，必须单列其参数和记忆的所有者、监督及梯度能力，不能把编码器称为已训练的物理源模型。

三个能力必须区别：

1. **NumPy 完整耦合事件**：自生成源、场、shard 因果推进，但现有参考场 solver 转为 NumPy；整个事件为 nondifferentiable reference，只有数值贯通验收。
2. **给定合法源 Torch 场 BPTT**：场/记忆图连续，给定源是条件输入；不是源—场端到端。
3. **stop-gradient 源重算 + Torch 场事件**：源消费当前预测场的 detached copy，数值上连续反馈，梯度仅沿场/学习记忆或显式 Torch 部分传递；不是穿过 source/shard 的端到端梯度。未来 fully differentiable Adapter 必须分别声明 source、field、shard 能力并测试。

监督 loss 保持 Torch 图；记录指标用 detached 值。不得把 backward 验证称作参数拟合，本轮没有 optimizer。

## 时间与控制

`coupling base step` 是一次源/场/shard 完整交换的实际控制步；固定段内 K 个 base 必须调用 K 次完整事件。marker 子步由初始化实际步定义，residual 跨窗保留，gather 是实际整数子步数乘实际粒子步。它们不等于 base，也不等于记录间隔。

`StepControl` 显式携带开始/结束物理时刻、正 dt、控制来源、事件相位、事件号与来源。普通事件结束时间须为开始时间加 dt；registered startup 可有重复场时间，仍有正调度控制，须显式 phase，不能当普通零步。区间内按控制序列累计物理时间并匹配真实记录；文件 index 仅为来源身份，不作时钟。控制应覆盖全部 burn-in 和预测事件；不从 namelist 数组推断，不自动 subdivide 一个已记录大步。

`external_control` 允许自主任务；`recorded_schedule_replay` 只允许已明确声明的离线任务。外层序列记录 origin 并进入检查点指纹。没有中间真实记录时不生成均匀伪标签，不用左右真值插值。已有预测 U/rawD 定义的数值场采样继续由源模块处理，保留其 static/short-interval/residual 分支，不加无依据 clamp。

训练展开长度是实际事件数与累计秒数；监督间隔由真实记录时间集合决定。监督 mask 按 field、marker projection、独立 shard density、shard position 分开；没有某对象标签时 mask=false，缺失不是零值。projection 标签有自身事件时刻，不把五通道的时间硬套到 shard 或场输出时间。

## burn-in 与初始化

初始化要求完整 U/rawD/前场、marker 全状态、feedback/rawDP、shard、实际时钟/控制及几何配置；合法 restart 按注册清零/启动，普通快照不清零。学习记忆可显式 cold-start，物理状态不可 cold-zero。缺 marker 初态不能用更长 burn-in 弥补。

预热完整物理状态、投影历史、源残差记忆与场时间记忆。默认从已认证轨迹初态或带来源/配置指纹的因果缓存推进；预热 loss mask 全 false。可有历史一致教师重放，但只允许下述一致替换规则，不插入未来标签。不同 burn-in 长度必须固定预测起点和预测物理时长，分别给出起始初态/缓存支撑、真实记录数、事件数和物理时长。

候选由可认证历史支撑允许的时长/记录节点产生，不预设本轮最优数字。后续比较判据：相同任务与起点的九场/梯度/投影/shard 误差，物理支持域、累计误差、完整状态差异及成本；潜记忆稳定仅是一项诊断，不证明完整状态充分。实验预算和候选具体值另行批准。

## 输入、教师与噪声

当前推进只接收上一次已提交状态、当前显式控制及当前时刻已生效参数。真实右端在事件提交后进入 loss/指标，绝不进入当前 advance。计算 loss 不替换状态。正式 AR 不读未来记录的物理量，离线 recorded control 资格另外报告。

field/source/shard 教师 mask 分开列出，但自生成闭环中禁止独立随机拼接：允许 `none`，或在明确事件边界读取**整个一致物理 bundle**（U/rawD/前场/marker/feedback/rawDP/shard/clock），并且声明学习记忆重置后再预热或从同一因果轨迹缓存恢复；三 mask 必须共同为 true。替换发生在该节点 loss 之后、下一事件之前，并记为教师 intervention，不自动调用 jorek_restart。给定源任务可仅源条件教师，但必须使用合法 source interval/rawDP/gather，且任务名不同；本轮完整事件迭代器不悄悄启用此分支。

噪声默认只作为**观察扰动**作用于允许的历史/当前完整原生系数输入，真实标签和已提交物理状态不改。声明每场/degree native-unit scale、显式 amplitude、随机种子、空间相关基与时序相关标识、训练 split 统计来源。共享 DOF 同一扰动、Dirichlet mask 禁止扰动，约束/支持域 validator 必须通过；同一扰动系数重新求 Gauss 值、兼容梯度和闭合。rawD/投影历史/云不独立扰动。

若未来要物理状态噪声，须另有保持完整历史一致的轨迹变换，不能把当前 U 噪声当新的合法 committed state。观察噪声不隐含 TF；教师 bundle 替换先完成一致性，再在模型允许的观察支路加噪。本轮 amplitude 仅在解析 fixture 指定，真实噪声尺度不搜索。

## 连续状态、TBPTT 与检查点

独立样本从认证初态或认证缓存开始，再执行全部 burn-in；顺序片段传入上一段完整已提交状态与 cursor，不重复 burn-in，不从标签初始化。不要求每个窗口以 restart 为中心；后移窗口可以从因果延续状态/认证缓存开始。

TBPTT 只 detach 图中所有跨段 Tensor，保留物理值、raw histories、cloud、时钟、学习记忆和控制 cursor；不清空记忆，不清 residual，不产生 restart。截断 cursor 与监督 mask 是独立配置。NumPy reference 无图也不重置状态。

外层 checkpoint envelope：schema/version、sequence/控制/参数/几何/模型指纹、完整 runtime（既有模型 checkpoint 或 Adapter payload）、source/shard 配置、学习记忆及梯度能力声明、cursor、预测起点、burn-in 结束 cursor、监督mask、教师策略与已执行intervention、TBPTT cuts、噪声配置/RNG 状态、完整支撑及 split。参数更新后缓存必须失效或重新认证，不能沿用旧权重轨迹冒充当前模型状态。本轮 checkpoint 不含 optimizer。

## split 与资格

先声明 run/时间 blocks，再为候选列出初始化、历史、缓存来源、全部回放和预热及预测支撑；全支撑须在同 split，其他 split 的标签不得通过初始化/统计/缓存泄漏。缓存只把计算搬到先前，不缩短证据支撑。普通快照/restart 是初始化与事件边界，不是固定窗口中心。

动态资格分四轴：数据状态证据、当前 Adapter/model 的实际能力声明、控制来源与任务适配、全部支撑 split 隔离；合并条件源/自生成资格由这些轴与任务所需字段计算。不硬编码 false 或冻结旧缺口清单。两 run 参数从原始绑定保留，特别 ns_radius_min=0.05/0.01，不用 run ID 替代条件。

## 统一 loss 与指标

主指标入口按 run、task、burn-in 秒/记录数、预测物理时长、实际 dt/前 dt 比及教师策略分组；使用 Pearson r（加权中心化协方差），不称作 R²。常量预测/目标 r=null，状态分别列明；空集合所有指标 null；目标范数精确零时相对 L2=null 且标 zero_reference，绝对误差仍报告。近零目标用调用者显式的同单位 threshold 判 undefined，不用 epsilon 造完美分数。

marker 五通道分别报 r、relative L2、MAE/RMSE；独立 shard density 单列，不使用 c5。九场逐场报告完整原生 degree（节点系数使用 unique DOF 或明确的存储节点口径）、Gauss 值、由同一系数求值的兼容梯度每方向。Gauss 积分使用真实体积权重，空间区域 mask 显式、不得丢掉不利样本。原生 degree 单位不同，不将 degree 混成有物理意义的单一空间范数。

shard 用固定 identity/group 匹配，位置必须转换到 Cartesian 米（R/Z/phi 不是三个米坐标）；逐坐标与轨迹 Pearson r、米单位位置 RMSE、相对共同初始位置的位移误差。unmatched/liveness mismatch 独立列数和身份，不能静默丢弃；匹配子集误差标 coverage，不宣称总体误差。

loss 只含 coefficients、rawD、Gauss values、compatible gradients 四类数据监督，不新增 PDE residual。每场/degree/方向保存未加权 normalized MSE、native-unit-squared MSE、显式 weight 的贡献，以及实际总loss；归一化尺度必须正且来源为 train-only statistics 或声明的 fixture physical scales。总loss 是各类各场 mean 后加权之和，逐场贡献除九后求和必须等于 total。

指标首先逐真实节点计算，再同组报告等节点时间平均；非均匀记录不能称等节点平均为物理时间积分。如需要时间加权，必须显式 interval weights 并保存分母。本轮不预设任何性能 gate。

## 无训练验收计划

先测试再实现：K 次事件与两个顺序片段的完整状态相同、checkpoint 恢复相同；控制变步长按累计时间匹配标签；右端标签篡改不影响 advance；loss 不替换状态；TBPTT 保值截断图；教师不一致 fail-fast；观察噪声保持边界/共享 DOF，并由解析线性场验证兼容梯度；Pearson/相对误差/常量/空集/shard 匹配由独立手算样例验证。

真实数据只核验内部时刻、显式 recorded controls、完整系数/rawD、参数与实际标签 mask；缺初态/普通 rawDP/生产支持域不补零。fixture 贯通不得给生产数据自动加资格。首次训练及 burn-in 基线实验留待本轮贯通验收后批准。

## 已实现的接口与验收范围

- `src/data/base_coupled.py`：显式逐步控制、真实节点及对象自身时刻、完整支撑隔离、动态四轴资格、固定预测起点的 burn-in 候选比较；`read_recorded_node` 读取原生九场与 rawD，不伪造缺失标签。
- `src/training/base_coupled.py`：完整事件迭代、顺序状态携带、一致教师 bundle、TBPTT 截图与完整检查点。检查点包含模型权重、物理状态、学习记忆、控制 cursor、教师计划及 RNG；恢复核验配置和序列指纹。`ReferenceEvents` 不可微；`TorchRecomputedSourceEvents` 保持九场/记忆图，NumPy 源及 shard 停止梯度，不能称为源—场端到端可微。
- `src/training/coupled_metrics.py`：`recorded_supervision` 统一入口、逐场四类 loss、五通道/独立密度源、九场三种表示、固定身份 shard 指标与显式轨迹聚合。归一化尺度由调用者提供并绑定来源，本轮不拟合统计量。
- `src/training/base_coupled_noise.py`：完整原生观察噪声、共享 DOF/边界约束及兼容梯度/闭合重求。它是观察接口，尚未接入模型的观察支路；不把加噪观察偷偷替换成物理 U。

`run_sequence` 的 `stop_cursor` 支持分片；继续传同一 `SequenceSession`。真值节点通过 `loss_fn` 观察，默认没有教师替换。条件源的数据资格可表达，但完整事件 runner 对尚未提供专用 Adapter 的 conditional-source 执行明确拒绝，不能把自生成源执行冒充条件源任务。

测试使用小规模解析/参考 fixture，覆盖固定及变步长 K 次完整事件与分片一致、恢复至新模型后的权重/状态一致、未来标签隔离、loss 不重置状态、教师计划恢复及支撑校验、TBPTT 保值断图、解析噪声/梯度、常量/空集/极小非零范数和轨迹身份。fixture-manifest 的 eligible 仅适用于该 fixture，不是 phaseA 认证结果。

验收结果：新增接口 16 项通过；连同既有耦合、原生窗口、控制检查点、上下文与时间特征回归共 **107 项通过**（21.02 秒）。JUnit 保存在 `output/native-base-step-training-2026-10-08/final-interface.xml` 与 `regression.xml`；`git diff --check` 通过。

真实记录核验产物为 `output/native-base-step-training-2026-10-08/phaseA-record-interface.json`：v400 4112～4115、injInner 2710～2713 是有限的接口检查例子，不是选择的训练窗口/最优预热长度。直接核验场文件内部时间和 rawD，以记录表显式构造 recorded_schedule_replay；完整场形状均为 `[9,4,1,4125]`。两例控制来源、split 隔离各通过 2/2；数据状态、执行模型绑定及合并资格各 0/2，逐例保留 pending 原因。保存完整 namelist 绑定和 ns_radius_min=0.05/0.01；缺身份、独立 shard density 或普通反馈历史的标签不补零。未在真实生产状态上执行耦合 rollout；生产源码 dirty 版本及锚点限制继续以 anchor-certification 报告为准。

可复核命令（输出文件已存在时审计工具拒绝覆盖，请选新路径）：

```bash
PYTHONPATH=. .torch-cuda-env/bin/python tools/audit_base_coupled_contract.py --spec config/native-base-step-phaseA-example.json --output /tmp/phaseA-record-interface-review.json
.torch-cuda-env/bin/python -m pytest -q tests/test_base_coupled_training.py tests/test_coupled_metrics.py tests/test_base_coupled_noise.py
```

本轮没有 optimizer.step、参数拟合、训练、超参搜索或生产 JOREK 重跑；JOREK 源码和原始数据只读。首次训练预算、burn-in 基线与真实噪声幅度仍待后续贯通验收后批准。
