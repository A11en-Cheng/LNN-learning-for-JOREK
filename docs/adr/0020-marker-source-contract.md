# ADR-0020：marker 源对象契约、降阶 marker 链、增广状态与登记 restart 策略（2026-10-07）

状态：已接受（2026-10-07）。修订并覆盖 `roadmap_v2.md` 中与之冲突的表述
（§1.2 #10、§3 接口表、§4 Phase A/B1/C、§5 第 3/7 条、§6 A1/A2/C0、§7 风险表），
补充 ADR-0019 的 restart 边界措辞，冻结 B1/B2/C 共用的 S 接口。ADR-0018/0019 仍是
现行架构的权威依据；ADR-0001–0017 是 v1 历史，引用一律标注"（历史）"。

> **2026-10-08 定向修订**：用户授权修改roadmap，故决定6的“原文不动”被
> [ADR-0021](0021-native-state-cfc-window-curriculum.md)取代；本ADR的双源对象、持久状态、
> 事件顺序、共享S与restart物理语义不变。下文“当前实现尚缺”是2026-10-07决策时快照，
> 最新实现范围看conformance及2026-10-08原生状态/场步报告，不由决策文字推定完成。

> **为什么需要本 ADR**：2026-10-07 的源码复核与数据扫描显示，`roadmap_v2.md` 的三处
> 原始表述与 JOREK 源码/实测数据不符（源对象、B1 结构、restart 策略），且 B1/B2/C 的
> 共享 S 接口尚未冻结。`roadmap_v2.md` 本次不改动；本 ADR + 
> `report/conformance-roadmap-v2-2026-10-07.md` 构成对它的权威勘误与一致性映射。

## 背景与证据（全部可复核）

### 1. 源对象：marker run 里记录投影是实际反馈对象（电子能量源及电荷闭合输入）

`models/model502/mod_elt_matrix.f90`（本仓库只读源码 `JOREK_code/jorek`）：

- `:944`：`if (allocated(...) .and. (.not. use_marker))` —— marker 模式下解析
  ADAS 电离能分支被跳过；`:972` 的 `else if (.not. use_marker)` 同样跳过；
- `:983-987`（marker 分支）：`E_ion = 0`、`dE_ion_dT = 0`、`Z_imp = 0`；
- `:989-1003`：marker 分支直接用投影 aux 变量
  `aux_dEion_dt = eq_aux_g(:,1)`、`aux_rad = eq_aux_g(:,2)`、`Z_eff = eq_aux_g(:,3)`、
  `Z_imp = eq_aux_g(:,4)`、`n_imp = eq_aux_g(:,5)`，并令
  `Z_imp /= n_imp`、`Z_eff /= n_imp`；
- `:1079`：`if (use_marker) delta_alpha_e = m_i_over_m_imp * delta_Z_imp` —— 电子密度/
  杂质压力梯度直接用投影的 `Z_imp`；
- `:1087`：`if (.not. use_marker)` 才走 ADAS 日冕 Z_eff 计算；`:1129` 的
  `Z_eff = Z_eff * n_imp` 把 marker 投影的电荷矩回代进电阻率/输运系数；
- `:1179-1199`：辐射 `Lrad` 只在非 marker 分支计算；marker 分支
  `Lrad = 0`、`dLrad_dT = 0`（真实辐射来自投影的 `aux_rad`）；
- `:1214-1245`：`call total_imp_source(...)` 只给出 `source_bg/source_imp`
  （杂质密度源），`rhs_ij_5 :1542`、`rhs_ij_8 :1717` 用它；
- `:1728`：方程 9（T_e）的源是
  `- v * BigR * (aux_dEion_dt + aux_rad) * xjac * tstep` —— 直接消费 marker 投影。

`particles/examples/marker_spi_source.f90`：

- `:535-537` 五通道定义（`Z_imp`、`Z_eff`、`N_imp` 的 `weight·timesteps` 形式）；
- `:540-601` 电荷态/电离能/辐射的 ADAS 逐粒子更新与 `dEion_dt`/`rad_sink` 累积；
- `:623-643` 用形函数把五通道散射进 `feedback_rhs`（`v_1..v_5`）；
- `:442-443`、`:670-676` `feedback_rhs /= rhs_gather_time`，
  `rhs_gather_time = n_steps*timesteps`（**gather-time 平均语义**）；
- 生产 namelist `data/EHL-2/phaseA_dataset_.../jorek_namelist_spitest:528`
  为 `use_marker = .t.` —— 本项目 run 全部走 marker 分支。

**结论**：对 marker run，"被复刻对象 = inj_source 语义、projections 只是监督代理"
（roadmap_v2 §1.2 #10、§3、§5.3、Phase A ②）**不成立**。marker run 的实际场源拆成两条：

1. **解析 shard 密度源**：`total_imp_source → inj_source`，Gauss 点逐点沉积、
   进 RHS、对场显式（杂质密度通道；`jorek` 内部路径）；
2. **marker feedback 五通道**：`projections*.h5` 记录的
   `(dE_ion, rad_sink, Z_eff, Z_imp, N_imp)` 是电子能量 RHS 源与电荷/杂质闭合输入
   —— B1 的目标就是**记录投影本身**。

### 2. B1 缺 marker 链：证据

- 解析 shard 沉积与记录投影的对象级差异（B1 报告 §5）：最小二乘标度差
  11–5.5e5、中位积分比 179–2.1e6、空间相关 0.06–0.29、
  **48–92% 的投影目标质量落在解析沉积为 0 的节点上**；把标度强校到投影会把原子
  预算放大 **22×**（`.scratch/phase-b1-source/metrics.json`）。
- 解析骨干的逐 shard 烧蚀率对照是 1.0021（P10 0.976/P90 1.023），说明
  **烧蚀/碎片演化本身已经对**；缺的是"烧蚀产物如何成为 marker 云、如何输运、
  如何按 marker 出通道"这一环。
- `part*.h5` 快照：1e6 slots、172 MB/帧（`part02710.h5` 实测）、
  每 10 步一帧、单 run ~840 帧 ⇒ 单 run ~144 GB；全粒子逐步回放需要
  ~1e10 粒子步 ×（ADAS 插值 + 场插值 + 4e3 节点沉积），与训练回路不兼容。
  ⇒ **全粒子回放不可行（硬性事实）**；每 10 步快照足以标定降阶输运
  （`i_life`/`P_imp`/`weight`/`x`/`v`，`part04110.h5` 存活 11,168、
  `part11100.h5` 存活 253,566）。

### 3. restart 事件：v400 run 真实存在且生成侧不重导

`data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400` 时间表
（`macroscopic_vars.dat` 的 `@times`）实测 9 个 `dt=0` 事件：

```
2702, 4111, 5871, 6871, 8371, 9421, 10421, 11421, 11711
```

（前一帧分别为 2701, 4110, …, 11710）；`injInner` run 为 6 个
（2704/5681/6441/7221/8611/9291）。事件 n 的 `projections{n-1}` 相对
`jorek{n-1}` 偏移恰为 `-dt_source`（2703: −5.0、4110: −0.5 …），
`projections{n}` 与 `jorek{n}` 对齐。用户决定：**不重导数据，切窗口时按登记事件跳过**
（2026-10-07 指令）。登记表：`data/restart_events.json`。

### 4. 数据契约细节（G）

- `interval_start/end` 在原生布局里**由 `t_now` 推导**（`jorek`，与
  `macroscopic_vars.dat` 交叉校验），不是文件自描述字段 ⇒ 契约改为
  "推导的 interval + 登记 provenance"；
- `freeboundary` 在原生 H5 是**全局 `'F'/'T'` 标志**，不是逐节点 int8；
  `axis_node` 不存在（`axis_node=F`/`treat_axis=F`）；节点级 `boundary`
  是唯一逐节点边界口径；
- 预算序列 `xtime_spi_ablation` 在每次 code restart 归零（B1 报告 §4.2：
  末值/分段总量 ≈ 14.8×），预算必须**按 restart 分段**核算。

## 决定

1. **双源对象契约（A）**。源对象 `source_object` 有两个合法取值：
   `analytic_shard`（`total_imp_source → inj_source` 语义，杂质密度源；B1 的确定性
   碎片沉积）与 `marker_feedback`（记录投影五通道；marker run 的电子能量源与
   电荷/杂质闭合输入）。
   两者不得混用、不得静默互换；每条导出/记录必须登记 provenance。对 marker run，
   B1 的监督目标就是记录投影本身（不再是"代理"）。
2. **B1 = 完整 marker 链（B）**：
   `shard 烧蚀/粒子诞生（确定性）→ 粒子电荷态/电离能/辐射（ADAS，确定性）→
   降阶 marker 输运（场向平流 + 垂直扩散，确定性主项）→ FEM 沉积 + gather-time 平均`；
   NN 只学**有界残差**（输运/闭包/非局域），能带**跨 step 记忆**（split heads）。
   全粒子回放不可行（背景 §2）⇒ 降阶实现目标位于 `src/models/source_module.py::MarkerChain`
   （golden-angle 确定性采样、ADAS SCD/ACD 电荷态、解析 ψ 梯度场向平流、
   `-D⟂∇ln n_imp` 扩散、双线性 FEM 沉积、`÷ gather_time`）。
   当前实现只覆盖单窗口云构造，尚不满足决定 3 的持久云状态。登记的两项降阶差异：
   (i) 每子步发射 `markers_per_shard` 个固定数密度包，
   JOREK 按窗口源强分配粒子数（包分配差异由有界残差学习）；(ii) 输运系数在
   shard 中心评估（marker 云按组平流）。
3. **跨窗口状态与 Phase C 循环（B/D；2026-10-07 源码勘误）**：先区分三种对象。
   **JOREK 参考状态**在正常窗口之间保留活 marker 集合：每个 marker 的位置 `x`、
   单元/局部坐标 `i_elm/st`、权重、`P_imp` 全电荷态概率、上一步速度 `V_prev`
   （Adams–Bashforth 推进需要），以及出生/寿命信息；还保留上一投影的完整节点
   `values`，供下一粒子窗口的输运计算与下一投影的 `deltas` 差分使用。场的前后
   两个状态参与窗口内插值，shard 状态另行推进。`P_imp` 的完整分布用于逐电荷态
   ADAS 电离/复合和辐射，单靠一、二阶电荷矩不能逐项复现。**降阶模型状态**
   可以压缩活 marker 集合，但必须显式保留可推进的云状态及上一投影的降阶表示；
   这属于标定过的近似，不等于 JOREK 完整自由度。仅用
   `h_marker`（GRU 潜状态或五通道全局摘要）替代持久 marker 云，不宣称与 JOREK
   状态一致。**学习记忆** `h_residual` 只服务于有界残差，不能冒充物理状态。
   正常窗口语义为
   `(u_{k−1},u_k,p_shard,k−1,m_{k−1},S_{k−1}) → (m_k,S_k) → u_{k+1}`；
   其中 `m` 是持久云状态，`S` 是投影对象而非粒子状态。推理路径不含未来记录量。
   当前 `src/models/source_module.py::AugmentedState` 仍只有 `(u,p_shard,h_marker)`，
   `MarkerChain` 每窗口重建云；这**尚未实现**上述跨窗口物理状态契约，不得把骨架
   当作已完成的 JOREK 对齐。依据：`marker_spi_source.f90:452-486,598-602,655-663`、
   `mod_particle_types.f90:53-79`、`mod_fieldline_euler.f90:56-78`、
   `marker_spi_source.f90:496-519`、`mod_uncoupled_projection.f90:258-294`。
4. **共享 S 接口（E，冻结）**：五通道顺序
   `(dE_ion, rad_sink, Z_eff, Z_imp, N_imp)`；节点 dof-0 值（`[nodes, 5]` 内存布局，
   是完整 JOREK 投影节点自由度的降维视图）；
   单位 = 记录投影归一化（`t_norm/E_norm` 与 `n_norm=central_density*1e20`）；
   gather-time 平均语义 = `Σ_substeps(deposit)/(n_steps*timesteps)`；
   marker run 用记录投影作真实反馈对象。**五通道不能定义为一个固定的
   `5→9` 加性 RHS 矩阵**：通道 1/2 在方程 9（电子能量）以
   `-(aux_dEion_dt+aux_rad)` 直接入 RHS；通道 3 是电荷平方矩，通道 4 是
   电荷矩，通道 5 是数密度矩。model502 在高斯点截断/除以带下限的通道 5，
   再通过平均电荷、电子密度、`alpha_e`、`Z_eff`、电阻率及离子—电子能量
   交换等闭合，影响方程 1/2/6/7/9 的 RHS 或矩阵系数；通道 4/5 的投影
   增量还进入方程 9。方程 3/4/5/8 在该单元装配函数中没有直接引用
   marker 五通道，但联立九场求解后仍可间接受影响。解析 shard 密度源另进
   方程 5/8，不可混入此五通道映射。依据：`mod_elt_matrix.f90:294-325,
   989-1020,1065-1083,1127-1165,1292,1494,1542,1638,1717,
   1727-1728,1787-1788,1966,2101`。B1（`SourceModule`）、B2
   （`FieldPropagator.source_proj`）、C（闭环循环）共用
   `src/models/source_contract.py`；下游不得自行改变对象/单位/顺序，
   当前 B2 的零初始化五通道线性 forcing 只是待训练的近似接口，**尚未**实现
   上述 JOREK 逐方程/系数依赖，不能据此声称完成精确五通道到九场方程映射。
5. **restart/零步长策略（C；2026-10-07 实测勘误）**：登记过的 `dt=0`
   **流体输出重复帧**允许存在；训练时跳过跨界配对 `(n−1 → n)` 和时间滞后的
   `projections{n−1}`，事件帧 n 可保留。此处的 `S_n=projections{n}` 时间上
   与 `jorek{n}` 对齐，**但不是**零长度粒子采样窗口：重启启动事件先把新建的
   零 RHS 投影为 `projections{n−1}`，供重复时间的首个流体步装配；之后粒子从
   `part_restart.h5` 的时间正时长推进，才生成 `projections{n}`。全部 15 个
   边界的 `projections{n−1}` 完整 `values` 为零；13 个有源阶段的
   `projections{n}` 非零。零投影被首步源码调用链使用，不能写成“运行时跳过”。
   JOREK 粒子状态从 restart 文件恢复，但投影节点在程序启动时重新清零，
   故重启是**粒子状态延续、投影反馈重置**的非连续边界；模型的持久云与前投影
   必须有对应的显式重启策略。未登记 `dt<=0` 仍 fail-fast；
   `zero_dt_policy="registered"` 为数据加载策略，不是 JOREK 动力学定义。
   证据见 `report/jorek-marker-source-restart-primary-evidence-2026-10-07.md` §3–4。
6. **文档隔离与一致性（F）**：`roadmap_v2.md` 保持原文不动；本 ADR +
   `report/conformance-roadmap-v2-2026-10-07.md` 是它的勘误/一致性层，
   含"规划条目 → 实现/证据/状态"表与 B1.2/B2.1 子阶段标签。旧 ADR（0001–0017）
   在任何引用处标注"（历史）"。
7. **数据契约细节（G）**：契约允许"推导的 interval + 登记 provenance"；
   登记 `freeboundary` 全局标志归一化规则（逐节点 `int8` 由生成侧给出或按
   boundary 语义推导）与 `axis_node` 缺失（原生布局无磁轴帧，B0/A′ 已登记）；
   预算一律按 restart 分段（`src/data/budget.py::detect_restart_events` /
   `restart_segments` / `native_step_budget.restart_segment_boundary`）。
8. **可学习残差仍然有界**：`dE_ion/rad_sink` 可走学习记忆头、
   `Z_eff/Z_imp/N_imp` 可走瞬时修正头；这种 split-head 是**降阶误差修正的
   模型选择**，不是 JOREK 的物理状态划分。JOREK 的 `P_imp` 电荷态分布本身
   也跨粒子子步和窗口延续。log-space 乘性修正 clamp 到注册先验界；零初始化
   = 确定性主项精确复现（沿用 ADR-0018 决定 8）。
9. **推理初始化采用完整检查点（A，用户确认）**：遵循 JOREK 的启动、粒子推进、
   投影事件、流体求解和 shard 更新顺序。中途启动允许读取一次 JOREK 流体/shard
   检查点及匹配的 marker 快照，恢复决定 3 的物理状态，再构造登记过的降阶表示。
   匹配以文件内部 `t_now`、`/time`、单位换算和事件位置为准；JOREK 的粒子时间
   可以落后流体时间一个来源步，不能按同名文件强行赋为同一时刻。启动后只用
   自生成状态推进，不逐步读入后续记录的 marker、源或场。已确认无既有活 marker
   的注入前启动使用空云；有源阶段缺少可恢复的 marker 初态时，该启动点不受支持，
   不默认补零或从九场/shard 猜测电荷态历史。

   必须区分三种边界：**JOREK 程序重启**按决定 5 恢复粒子、重建零投影并执行
   启动事件；**连续 rollout 的窗口切分**保持云、反馈、场历史及步进时间信息；
   **模型自身检查点续跑**恢复完整模型状态，不因重新载入模型而附加一次 JOREK
   重启。连续运行的粒子余留时间 `step_rest_time` 跨循环保留，JOREK 程序重启
   才重新置零（`marker_spi_source.f90:311-355`）；不得把训练 chunk 边界当成
   物理重启。该决定确认的是工作流和状态语义；降阶误差仍须明确登记、实测，
   不宣称与全粒子 JOREK 逐值相同。当前源码尚缺完整初始化适配器、持久云传递及
   相应续跑验证，不能把本决定视作实现完成。

## 对 roadmap_v2 的覆盖清单（不改原文件，按本 ADR 读）

| roadmap_v2 原文位置 | 原文要点 | 本 ADR 的覆盖 |
|---|---|---|
| §1.2 #10 | "背景三源不存在；projections 与内部解析沉积不是同一对象" | 保留对象差异事实，**反转结论**：marker run 里 projections 是真实场源（决定 1） |
| §3 接口表 (1) 源强行 | "被复刻对象是 inj_source 语义；projections 只作监督/交叉校验" | 改为双源对象；marker run 的目标就是记录投影（决定 1） |
| §4 Phase A ①② | "被复刻的是 total_imp_source→inj_source；projections 不得作推理输入" | 推理输入仍是自生成 S；但 marker run 的 S 真值 = 记录投影（决定 1/4） |
| §4 Phase B1 | "shard/p + 解析骨干 + 记忆 + 有界修正" | 改为持久降阶 marker 云 + 前投影反馈 + 有界学习残差；现有 `h_marker` 不等价（决定 2/3） |
| §4 Phase C | `S_k = Source(u_k, p_k)`；p = shard 状态 | 增广状态循环（决定 3） |
| §5 第 3 条 | "源对象语义以 inj_source 为准；projections 只做监督" | 决定 1 |
| §5 第 7 条 / §6 A1/A2 / §7 时序行 | "dt<=0 直接报错；restart 边界剔除或登记" | 决定 5（登记事件保留并跳过跨界/滞后） |
| §5/§6/§7 无"共享 S 接口" | 接口未冻结 | 决定 4（新增冻结接口模块） |

## 后果

- B1 报告 `report/phase-b1-source-module-2026-10-07.md` 的 "proxy（登记代理）"
  框架作废（顶部加勘误）：记录投影是 marker run 的真源；"proxy vs inj_source 差异"
  一节改写为"解析 shard 沉积 vs marker feedback 的对象差异与交叉校验"。
- `report/data-contract-v3.md` / `data-export-spec-v3.md` 更新源对象、
  restart 登记、推导 interval、freeboundary/axis、预算分段措辞。
- `src/data/native_run.py` 默认 `zero_dt_policy="registered"`；
  `StepContractReport` 增 restart 登记/跨界/滞后字段，删去“退化粒子窗口”假设；
  shard 时间标注改为真实帧时间（`jorek{k}` = T_k）。
- `src/data/budget.py` 增 restart 分段（`detect_restart_events`、
  `restart_segments`、`segmented_total_atoms`、`native_step_budget
  .restart_segment_boundary`）。
- `src/models/source_module.py` 需使 `MarkerChain`/`AugmentedState` 真正携带
  持久降阶云与上一投影，不能仅携带 GRU；`src/models/source_contract.py`
  冻结共享 S 接口；`FieldPropagator` 的 5 通道线性源注入仍是近似，
  不能作为九方程精确耦合已完成的证据。
- 未解决/待办（不属于本 ADR 的硬问题）：marker 快照标定降阶输运系数
  （Phase B1.2 的下一步）、`part*.h5` 的 `i_life` 语义确认、内部
  `inj_source` 导出（用于解析 shard 密度源的监督）。
