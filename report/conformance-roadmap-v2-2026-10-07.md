# roadmap_v2 一致性报告（conformance）：规划条目 → 实现/证据/状态（2026-10-07）

**背景与更新**：2026-10-07曾要求roadmap不改；2026-10-08用户授权修订，现按
[ADR-0021](../docs/adr/0021-native-state-cfc-window-curriculum.md)更新完整原生残差学习、
兼容梯度、参数路由、CfC/可替换核心与TF→AR课程。物理流程仍以
[ADR-0020](../docs/adr/0020-marker-source-contract.md)为准。旧ADR（0001–0017）
一律按“（历史）”处理。下文旧条目的组件通过不等于新完整场步/闭环通过。

## 2026-10-08 当前增量状态

| 条目 | 状态 | 边界 |
|---|---|---|
| 完整原生状态/历史/检查点 | 已有组件验证 | 见native-state-fidelity-resume报告，不是训练续跑协议 |
| 原生联合学习场步 | 结构接通；本轮相关48测试通过 | 全degree、可微单步、零反馈推理Adapter；节点P1空间近似和共享index均值；无物理BC验收 |
| 新架构残差主干/CfC/兼容梯度/参数路由 | 决策与契约已更新，待实现 | 当前77维输入只含U/rawD/时间，没有显式梯度、物理condition或CfC |
| 因果窗口与TF→AR课程 | 待实现/训练 | 见native-window-cfc-curriculum-design；不把scheduled sampling叫作DAgger |
| 五通道与自生成源C0 | 未验收 | 局部公式/载荷已验证不等于反馈接通；推理Adapter仍拒绝非零反馈 |

两个phaseA逐步片段已核查，不能继续笼统写“细粒度数据未交付”；完整run/初始化/投影历史覆盖
仍需窗口manifest逐项检查。当前残差场步没有使用旧LTCFieldStep，旧时间模块结论不能直接套用。

> **持久状态实现补充（2026-10-07）**：一次原生检查点读取、小规模完整参考槽位状态、
> 跨窗口云/反馈/余留延续、模型续跑及独立 JOREK restart 启动事件已实现并有组件测试。
> 依据与命令见 [实现报告](marker-persistent-state-implementation-2026-10-07.md)。
> 降阶适配仍为二维/双线性/lumped/PLT-only，未标定云压缩，未验证完整自生成源闭环；
> 五通道到九方程精确耦合仍未实现。历史测试数字不作为本次验收记录。

## 0. A–G 逐条结论

| 编号 | roadmap_v2 原表述 | 覆盖决定 | 代码/数据证据 | 状态 |
|---|---|---|---|---|
| **A** 源对象 | §1.2 #10、§3 接口表(1)、§5.3、Phase A②：被复刻对象 = `inj_source` 语义；`projections` 只作监督/交叉校验 | ADR-0020 决定 1：**双源对象**；marker run 里记录投影是能量/辐射/电荷通道真实源，解析 `total_imp_source` 只供杂质密度源 | `mod_elt_matrix.f90:944,972,983-1003`（marker 分支 E_ion/Lrad=0、消费 `eq_aux_g`）、`:1079,1129`（Z_imp/Z_eff 回代）、`:1179-1199`（Lrad=0）、`:1219,1542,1717`（source_imp 密度源）、`:1728`（eq9 用 `aux_dEion_dt+aux_rad`）；`marker_spi_source.f90:535-601,623-643`；`jorek_namelist_spitest:528 use_marker=.t.` | ✅ 已修正（ADR-0020 + 契约 v3 §1.1/1.2 + B1 报告勘误） |
| **B** B1 结构 | Phase B1：shard/p + 解析骨干 + 记忆 + 有界修正 | ADR-0020 决定 2/3：持久 marker 云、ADAS 电荷态、降阶输运与投影，NN 只学有界残差 | `MarkerChain` 接收/返回 MarkerRuntime，完整 P_imp/v/身份、前反馈及 clock 跨窗口延续；GRU 为残差记忆 | 🟡 持久参考组件已验证；降阶标定待做 |
| **C** restart | §5.7、§6 A1/A2、§7：时间戳严格递增、`dt<=0` fail-fast、restart 边界剔除或登记 | ADR-0020 决定 5/9：登记流体重复时间，区分程序重启与窗口切分；粒子恢复、反馈清零 | 15 个边界的前编号投影全零；后编号投影来自正时长采样，见一手证据报告；loader 已有登记跳过策略 | 🟡 零投影启动、正粒子窗口与初始化已有组件验证；完整闭环待验证 |
| **D** Phase C 循环 | §3 骨架/Phase C：`S_k = Source(u_k, p_k)`，p = shard 状态 | ADR-0020 决定 3/9：一次完整检查点初始化，持久云/前投影/场历史按 JOREK 顺序延续，随后自回归 | `AugmentedState`/`ReferenceRollout` 已携带物理云、反馈、场历史/时间与计划；模型续跑有逐位一致验证 | 🟡 参考事件驱动器已实现；未宣称完整闭环 |
| **E** 共享 S 接口 | 无（B1/B2 对象/单位语义未冻结） | ADR-0020 决定 4：前两通道进入电子能量 RHS，后三通道参与闭合与矩阵系数；dof-0 是降维视图 | `source_contract.py` 已有五通道形状/顺序校验；`source_proj` 的线性 forcing 尚未复刻逐方程依赖 | 🟡 数组接口已有；精确耦合未实现 |
| **F** 文档隔离 | roadmap 引用 ADR-0015/0016 未标"（历史）"；无 conformance 表 | ADR-0020 决定 6：roadmap 不改；旧 ADR 引用标"（历史）"；conformance 表（本报告 §1/§2）；子阶段标签 B1.2/B2.1 | 本报告 + `docs/adr/README.md` 更新 + ADR-0020 覆盖清单 | ✅ 已修正 |
| **G** 契约/预算 | A1 要求显式 interval/freeboundary/axis_node，与原生布局不一致；20× 已结案但预算未分段 | ADR-0020 决定 7：允许"推导 interval + 登记 provenance"；登记 freeboundary 全局标志/axis 缺失归一化；预算按 restart 分段 | 契约 v3 §4.4；`native_run.py` 推导 interval + provenance；`budget.py::detect_restart_events`/`restart_segments`/`segmented_total_atoms`/`restart_segment_boundary`；测试 `test_restart_segmented_budget` | ✅ 已修正 |

## 1. 规划条目 → 当前状态（roadmap_v2 §0–§8）

### §0 定位与隔离

| 条目 | 状态 | 证据 |
|---|---|---|
| 取代 `new_version.md`；目标 = 自生成源 + stencil-conv 九场 + 错步闭环；推理路径无未来记录量；物理性优先、不预设 gate | ✅ 保持 | ADR-0018/0020；本仓库代码/测试 |
| 文档隔离：只以现行 ADR + 契约 §3.1a 为依据；旧 ADR 标"（历史）" | ✅ 保持并加强 | `docs/adr/README.md`；ADR-0020 决定 6 |

### §1 框架盘点

| 条目 | 状态 | 证据/备注 |
|---|---|---|
| §1.2 #10 背景三源不存在；projections 与解析沉积非同一对象 | 🟡 覆盖 | 对象差异事实保留；"projections 只作代理"被 ADR-0020 决定 1 覆盖 |
| §1.2 #14 无子帧不得造伪真值 | ✅ 保持 | 不造插值伪真值；restart 的粒子采样窗口不能从重复流体时间推作零长度 |
| §1.2 #15 记录输出时序（`S_n` 驱动 `u_n→u_{n+1}`；`part{n}`=T_{n−1}） | ✅ 保持 + 补充 | ADR-0019；补充 restart 登记规则 |

### §2 目的（A 自洽 / B 时序 / C 接口）

| 条目 | 状态 | 证据 |
|---|---|---|
| 给定 `(u_k,p_k)` 无未来源推进（A） | 🟡 组件就绪 | B1 marker 链/B2 推进器/C 骨架；闭环训练另派（非目标） |
| 错步格式训练/推理（B） | ✅ | 契约 v3 §2；loader 配对断言 |
| 确定性复用 / 小网络修正 / 端到端边界（C） | ✅ | ADR-0020 决定 2/8；split-head 有界残差 |

### §3 骨架与接口

| 骨架环节 | 状态 | 证据 |
|---|---|---|
| (1) 源：解析 shard 密度源 + marker feedback 五通道 + FEM 沉积 | ✅ 修正 | ADR-0020 决定 1/2；`source_module.py`；契约 v3 §1.1 |
| (2) 九场一步推进 = stencil-conv（禁 GNN/谱） | ✅ | B2/B2.1 报告；`implicit_stencil.py`/`field_propagator.py` |
| (3) 碎片推进 = `FragEvolve` 确定性 + 小修正 | ✅ | `source_module.py` 碎片 4 阶子步；B1 验收 1.0021 |
| 共享 S 接口（新增） | ✅ | ADR-0020 决定 4；`source_contract.py` |

### §4 Phase A–E

| 阶段 | 状态 | 证据 |
|---|---|---|
| Phase A（契约 + 数据粒度） | ✅ A0/A′ 完成；A1 细粒度数据 = 用户生成（已登记开放问题） | 契约 v3、导出规范 v3；`phase-a-prime-native-loader-2026-10-07.md` |
| Phase A 回退（帧级自洽，禁插值伪真值） | ✅ registered 元数据已更正 | 正粒子窗口从启动零投影时间推导，未登记仍失败 |
| Phase B0（stencil + 边界） | ✅ | B0 报告（16 测试；BC 映射/多层 NaN） |
| Phase B1 解析骨干（B1.1） | ✅ | B1 报告：烧蚀 1.0021、原子预算 1.0002、非局域分支必需 |
| Phase B1 marker 链（B1.2，ADR-0020 新增） | 🟡 持久参考组件已实现 | marker 初态必需；云/反馈延续；大规模降阶压缩与输运标定待做 |
| Phase B2/B2.1（原节点实现）→完整原生残差推进 | 🟡 组件/结构已有，目标未验收 | 旧B2/B2.1测试仅覆盖旧组件；CfC、梯度/condition、物理BC与非零反馈接入待做 |
| Phase C（错步闭环） | 🟡 状态与事件组件已有验证 | 一次原生检查点读取、持久云/前反馈、模型恢复与程序 restart 区分已实现；完整闭环待做 |
| Phase D（基线测定） | ⏸ 未做（依赖 C；不预设 gate） | 契约/ADR 口径就绪 |
| Phase E（JOREK 单步对照） | 🟡 B2.1 已有单步弱形式对照；接口/成本报告未做 | B2.1 §5.2 |

### §5 训练与实验规范

| 条目 | 状态 | 证据 |
|---|---|---|
| 1 推理路径无未来记录量 | ✅ | B1 测试 `test_module_output_does_not_depend_on_future_frame`；B2 无未来帧输入 |
| 2 时序契约 `(u_k, p_{k−1}, S_k) → u_{k+1}` | ✅ | ADR-0019；loader/契约断言 |
| 3 源对象语义 | 🟡 覆盖 | ADR-0020 决定 1（双源对象；marker feedback 为真源） |
| 4 stencil-conv only | ✅ | B2/B2.1；无 GNN/谱分支 |
| 5 condition 物理子集 + FiLM | ✅ | B2/B2.1；导出规范 §4.2 |
| 6 修正有界 + 输入 clamp | ✅ | B1 clamp ±log2；B2 FiLM 恒等初始化 |
| 7 时间断言/restart | 🟡 覆盖 | ADR-0020 决定 5：登记事件通过、未登记 fail-fast、跨界/滞后跳过 |
| 8 不预设性能 gate | ✅ | 全部报告登记数字，无阈值 |
| 9 旧组件复用但不作对照 | ✅ | ADR-0018；契约/报告标注（历史） |
| 10 stencil 每层 BC 复位 | ✅ | B0/B2/B2.1 测试 |
| 11 参数路由与残差耦合 | 🟡 ADR-0021已修订，待实现 | 旧浅层/形状不变测试只为旧设计证据，不作新模型限制 |

### §6 实验清单

| 编号 | 状态 | 证据 |
|---|---|---|
| A0 数据契约 v3 | ✅ | `report/data-contract-v3.md`（2026-10-07 修订） |
| A1 细粒度数据接入 | 🟡 逐步phaseA片段已核查；全覆盖待审计 | native-state-fidelity-resume报告；不同对象输出粒度分开处理 |
| A3 原生训练窗口 | ⏸ 契约已写，未实现 | native-window-cfc-curriculum-design报告 |
| A2 旧数据配对修正 | ✅ | loader 按 `(u_k, p_{k−1}, S_k)` + 登记 restart；测试全绿 |
| B0 stencil 算子 | ✅ | B0 报告 |
| B1 自生成源 | 🟡 B1.1 已有验证 / B1.2 持久参考组件 | 原生初始化与续跑已验证；大规模降阶标定待做 |
| B2 九场推进器 | 🟡 完整场结构已接通，未训练/物理验收 | native-joint-field-step报告；新架构见ADR-0021 |
| C0 错步闭环 | 🟡 状态与初始化组件已实现 | 参考事件顺序/续跑已验证；自生成场闭环未验收 |
| C1 路径差归因 | ⏸ | 未做（Phase C 另派） |
| D0 基线测定 | ⏸ | 依赖 C |
| E0 JOREK 单步对照 | 🟡 | B2.1 弱形式单步对照；接口/成本报告未做 |

### §7 风险表

| 风险行 | 状态 | 备注 |
|---|---|---|
| 源对象争议 / 时序配对错误 / restart 边界 | 🟡 组件已对齐 | 零投影启动事件、正时长采样、完整检查点读取及续跑新增测试；不等于完整闭环 |
| 显式 stencil cap / ZK∥ 冻结 / 边界 BC / Mach-1 | ✅ 已处理或登记 | B2.1 §5/§6；未复刻离对角项登记 |
| 记忆通道退化 | 🟡 降阶误差待标定 | 物理云/完整 P_imp/推进历史已显式保留；split heads + h_residual 仅作学习残差 |
| 闭环发散 | ⏸ | Phase C |

### §8 与 v1 差异清单

源对象/训练语义及restart按ADR-0020更新；梯度、时间核心和参数限制按ADR-0021定向修订，
不再笼统认为旧结论全部适用。

## 2. 子阶段标签（供报告/issue 引用）

| 标签 | 含义 | 状态 |
|---|---|---|
| A0 | 数据契约 v3（S 对象/时序/粒度） | ✅ |
| A′ | 原生逐步 loader + 验收 CLI + condition/budget 读取 | ✅ |
| A1 | 用户生成细粒度数据接入 | 🟡 两个逐步片段核查通过；其余覆盖待manifest |
| A3 | 原生因果窗口/梯度/参数数据契约 | 🟡 设计完成，待实现 |
| B0 | stencil 算子 + JOREK 边界层 | ✅ |
| B1.1 | 解析 shard 骨干（烧蚀/碎片演化/预算） | ✅ |
| B1.2 | 持久降阶 marker 链 + 前投影反馈 + 有界残差 | 🟡 完整参考状态/初始化/传递已验证；降阶标定待做 |
| B2 | 全原生残差推进 + 参数条件化 + CfC/可替换时间核心 | 🟡 仅原生场步结构已有验证 |
| B2.1 | 逐项权重/通道映射/准稳态/边界复刻 | ✅ |
| C0 | 错步闭环（含完整检查点初始化及增广状态） | 🟡 物理状态与续跑组件已验证；完整闭环待做 |
| C1 | 路径差归因 | ⏸ |
| D0 | 新架构基线测定 | ⏸ |
| E0 | JOREK 单步对照与接口 | 🟡 |

## 3. 旧 ADR 引用审计（F）

`roadmap_v2.md` 引用旧 ADR 的**未标"（历史）"**位置（文件按指令不改，故按
`docs/agents/domain.md` 的替代路径把规则重述进现行 ADR）：

| roadmap_v2 行 | 引用 | 处置 |
|---|---|---|
| :59、:143、:157 | ADR-0016（"projections ≠ inj_source"、源对象语义、投影定位） | ADR-0020 决定 1 重述并覆盖（marker run 记录投影 = 真源）；ADR 索引已标历史 |
| :142 | ADR-0016（FEM 组装确定性） | ADR-0018 决定 3/5 已重述；ADR-0020 决定 2 沿用 |
| :190、:334 | ADR-0015（学习修正启用条件/先例） | ADR-0018 决定 8 已重述；ADR-0020 决定 8 沿用 |
| :256 | ADR-0012（LOO 可选） | ADR-0018 决定 10 已重述 |

本任务新增/修改的文档（ADR-0019 补充、ADR-0020、契约 v3、导出规范 v3、
B1 勘误、本报告、`docs/adr/README.md`、`docs/agents/domain.md`、`CONTEXT.md`）
全部使用"（历史）"标注或直接引用 ADR-0018/0019/0020。`docs/adr/README.md`
已把 0016 的"projections 只作监督"结论标注为被 ADR-0020 覆盖。

## 4. 硬问题（按"只有数据/代码做不到才停"的协议）

**H1（已确认的硬事实，不影响规划目标）：全粒子回放不可行。**

* 反证：`part02710.h5` = 172 MB / 1e6 slots（`P_imp` 1e6×11 float64、
  `x`/`v` 各 1e6×3 float64、`weight` 1e6）；单 run ~840 个 part 帧
  ≈ 144 GB；全粒子逐步回放需 ~1e10 粒子步 ×（ADAS + 场插值 + 4e3 节点沉积）。
* 替代方案（已采用）：**降阶 marker 链**（ADR-0020 决定 2），
  `part*.h5` 每 10 步快照用于标定输运系数；NN 学有界残差。
* 需要用户批准的规划变更点：无（roadmap 目标"自生成源/无未来量/物理性优先"
  不被削弱；B1 结构按 ADR-0020 决定 2 执行）。

**H2（登记而非阻断）：内部 `inj_source` 未由生成侧导出。**

* 现状：解析 shard 密度源只能按 `inj_source` 语义自算 + 与 marker feedback
  交叉校验；B1.1 的逐 shard 验收（1.0021）不依赖它。
* 替代方案：① 生成侧按导出规范 §3 增导 `inj_source`（首选）；
  ② 继续以 marker feedback 为 marker run 真源（已采用，ADR-0020 决定 1）。
* 需要用户批准：无（不改变规划目标；若生成侧增导，B1.1 监督可直接升级）。

## 5. 验证命令

```bash
.torch-cuda-env/bin/python -m pytest -q
# 历史记录（本次持久状态实现前）：276 passed, 1 skipped, 2 warnings

.torch-cuda-env/bin/python tools/check_export_v3.py \
    --run-dir data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner \
    --zero-dt-policy registered --quiet --no-budget
# 登记事件通过（2704 等 6 个事件）；cross-boundary/lagging 跳过并登记
```

流程记录：`gh` CLI 在本环境缺失（`which gh` → not found），未认领/关闭 issue；
未使用 `git`（`.git` 只读）；未改 `tests/legacy/**`；未删除任何数据。

### 持久 marker 状态实现后的验收（2026-10-07）

- `.torch-cuda-env/bin/python -m pytest -q`：**290 passed, 1 skipped, 2 warnings / 188.46 s**。
- 真实 phaseA v400 / 4110 完整检查点读取：1,000,000 槽、11,168 活 marker，内部错步时间、P_imp 与 `v→V_prev` 校验通过。
- injInner 整个 run 的 registered restart 验收 CLI（`--quiet --no-budget`）：exit 0。
- 独立公式、两窗口序列化/恢复逐位一致、停止注入后的云延续、启动零反馈及时间/缺失初态失败测试见[实现报告](marker-persistent-state-implementation-2026-10-07.md)。这些是组件验证，未进行正式训练或完整自生成场闭环验收。
