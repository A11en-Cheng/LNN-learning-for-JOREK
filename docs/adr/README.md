# ADR 索引与文档隔离（2026-10-04）

本目录保持单一 ADR 序列，但自 2026-10-04 起存在两条路线，阅读时必须隔离：

## 现行（新架构，权威）

| ADR | 主题 |
|---|---|
| [0018](0018-architecture-first-and-physics-structured-build.md) | 架构优先：自生成源 + stencil-conv 九场 + 物理结构化验收；撤销教师源 gate、GNN/谱、全集 concat、"无 condition"对照臂 |
| [0019](0019-recorded-output-timing-contract.md) | 记录输出时序契约：`S_n` 驱动 `u_n→u_{n+1}`；`part{n}` 在 `T_{n−1}`；旧管线 `source[:, k+1]` 属一步未来泄漏 |
| [0020](0020-marker-source-contract.md) | marker 源对象契约（双源对象；marker feedback 为 marker run 真源）+ 降阶 marker 链 + 增广状态 `(u, p_shard, h_marker)` + 共享 S 接口 + 登记 restart 策略；覆盖 roadmap_v2 的冲突表述 |
| [0021](0021-native-state-cfc-window-curriculum.md) | 完整原生残差推进、同场值/梯度、分组参数路由、CfC及可替换时间核心、因果窗口与TF→AR课程；定向修订0018参数限制及0020的roadmap冻结规则 |
| [0022](0022-source-conditioned-joint-native-advancement.md) | 含源联合学习主路径；冻结容量仅为互斥参考模式；补充0021，不改0020事件与源对象 |
| [0023](0023-explicit-marker-local-subcycling.md) | 非默认 marker 局部细分近似、独立输运历史与互斥检查点；v2 明示外边界末段驻留近似，不改源公式/base 耦合顺序 |

**新架构工作（roadmap_v2 Phase A–E）的阅读顺序**：
`roadmap_v2.md` → `docs/adr/README.md`（本文件）→ ADR-0018/0019/0020/0021/0022 →
`report/model502-schema-and-source-protocol.md` §3.1a →
`report/marker-projection-causality.md`（修正版）→
`report/conformance-roadmap-v2-2026-10-07.md`（roadmap 条目 ↔ 实现/证据状态）。
roadmap_v2 与 ADR-0020 冲突的表述（源对象、B1 结构、Phase C 循环、restart 策略、
共享 S 接口）以 ADR-0020 为准。
时间核心、值/梯度、参数路由及课程按 ADR-0021 的定向修订；它不更改 ADR-0020 的物理流程。含源主路径按 ADR-0022；闭合特征不等于显式物理算子。
marker 输运数值模式按 ADR-0023：原生固定子步为默认，局部细分必须显式选择，不能把数值近似称作 JOREK 固定步长复刻。

规则：

1. 新工作的规则、配对、验收以 ADR-0018/0019/0020/0021/0022 与 roadmap_v2 为准；
   按上述分域覆盖规则处理冲突，旧 ADR 不作为依据。
2. 引用旧 ADR 必须标注"（历史）"，且不得把旧 ADR 的量化 gate、教师源训练语义、
   LOO/评估协议当作新工作的前置或验收。
3. 任何冲突以现行 ADR 的定向修订为准；发现旧文档污染入口时，登记到
   `report/v2-doc-isolation-and-dirty-data-audit-2026-10-04.md`。

## 历史（v1 插值器路线；背景与证据）

| ADR | 主题 | 现行相关性 |
|---|---|---|
| 0001 | 九场状态与源项契约 | 契约对象仍有效；时序/配对以 ADR-0019 + schema §3.1a 为准 |
| 0002 | 路线图单一来源与分阶段验收门槛 | 量化 gate 语义被 ADR-0018 修订 |
| 0003 | `.torch-cuda-env` 运行环境（CUDA 13 / torch cu130） | 仍有效（基础设施事实） |
| 0004 | EHL-2 数据选择与两阶段训练顺序 | 数据选择部分仍有效；顺序被 roadmap_v2 取代 |
| 0005 | 源项规范 v2 与原生/导出分层 | 分层事实仍有效；训练语义以 0018/0019 为准 |
| 0006 / 0007 | 参数通道策略 / namelist 默认值 | 取值链仍有效；注入方式以 ADR-0018（物理子集 + FiLM）为准 |
| 0010 | Physics Loss 只在插值点 | v1 训练语义；物理损失暂停（ADR-0017/0018） |
| 0011 | 场-碎片-源耦合架构 | v1 设计；现行以 roadmap_v2 Phase B/C |
| 0012 | blocked 划分 + LOO | 可选泛化评估；不作前置或 model selection（ADR-0018） |
| 0013 | 子区间积分残差 | v1 物理损失形式；暂停 |
| 0014 | 隐式 CfC + 粒子矩 + 课程式 teacher forcing | v1 架构；未采用 |
| 0015 | 解析骨干交付、学习修正暂停 | 解析骨干事实仍有效；修正启用条件已并入 ADR-0018 第 8 条 |
| 0016 | 取消反组装 gate；Eulerian 矩降级 | （历史）对象差异事实仍有效；"projections 只作监督"的结论被 ADR-0020 决定 1 覆盖（marker run 里它就是真源） |
| 0017 | 物理先做独立预测器验证 | 物理暂停原则仍有效；不得把残差当训练损失 |

其他 v1 证据/历史文档（引用时须标注"（历史）"并先读其顶部勘误）：
`new_version.md`、`report/roadmap-review-2026-09-29.md`、
`report/stage-summary-2026-10-03.md`、`report/ab-condition-2026-10-03.md`、
`report/physics-residual-correction.md`、`report/spi-analytic-baseline.md`、
`report/roadmap-v2-readiness-2026-10-03.md`。
