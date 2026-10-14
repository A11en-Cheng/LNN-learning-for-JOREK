# ADR-0002：路线图单一来源与分阶段验收门槛

> **文档隔离（2026-10-04）**：v1 路线历史文件。新架构工作只以 ADR-0018/0019 + roadmap_v2 为准；引用须标注"（历史）"，冲突以 0018/0019 为准（索引见 README.md）。

状态：已接受（2026-09-06）

## 背景

原 [new_version.md](../new_version.md) 按月份承诺 Phase 1–4，未设 gate；审计报告 §8–§11 判定其与现状脱节，
并给出修订后的 gated 任务结构。用户在 grilling 中确认：以审计修订版重写 new_version.md 作为唯一路线图来源，
并采纳其量化验收标准作为进入 MoE 的硬门槛。

## 决定

1. `new_version.md` 是路线图唯一来源，按审计 §9 的 gated 结构重写；审计报告保留为证据附录，不再并行维护路线图。
2. Phase 1 的 gate（全部通过后才允许任何 MoE 工作）：
   - schema/source/迁移 smoke tests 全绿；
   - split manifest 落地，统计量仅由训练 split 计算；
   - 固定种子单样本过拟合：训练 loss 下降 ≥99%，无 NaN；
   - checkpoint reload 输出一致 ≤1e-6；
   - 至少一个 free-run horizon 优于 persistence baseline，指标无 NaN；
   - checkpointing 梯度等价最大绝对差 ≤1e-5，且在约定 GPU/批量下峰值显存低于预算。
3. MoE 被 gate 阻塞；最小握手（concat/add MLP）先于 Cross-Attention；gradient firewall 消融先于在 MoE 上启用物理 loss。
4. 源项、3D 频谱、JOREK runtime 替代等条目是带数据/单位/验收阈值的研究假设，不写时间承诺；数据或单位无法确认就停止对应任务。

## 后果

- 每张 ticket 必须声明 blocking edges；gate 失败即冻结下游扩展，不得以“调参再试试”绕过。
- 任何报告的数字必须附 provenance（checkpoint、config、seed、数据版本）。
