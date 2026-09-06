# 0007 单样本过拟合 gate

Blocked by: #0006

交付：固定 seed，单个小图/短序列（`SEQ_LEN=3` 起步）训练至 loss 下降 ≥99%；产物含 loss 曲线、
checkpoint、seed/config manifest；reload 后曲线可复现。

验收：训练 loss 下降 ≥99% 且无 NaN；reload 输出一致 ≤1e-6。失败即冻结 Phase 2/3（ADR-0002）。

