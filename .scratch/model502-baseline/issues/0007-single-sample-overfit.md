# 0007 单样本过拟合 gate

Blocked by: #0006

交付：固定 seed，单个小图/短序列（`SEQ_LEN=3` 起步）训练至 loss 下降 ≥99%；产物含 loss 曲线、
checkpoint、seed/config manifest；reload 后曲线可复现。

验收：训练 loss 下降 ≥99% 且无 NaN；reload 输出一致 ≤1e-6。失败即冻结 Phase 2/3（ADR-0002）。

状态：done。`tests/train_single_sample_overfit.py`（可复现 CLI，gate 内嵌）+ 快速 harness 测试。
真实数据（scan_1d22 帧 2710-2730，GPU，seed=0，1500 epochs）：loss 0.2534 → 1.78e-5
（下降 99.993%），无 NaN，reload 最大绝对差 4.14e-7 ≤1e-6，dt=4.41e-6 s。产物在
output/model502_baseline/single_sample_overfit/（checkpoint/loss_curve/manifest）。
