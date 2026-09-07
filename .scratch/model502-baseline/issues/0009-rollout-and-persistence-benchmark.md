# 0009 free-run rollout 与 persistence 基准

Blocked by: #0007

交付：held-out 序列上的逐场误差、rollout 曲线、质量漂移、PDE residual；报告附 checkpoint/config/seed/数据版本。

验收：至少一个 free-run horizon 优于 persistence；指标无 NaN；质量漂移阈值预先登记。此表是 MoE 的对照基线。

状态：harness done，gate FAIL（已记录，Phase 2/3 冻结）。已交付 harness：`tests/train_baseline.py`
（train-split、manifest 统计归一化、单调段校验）+ `tests/run_baseline_benchmark.py`（warmup 状态、
逐场 RMSE、persistence 对照、R 加权 rho 质量漂移、预登记阈值 0.05）。

v0（400 步、2 run）与 v1（1000 步、6 run）在两个 held-out run 上均全部 horizon 劣于 persistence
（逐场差 15-25%），质量漂移分别 -15% / +24%，均超阈值；无 NaN。训练 loss 正常下降（0.0196→0.0018），
说明不是容量问题。根因诊断：场景条件未接入（跨场景平均）+ 4.4µs 步长下 persistence 极强（帧自相关
0.97-0.99）。决议：#0011 条件注入提前为 gate 重试前置；重试时同场景族训练 + 更长步数。
