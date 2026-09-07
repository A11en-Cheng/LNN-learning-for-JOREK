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

2026-09-07 更新（#0011 完成后重试）：v2（1000 步、6 run、44 维条件注入）仍 FAIL：两个 held-out run
逐场 RMSE 比 persistence 差 15-25%，漂移累至 0.5。一步诊断：真实 dt=4.09e-7 s 下预测 |Δ| 是真实单步
增量的 56×——模型在平静段帧上输出剧烈段量级的 delta。根因更新为：状态依赖性未充分学到 +
训练窗口相位分布偏斜（剧烈段主导）。下一步候选：#0006 的 delta 语义复核（decoder 输出改为
rate×dt 以强制尺度一致性，并补 CfC timespan 缩放单测）+ 训练窗口按相位重采样后再重试。

2026-09-07 再更新（rate+dt_ref 契约）：残差改为 rate，更新规则 `x+rate·dt/dt_ref`（dt_ref=训练帧步均值，
随 checkpoint 保存）；修复了 dt≈4e-6 导致的梯度消失。v4（1000 步、6 run）短 horizon（10 步）漂移回到
预登记阈值内（0.0002 vs 0.5），但逐场 RMSE 仍比 persistence 高 3-39%，无改善 horizon；100 步 horizon
漂移重新累积至 ~9。结论：幅度偏置已修，剩余是"学出状态依赖单步动力学"的能力问题。待决杠杆：
更长训练+LR 调度、gate 专用 horizon-1 训练、逐场损失权重、更长时序上下文、窗口相位重采样。
gate 维持 FAIL，Phase 2/3 冻结。#0007 过拟合 gate 在新契约下重跑 PASS（99.998%、reload 5.1e-7）。

2026-09-07 末次更新（JOREK 单位制）：模型内部时间切到 JOREK 单位（dt=t_now 差，梯度天然 O(1)），
物理秒=dt·t_norm 仅用于投影对齐/PDE/报告；归一化关系已按源码写入 definitions.md §8。
v5（1000 步、6 run）逐场比率 1.26-1.35、漂移超阈值，仍 FAIL。与 v4（秒+dt_ref）同档，
证明单位约定非堵点；保留 JOREK 单位制作为最终约定。结论：1000 步/6 run 的基线学不出
状态依赖单步动力学；待决杠杆不变（更长训练+LR 调度、horizon-1、逐场权重、相位均衡采样）。
