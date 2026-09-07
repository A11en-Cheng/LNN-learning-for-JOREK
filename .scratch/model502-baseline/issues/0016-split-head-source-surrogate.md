# 0016 分头源项代理

Blocked by: #0012, #0015

交付：把源项代理拆成两个头——瞬时头（9 场状态 → Z_eff/Z_imp/N_imp，节点局部 MLP）与
历史头（9 场状态 + 3 帧过去投影 → dE_ion/rad_sink，节点局部 MLP）；沿用 train/val split 与
train-only 源项统计。

验收：五场整体 R² ≥0.5，电荷三场 ≥0.6，能量/辐射两场显著高于瞬时基线（0.21/0.25）；
逐场报告 + provenance。

状态：done（验收全过）。`tests/run_source_surrogate_split.py`：瞬时头（9 状态→Z_eff/Z_imp/N_imp）
+ 历史头（9 状态 + 3 帧过去投影→dE_ion/rad_sink）。val R²：Z_eff 0.705、Z_imp 0.714、N_imp 0.607、
dE_ion 0.808、rad_sink 0.864——五场全 ≥0.6，能量/辐射两场大幅超越瞬时/GNN 基线。
方向 3（神经 marker 模块）得到强验证；下一步量化相对 marker 模块的墙钟加速。

较长训练（`tests/train_source_surrogate_long.py`，全 train split、12 万样本、400 epochs、hidden 192）：
val R² Z_eff 0.613、Z_imp 0.637、N_imp 0.651、dE_ion 0.952、rad_sink 0.955；
loss 曲线与报告在 output/model502_baseline/source_surrogate_long/。
