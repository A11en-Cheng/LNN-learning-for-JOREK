# 0015 marker→投影神经代理升级（GNN + 更长历史）

Blocked by: #0010, #0012

交付：在 #0012 节点局部 MLP 预研基础上，升级为 GNN（复用九场网格拓扑，捕获杂质输运/空间上下文），
并对 `dE_ion/rad_sink` 引入更长历史窗口（如 5-10 帧的九场状态与已得投影）作为输入；
训练/验证沿用 manifest 的 train/val split 与 train-only 源项统计。

验收：逐场 R²/RMSE 报告；GNN 在 Z_eff/Z_imp/N_imp 上不劣于节点 MLP、在 dE_ion/rad_sink 上显著提升；
登记单帧 CPU/GPU 成本与相对 marker 模块的墙钟加速；结论登记 provenance。

目标：电荷/密度三场稳定 >0.6，能量/辐射两场明显高于 0.25 并趋近 0.6；
若达成，正式立项"神经 marker 模块"方向 3。

状态：in-progress（v1 欠拟合）。`tests/run_source_surrogate_gnn.py`：GAT 2 层 + 过去投影历史
（特征=9 状态+5 过去投影）。首轮（12 帧、80 epochs）val R²：rad_sink 0.59（历史显著帮助辐射场）、
dE_ion 0.26、Z_imp 0.31、N_imp 0.34、Z_eff 0.12——电荷力矩反低于节点 MLP（0.55-0.68），
判断为欠训练/批处理方式不当。下一步：加 epoch、更大的 batch/容量，并以节点 MLP 为残差旁路再比。
