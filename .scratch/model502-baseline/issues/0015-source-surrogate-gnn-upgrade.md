# 0015 marker→投影神经代理升级（GNN + 更长历史）

Blocked by: #0010, #0012

交付：在 #0012 节点局部 MLP 预研基础上，升级为 GNN（复用九场网格拓扑，捕获杂质输运/空间上下文），
并对 `dE_ion/rad_sink` 引入更长历史窗口（如 5-10 帧的九场状态与已得投影）作为输入；
训练/验证沿用 manifest 的 train/val split 与 train-only 源项统计。

验收：逐场 R²/RMSE 报告；GNN 在 Z_eff/Z_imp/N_imp 上不劣于节点 MLP、在 dE_ion/rad_sink 上显著提升；
登记单帧 CPU/GPU 成本与相对 marker 模块的墙钟加速；结论登记 provenance。

目标：电荷/密度三场稳定 >0.6，能量/辐射两场明显高于 0.25 并趋近 0.6；
若达成，正式立项"神经 marker 模块"方向 3。
