# 0014 ADAS coronal 闭包原型 + 与记录投影的逐场校验

Blocked by: #0002

交付：实现确定性物理闭包——用预测/记录的九场原生状态 `(rho_imp, n_e, T_e)` 经 ADAS
SCD/ACD/coronal 表（非平衡假设外）重建源项力矩 `Z_eff, Z_imp`（可选 `dE_ion, rad_sink`），
按帧输出 `[nodes, 5]`；并用记录 `projections*.h5` 做 ground truth，逐场报告重建误差（R²/RMSE），
与 #0001 的统计协议共用 split。

验收：闭包单帧 CPU 成本 ≤ 模型单步的 2%（实测基准：模型 61 ms、闭包 naive 1.2 ms）；
逐场校验报告落盘；`Z_eff/Z_imp` 重建质量显著高于线性基线（R²>0.6 且 >线性 R²）；
成本结论与精度结论登记进 provenance。注意：dE_ion/rad_sink 历史依赖强，允许作为外生控制不重建。

