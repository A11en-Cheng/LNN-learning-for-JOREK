# 0014 ADAS coronal 闭包原型 + 与记录投影的逐场校验

Blocked by: #0002

交付：实现确定性物理闭包——用预测/记录的九场原生状态 `(rho_imp, n_e, T_e)` 经 ADAS
SCD/ACD/coronal 表（非平衡假设外）重建源项力矩 `Z_eff, Z_imp`（可选 `dE_ion, rad_sink`），
按帧输出 `[nodes, 5]`；并用记录 `projections*.h5` 做 ground truth，逐场报告重建误差（R²/RMSE），
与 #0001 的统计协议共用 split。

验收：闭包单帧 CPU 成本 ≤ 模型单步的 2%（实测基准：模型 61 ms、闭包 naive 1.2 ms）；
逐场校验报告落盘；`Z_eff/Z_imp` 重建质量显著高于线性基线（R²>0.6 且 >线性 R²）；
成本结论与精度结论登记进 provenance。注意：dE_ion/rad_sink 历史依赖强，允许作为外生控制不重建。

状态：done（负结果已记录）。`src/physics/adas_closure.py`：ADF11 解析（scd96/acd96 Neon，温度轴为
log10 eV）、日冕三对角求解、Te 轴标定（least_squares），单测通过（解析形状、温度单调、有限性）。
用密度加权矩比率 `Z_imp=field3/field4`、`Z_eff=field2/field4` 校验：验证样本 R²≈0（闭包与线性基线皆然），
结论=热猝灭阶段不满足日冕平衡，瞬时状态闭包不能重建源项力矩（marker 记忆/非平衡不可忽略）。
成本 3.5ms/帧 = 模型步 5.8%（naive；预计算矩网格可压到 2% 内，但负精度结果下暂不优化）。
`tests/run_adas_closure_validation.py` 产出 report.json。#0012 的"闭包控制"模式据此改判：暂用真值/冻结，
闭包留作慢相或平衡相的可选项。

2026-09-07 追加（按用户指示改用 JOREK 同思路）：新增 `src/physics/charge_state_closure.py` 时间推进
电荷态闭包（显式欧拉 SCD/ACD + 逐态 Lz + max(P,0)，与 marker_spi_source 的 CE_marker=.false. 分支一致，
ADAS 轴换算 cm⁻³→m⁻³、eV→K）。单测通过（归一化、有界、有限），但 val 序列校验同样 R²≈0
（`tests/run_charge_state_validation.py`）。结论：源项力矩受 marker 输运/烧蚀历史主导，
节点局部的电荷态时间积分也无法重建；含源推演必须把投影作为外部控制（真值/冻结），闭包路线关闭。
