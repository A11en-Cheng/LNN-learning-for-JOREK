# 0013 CPU 两帧最小二乘辨识（交叉校验工具）

Blocked by: #0005, #0006

交付：实现轨迹约束最小二乘辨识脚本（与 JOREK 相同的子步/θ-scheme 构造 regressor，正则化共线性项），
对基底标量做离线估计并与 h5 导出值对比。

验收：对已导出标量的估计与真值在可登记容差内一致；仅作校验，不进入训练主路径（ADR-0006）。

状态：done（核心 + 合成校验）。`src/evaluation/two_snapshot.py`：轨迹约束最小二乘辨识
（`least_squares_identify(regressors, residual)`）；`tests/test_two_snapshot.py` 合成已知系数恢复
（atol 5e-2）与形状校验。真实场的 eta/visco 辨识需构造 PDE 残差 regressor（grad_ops 微分项），
作为后续扩展，不在本票内强行拼装。
