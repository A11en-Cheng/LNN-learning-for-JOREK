# 0010 源项归一化与注入

Blocked by: #0002, #0009

交付：源项按训练 split 统计归一化；`require_source=True` 的 Dataset 路径；把已存在的 `control_encoder`
接入 processor 与 trainer；无源帧拒绝零填充。

验收：源项 round-trip 测试；强源项样本前向/反向无 NaN；与无源基线共用同一状态 seam。

状态：done。processor 的控制通道改为五场投影 `[dE_ion, rad_sink, Z_eff, Z_imp, N_imp]`
（`control_encoder` 5→latent，`validate_projection_tensor`）；trainer 的 train_step/eval_step/multihorizon
默认把 `batch.source` 作为 controls 传入（显式 controls 优先），无源帧 controls=None 不零填充。
manifest 新增 `source_stats`（train split、1225 帧、5.05M 节点样本）；训练/基准脚本按该统计做
线性 z-score 并随 checkpoint 保存。测试：五场注入响应、三场拒绝、trainer 默认接线、round-trip、
真实强源帧 forward/backward 无 NaN。70 tests 全绿。
