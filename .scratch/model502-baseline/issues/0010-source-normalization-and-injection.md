# 0010 源项归一化与注入

Blocked by: #0002, #0009

交付：源项按训练 split 统计归一化；`require_source=True` 的 Dataset 路径；把已存在的 `control_encoder`
接入 processor 与 trainer；无源帧拒绝零填充。

验收：源项 round-trip 测试；强源项样本前向/反向无 NaN；与无源基线共用同一状态 seam。

