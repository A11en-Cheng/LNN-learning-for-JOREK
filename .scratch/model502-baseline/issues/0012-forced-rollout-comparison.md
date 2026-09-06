# 0012 含源 rollout 对比

Blocked by: #0010, #0011

交付：在 marker_spi_source 对齐后的二维含源数据上训练并做 rollout；与无源 baseline 同 seed/split 对比
误差、稳定性、显存、吞吐。

验收：强源项下无 NaN；相对无源 baseline 的改善可解释并登记 provenance。

