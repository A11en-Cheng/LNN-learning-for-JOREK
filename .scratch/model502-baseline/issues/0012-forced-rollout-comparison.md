# 0012 含源 rollout 对比

Blocked by: #0010, #0011, #0014

交付：在 marker_spi_source 对齐后的二维含源数据上训练并做 rollout；与无源 baseline 同 seed/split 对比
误差、稳定性、显存、吞吐。rollout 控制模式报三组：真值控制（上界）、冻结最后已知控制（诚实自主估计）、
ADAS coronal 闭包控制（#0014），并分解三者差距。

注：#0014 校验为负（热猝灭相非日冕平衡，闭包 R²≈0），闭包控制模式暂缓，先报真值/冻结两组；
闭包作为平衡相可选，不阻塞本票。

验收：强源项下无 NaN；相对无源 baseline 的改善可解释并登记 provenance。
