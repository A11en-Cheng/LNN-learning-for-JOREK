# 0011 条件注入与消融

Blocked by: #0005

注：#0009 的 gate 因跨场景平均 FAIL，本票提前执行（不再阻塞于 #0009）。

交付：零维参数 condition encoder（最小 MLP）接入，注入点单一定义；消融实验：固定状态改变条件时
输出/latent 有可检测响应，且 no-condition 对照成立。

验收：条件响应测试 + ablation 表；响应幅度与噪声水平可区分。

状态：done（代码与消融）。`JorekGNNEncoder(condition_dim=44)` 注入点 = encoder 输入
（condition MLP → 逐节点广播加在节点特征上，`condition=None` 即无条件的对照路径，与无 MLP 编码器逐位一致）。
条件通道扩展 7 个场景判别参数（n_spi、pellet_density、gamma_i/e_stangeby、particlesource、ellip、
bc_natural_open，默认值取自 preset_parameters.f90），manifest 产出 44 维 train-only 条件统计。
消融表（z-score 后、固定状态、未训练 encoder，mean|Δlatent|）：eta+1σ=0.0241、
ZK_i_par+1σ=0.0084、n_spi+1σ=0.0124、换 run 条件=0.0404；同输入噪声底=0，响应可区分。
60 tests 全绿。gate 重试（#0009）：同场景族训练 + 条件注入，下一轮执行。
