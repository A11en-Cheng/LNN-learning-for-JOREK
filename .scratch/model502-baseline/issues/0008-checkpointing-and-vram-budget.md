# 0008 梯度检查点与显存预算

Blocked by: #0006

交付：encoder/decoder message passing 封装为可 checkpoint 块；GPU 基准命令记录峰值显存
（含当时的并发占用，`nvidia-smi` 留证）。

验收：checkpoint 与非 checkpoint 梯度最大绝对差 ≤1e-5；峰值显存低于预算（RTX 5070 Ti 16GB，按并发打折）。

