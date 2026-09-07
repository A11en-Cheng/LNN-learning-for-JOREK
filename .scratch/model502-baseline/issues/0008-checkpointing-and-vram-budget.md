# 0008 梯度检查点与显存预算

Blocked by: #0006

交付：encoder/decoder message passing 封装为可 checkpoint 块；GPU 基准命令记录峰值显存
（含当时的并发占用，`nvidia-smi` 留证）。

验收：checkpoint 与非 checkpoint 梯度最大绝对差 ≤1e-5；峰值显存低于预算（RTX 5070 Ti 16GB，按并发打折）。

状态：done。encoder/decoder 每个 message-passing 块支持 `checkpoint=`（训练模式生效，eval 旁路）；
`tests/test_checkpointing.py` 验证开/关前向一致、梯度差 ≤1e-5、eval 语义一致。
GPU 基准（batch=2、hidden=64、4 层、3×fwd+bwd）：峰值显存 2742 MiB → 524 MiB（约 6× 节省），
并发占用 1.6GB 时总量远低于 16GB 预算；报告在 output/model502_baseline/checkpoint_vram/report.json。
config.yaml `model.checkpoint: true` 默认开启。57 tests 全绿。
