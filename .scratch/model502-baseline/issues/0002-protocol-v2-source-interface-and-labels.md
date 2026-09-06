# 0002 协议 v2：源项接口与原生/导出标签

Blocked by: #0001

交付：把 [协议文档](../../../report/model502-schema-and-source-protocol.md) 升级为 v2：以 marker_spi_source
真实输出为准（`part*.h5` markers + `projections*.h5` 五场 `[dE_ion, rad_sink, Z_eff, Z_imp, N_imp]`），
model502 背景源为可选第二通道；为每个字段标注"原生/导出/控制"，并写清单位与时间对齐。

验收：源项 schema 测试通过；协议 v2 与 `src/data/model502_schema.py` 一致；网络学习目标只保留原生九场。

