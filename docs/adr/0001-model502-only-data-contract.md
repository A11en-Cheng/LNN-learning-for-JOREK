# ADR-0001：Model 502 九场状态与源项契约是唯一数据契约

状态：已接受（2026-09-06）

## 背景

审计报告 [report/hyphy-glnn-audit-roadmap.md](../report/hyphy-glnn-audit-roadmap.md) 指出：parser 输出 9 场、
encoder/decoder 断言 5 场的契约分裂阻断了默认 forward 和 rollout；默认数据目录还是 8 通道的 BEST/Model 600，
与 Model 502 不兼容。d5a44f0 引入了 9 场 schema 与源项协议
（[report/model502-schema-and-source-protocol.md](../report/model502-schema-and-source-protocol.md)），
并已将 encoder/decoder/loss/benchmark 迁到 9 场。

## 决定

1. Model 502 的九场状态契约是唯一数据契约，字段顺序、变换、统计量协议以
   `src/data/model502_schema.py` 和协议文档为准。
2. 数据集将完全采用 Model 502 及其配套 `marker_spi_source` 产生的 HDF5；BEST/Model 600 数据从训练集中移除，
   不实现双 adapter。如需恢复，必须另立 ADR 重新论证。
3. 源项按 all-or-nothing 契约处理：`S_rho/S_T_i/S_T_e` 缺任一字段即该帧不是可训练的强制样本；
   禁止用零填充冒充源项。
4. 五场 checkpoint 与新契约不兼容，必须显式拒绝加载，禁止静默迁移或截断。

## 后果

- 所有 parser/model/loss/benchmark 只能引用 schema 模块中的常量，禁止模块私有字段索引。
- 任何新数据必须先通过 schema/source 契约测试和完整性检查。
- 路线图中源项相关任务被 source 数据交付清单（协议 §4）阻塞，不排进近期 ticket。

