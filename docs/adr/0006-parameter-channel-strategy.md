# ADR-0006：参数通道策略（导出为主、两帧辨识只作校验）

> **文档隔离（2026-10-04）**：v1 路线历史文件。新架构工作只以 ADR-0018/0019 + roadmap_v2 为准；引用须标注"（历史）"，冲突以 0018/0019 为准（索引见 README.md）。

状态：已接受（2026-09-06）

## 背景

装配 model502 九场方程需要一批零维参数；`jorek*.h5` 已含 `eta/visco/visco_par/tstep/central_density/central_mass`，
缺失 `ZK_i_par/ZK_e_par`、`D_perp` 基准值等。用户询问能否从相邻两帧在 CPU 上"完全推导"缺失参数。
结论：精确推导不可行（单对快照欠定、输出帧间是多步自适应子步的复合传播、hyper-* 高频项在输出采样率下不可辨识），
但轨迹约束最小二乘辨识可行，可作交叉校验。

## 决定

1. 主路径：在 `mod_export_restart.f90` 补导出缺失的零维基底标量（`ZK_i_par/ZK_e_par`、`D_perp` 基准等），
   每个标量即一行 `HDF5_real_saving`；不为此重新收集数据。
2. 兜底校验：用相邻两帧（或短轨迹）在 CPU 上做约束最小二乘辨识
   `min ||s_{t+1} − s_t − Δt·R(s;θ)||²`，仅作交叉校验，不作为精确推导来源。前提约束：regressor 用与 JOREK
   相同的子步/θ-scheme 构造；对强相关项（如 `j` 与 `∇²u`）加正则化；T/Zeff 依赖形状需函数族假设。
3. hyper-* 截断项（`eta_num/visco_num/ZK_par_num` 等）不导出、不辨识，由网络隐式吸收（ADR-0005）。

## 后果

- Phase 0.3 的 JOREK 侧导出改动是外部依赖，独立成 `ready-for-human` ticket。
- 辨识脚本作为校验工具排在参数通道 ticket 之后，不进入训练主路径。
