# ADR-0005：源项规范 v2 与原生/导出数据分层

> **文档隔离（2026-10-04）**：v1 路线历史文件。新架构工作只以 ADR-0018/0019 + roadmap_v2 为准；引用须标注"（历史）"，冲突以 0018/0019 为准（索引见 README.md）。

状态：已接受（2026-09-06）；2026-09-24 由 ADR-0011 补充：辅助闭包模块
（源项映射子网络、碎片动力学）允许以导出量为监督目标，主状态模型仍只学原生九场。

## 背景

协议 v1 假设源项为 `S_rho/S_T_i/S_T_e` 三场；实际 marker_spi_source 的输出是 `part*.h5` 原始 markers
与 `projections*.h5` 五场投影 `[dE_ion, rad_sink, Z_eff, Z_imp, N_imp]`（use_marker 模式），
而 model502 `mod_sources` 另有一组背景源 `(particle_source, heat_i_source, heat_e_source)`。
用户在 grilling 中确认：用真实输出建 v2；神经网络的学​​习重点是原生数据，不是导出量。

## 决定

1. 协议 v2 以 marker_spi_source 的真实输出为准：`part*.h5` 是源项主记录，`projections*.h5` 五场是
   源项在场上的投影，model502 背景源作为可选的第二控制通道；不再要求把一切统一映射为 `S_rho/S_T_i/S_T_e`。
2. 数据按"原生 / 导出"分层：原生量 = model502 的 9 个状态场
   （`psi, u(Φ), zj, w(ω), rho, Ti, Vpar, rhoimp, Te`，见 `mod_model_settings.f90` 的 `n_var=9`）。
   导出量 = 源项投影、粒子统计、诊断时间序列、输运闭包（`eta_T, ZK_i/e_par_T, D_prof, dcoef` 等）。
   神经网络的学习目标永远是原生 9 场；导出量只能作为控制输入或诊断，禁止作为学习目标。
3. 高阶截断项（hyper-*：`eta_num, visco_num, ZK_par_num` 等）由网络隐式吸收，不作为推理输入，
   也不作为参数辨识目标。

## 后果

- 协议 v2 必须给每个数据字段标注"原生 / 导出 / 控制"三类标签；模型输出只承诺 9 个原生场。
- 条件通道只接受零维参数与源项控制，不接受派生场作为输入。

## 时序与因果（2026-09-10 补充）

marker 投影与流体场的耦合是错步显式（staggered explicit）：投影 P_{k+1} 用旧场
F_k 算出、驱动产生 F_{k+1}，时间戳落在输出帧。粗采样下 `(x_i, P_i) → x_{i+1}`
因果正确（P_i ≈ f(x_i)）；已知近似为窗口内常数源假设与 marker 隐藏电荷态。
源码级推导与行号见
[report/marker-projection-causality.md](../report/marker-projection-causality.md)。
