# ADR-0019：记录输出的时序契约（2026-10-04 源码 + 数据核实）

状态：已接受（2026-10-04）；修正 CONTEXT 的 target-frame 约定、`report/marker-projection-causality.md`
的结论与旧训练管线的源配对。

## 背景

用户要求以 JOREK 代码为基准，确定 `part*.h5` / `projections*.h5` 是"当前步"还是"下一步"的量——
这直接决定数据集配对与新架构的推理顺序。核实材料：

- 源码：`particles/examples/marker_spi_source.f90`（粒子主循环）、
  `particles/diagnostics/mod_project_particles.f90`（投影写出）、
  `core/mod_jorek_timestepping.f90`（九场求解与 `jorek*.h5` 写出）、
  `particles/mod_particle_io.f90`（`part*.h5` 写出，含 `/time`）；
- 生产日志：`data/EHL-2/scan_1d22_ang12_massloss_neon/logfile_spitest`（非 restart 段的 7360 窗口）；
- 全量数据：3 个 run（`scan_1d22_ang12_massloss_neon` 620 个匹配索引、
  `scan_2d21_ang15_5_massloss_neon_v300`、`scan_1d22_ang15_5_massloss_neon_v400`）。

## 核实结果（事实）

设 `T_n` = 第 n 个 JOREK 流体步的时间（`macroscopic_vars.dat` 第 n 行），`dt_n = T_n − T_{n−1}`：

1. **`jorek{n}.h5` = 九场 `u(T_n)`**。`t_now` 与 `macroscopic_vars.dat` 第 n 步时间一致
   （620/620，误差为数值舍入）。
2. **`projections{n}.h5` = 采样窗口 `[T_{n−1}, T_n]` 的时间平均**，采样时粒子在
   `u_{n−1} → u_n` 的插值场中推进；写出发生在第 n 步完成之后、第 n+1 步开始之前
   （日志顺序：`After step 007360` → 粒子窗口 `sim%time=T_7359, target_time=T_7360` →
   `Written projection to projections007360.h5` → `After step 007361`）。其内部
   `t_now = T_n`，并被**第 n+1 步的矩阵求解消耗**。
   ⇒ **`S_n` 驱动 `u_n → u_{n+1}`**（同号配对：源与它作用的当前场同号、同时刻）。
3. **`part{n}.h5` = `T_{n−1}` 的粒子状态**（同号场帧早一个 JOREK 步）。全量统计
   `time/t_norm = T_n − dt_n`（620/620）；日志中该窗口起点的粒子权重
   （`0.16093158E+22`）与 `part07360.h5` 的权重总和（`1.609298E+21`）一致。
   ⇒ `part{n}` 是 `S_n` 采样窗口的**起始**粒子状态 `p_{n−1}`。
4. **restart 边界的 projections 偏移**：`scan_1d22_ang12_massloss_neon` 的
   3550 / 4150 / 4950 / 5330 / 6600 / 7350（均为 `available_restart_files.txt` 中的
   检查点）上，`projections{k}` 相对同号 `jorek` 早一个 `dt_source`（即偏移恰为
   `−dt_source`）；`part{k}` 始终满足 `time = T_k − dt_source`（常规规则，与 restart
   无关）。v400 的 5050 同类。处理规则：这些 projections 按决定 4 剔除/重对齐，
   `part` 无需额外处理。
5. 仓库 example 的事件数组把投影事件排在 JOREK stepper 之前，而生产数据的日志顺序相反；
   仅凭 example 的顺序推断数据语义会得出错误结论。新数据必须以实际驱动顺序 + 显式元数据为准。

## 决定

1. **时序契约（唯一配对规则）**：
   `(u_n, p_{n−1}, S_n) → u_{n+1}`，其中 `p_{n−1} = part{n}`（窗口起点粒子状态），
   `S_n = projections{n}`；`S_n` 的采样窗口是 `[T_{n−1}, T_n]`。
   碎片演化与源在同一窗口内产生：`p_n = FragEvolve(p_{n−1}, u_n)`，随后
   `u_{n+1} = Solve(u_n, S_n)`。
2. **旧管线判负**：`tests/train_cfc_interpolation.py` 用 `source[:, k+1]` 作为
   `u_k → u_{k+1}` 的控制项。按本契约，`S_{k+1}` 是 `u_{k+1} → u_{k+2}` 的源、且由
   `u_{k+1}` 参与算出 ⇒ 旧结果不仅 teacher-forced，还**多用了整整一步的未来源**。
   旧 teacher-source 指标不得作为基线、对照或 Gate（ADR-0018 已撤销其基线地位）。
3. **新数据自描述**：每个流体步导出 `(interval_start, interval_end, u_before, S_used,
   u_after, p_start, p_end)`；时间单位统一；不得再用文件名编号隐式约定时序。
4. **加载器/数据集断言**：manifest 与 dataset 必须校验
   `jorek t_now` 严格递增、`projection` 配对字段时间差 = 0、`part` 时间 = 配对场帧 − dt；
   restart 边界帧要么剔除，要么按显式区间重新对齐（ADR-0012 的 purge 规则同样适用）。
5. **文档同步**：CONTEXT、`report/marker-projection-causality.md`、stage summary、pitfalls、
   implementation-progress、roadmap_v2（Phase A/B/C、§5）按本契约修正。

## 后果

- 源模块的输入必须包含采样窗口的两端信息：`(u_{n−1}, u_n, p_{n−1})` 或等价的记忆表示；
  旧 source head 的 `(u_n, shard_n) → S_n` 同号瞬时配对（`shard_n` 在窗口末端）也是错的，
  重训时必须修正。
- 闭环推演顺序固定为：粒子演化（生成 `p_n` 与 `S_n`）→ 场求解（`u_n → u_{n+1}`）；
  评估只在记录帧上计算 closer（closer 不用于中间猜测帧）。
- 旧数据集（EHL-2）可按本契约重建配对；restart 边界帧需登记与剔除。

## 2026-10-07 补充：登记 restart 事件允许存在（被 ADR-0020 决定 5 覆盖）

**背景**：生产 run 的代码重启产生 `dt=0` 重复帧，生成侧决定**不重导**数据
（2026-10-07 用户指令）。实测 `phaseA_dataset_1d22_ang15_massloss_neon_v400`
有 9 个事件（2702/4111/5871/6871/8371/9421/10421/11421/11711），
`..._injInner` 有 6 个（2704/5681/6441/7221/8611/9291）；
每个事件 n 的 `projections{n−1}` 偏移恰为 `−dt_source`，`projections{n}` 与
`jorek{n}` 对齐。**本补充修订本文决定 4 与 §5.4 中"restart 边界帧一律剔除/重对齐"
的措辞**。

**修订后的规则**：

1. `dt=0` 帧必须登记在 restart 事件表（`data/restart_events.json`、
   `<run>/restart_events.json`、sidecar `phase_a_native_run.json` 的
   `restart_events` 键，或 CLI `--restart-events`）；**未登记**的 `dt<=0`
   仍是 fail-fast。
2. 登记事件 n 的处理：跳过跨界配对 `(n−1 → n)`（零步长）与滞后投影
   `projections{n−1}`（校验其偏移为 `−dt_source`）；事件帧 n 保留，
   `S_n = projections{n}` 与其对齐，其源窗口 `[T_{n−1}, T_n]` 是**退化窗口**——
   登记 provenance，不得伪造/插值。
3. loader 默认 `zero_dt_policy="registered"`；`"error"` 保留给新生成数据的
   严格去重模式，`"list"` 只作旧数据诊断。
4. 新导出仍应满足本文决定 3（自描述 `interval_start/end`）；
   旧数据的推导区间与登记 provenance 见数据契约 v3 §2.4/§4.3。

**后果**：`src/data/native_run.py`/`StepRunDataset` 默认接受登记事件并输出
`restart_cross_boundary_skipped`/`restart_lagging_projection_skipped`/
`restart_degenerate_source_windows` 报告；B1 训练跳过退化源窗口
（`dt_source<=0`），B2/C 的场步 `n→n+1` 仍可用。
