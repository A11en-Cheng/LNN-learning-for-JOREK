# Marker 投影场的来源与时序 — 源码 + 数据核实（2026-10-04 修正版）

本文档记录 Model 502 `use_marker` 模式下五个源项投影场
`[dE_ion, rad_sink, Z_eff, Z_imp, N_imp]` 的生成方式、与九个状态场的依赖关系、
以及与流体场更新的时间耦合。

> **2026-10-04 勘误**：本文旧版结论（"`P_i` 驱动 `u_{i-1} → u_i`、按目标帧配对"）
> 经 JOREK 源码 + 生产日志 + 全量数据统计核实**不成立**。现行契约见
> [ADR-0019](../docs/adr/0019-recorded-output-timing-contract.md)：
> **`projections{n}` 驱动 `u_n → u_{n+1}`；`part{n}` 是 `T_{n−1}` 的粒子状态。**
> 旧训练管线的 `source[:, k+1]` 配对因此是一步未来的源（泄漏）。

核实材料：

- 源码（`/home/ac_desktop/JOREK_code/jorek`，与 `/mnt/d/Proj/jorek` 同一树）：
  `particles/examples/marker_spi_source.f90`（粒子主循环）、
  `particles/diagnostics/mod_project_particles.f90`（`save_to_h5`）、
  `core/mod_jorek_timestepping.f90`（九场求解与 `jorek*.h5` 写出）、
  `particles/mod_particle_io.f90`（`part*.h5` 写出 `/time`）；
- 生产日志：`data/EHL-2/scan_1d22_ang12_massloss_neon/logfile_spitest`（索引 7360 非 restart 段）；
- 全量数据：`scan_1d22_ang12_massloss_neon` 的 620 个匹配索引，
  `scan_2d21_ang15_5_massloss_neon_v300`、`scan_1d22_ang15_5_massloss_neon_v400` 抽样复验。

## 1. 时间耦合：文件级时序契约（ADR-0019）

设 `T_n` = `macroscopic_vars.dat` 第 n 个 JOREK 流体步的时刻，`dt_n = T_n − T_{n−1}`：

| 文件 | 物理时间 | 角色 |
|---|---|---|
| `jorek{n}.h5` | `T_n` | 九场 `u_n`（第 n 步求解结果） |
| `projections{n}.h5` | 采样窗口 `[T_{n−1}, T_n]` 的时间平均；内部 `t_now = T_n` | 源 `S_n`，被第 n+1 步求解消耗 ⇒ **驱动 `u_n → u_{n+1}`** |
| `part{n}.h5` | `T_{n−1}`（比同号场帧早一个 JOREK 步） | `S_n` 采样窗口的**起点**粒子状态 `p_{n−1}` |
| `part{n+1}.h5` | `T_n` | 窗口终点粒子状态 `p_n` |

证据：

1. **源码**：粒子在 `u_{n−1} → u_n` 的线性插值场中走子步
   （`mod_fields_linear.f90` 用 `time_prev/time_now` 线性插值），
   窗口结束做 `jorek_feedback%rhs = feedback_rhs / (n_steps·timesteps)`（窗口时间平均）；
   随后 `do_jorek_timestep` 递增 `index_now`、组装矩阵求解、`t_now += dt_jorek` 后写
   `jorek{index_now}`（`mod_jorek_timestepping.f90`）。
2. **生产日志顺序**（`logfile_spitest`，索引 7360）：
   `After step 007360` → 粒子窗口（`sim%time=T_7359`，`target_time=T_7360`）→
   `Written projection to projections007360.h5` → `After step 007361`；
   即投影在下一步求解**之前**写出、被下一步**消耗**。
3. **粒子内容校验**：日志在该窗口起点打印的粒子权重 `0.16093158E+22` 与
   `part07360.h5` 权重总和 `1.609298E+21` 一致 ⇒ `part{n}` 确实是窗口起点状态。
4. **全量统计**（`scan_1d22_ang12_massloss_neon`，620/620）：
   `jorek` 的 `t_now` = `macroscopic_vars` 第 n 步时间；
   `projections` 的 `t_now/t_norm` = 同号场帧时间（614/620）；
   `part` 的 `time/t_norm` = 同号场帧时间 − `dt_n`（620/620）。
   第二、第三个 run 抽样复验一致。

**restart 边界例外**：`scan_1d22_ang12_massloss_neon` 的
3550 / 4150 / 4950 / 5330 / 6600 / 7350（均见 `available_restart_files.txt`）以及
v400 的 5050 上，`projections{k}` 比同号 `jorek` 早一个 `dt_source`（偏移恰为
`−dt_source`），而 `part{k}` 始终 = `T_k − dt_source`（常规规则，不受 restart 影响）。
这些 projections 必须按登记剔除或按显式区间重新对齐，不能直接进训练；`part` 无需处理。

**注意**：仓库 example 的事件数组把投影事件排在 JOREK stepper 之前，生产数据的日志顺序相反；
仅凭 example 的事件顺序推断数据语义会得到错误结论。新数据生成必须固定顺序并自带区间元数据。

**闭环配对（唯一规则）**：给定 `(u_n, p_{n−1})`：
① 粒子在 `u_{n−1} → u_n` 的插值场中推进，产生 `p_n` 并累计 `S_n`；
② 场求解 `u_{n+1} = Solve(u_n, S_n)`。
训练配对固定为 `(u_k, p_{k−1}=part{k}, S_k=projections{k}) → u_{k+1}`。

## 2. 五个投影通道的推导

### 粒子运动（决定沉积位置）

`calc_VBpsiU`（`mod_fields.f90` L235）在粒子位置插值
`[var_psi, var_u, var_Vpar]`（L253），构造速度 V 与磁场 B；
粒子按 `v_fieldline_adams_bashforth_push_cylindrical(...)`（L658）推进。
因此投影的空间分布由 **ψ、Φ、v_par** 决定。

### 局部流体采样（决定速率）

`calc_NeTe_imp`（`mod_fields.f90` L335）插值
`[var_rho, var_Te, var_rhoimp, var_rhon]`，得到背景密度 n_bg、杂质密度 n_imp、
中性密度 n_n、电子温度 T_e，并做闭包 `n_e = n_rho + Z_imp_tmp * n_rho_imp`（L526）。

### 五场表达式（`marker_spi_source.f90`）

| 通道 | 表达式 | 行号 |
|---|---|---|
| dE_ion | Σ_iZ (P_tmp(iZ) − P_imp(iZ)) · E_ion_tot(iZ) · weight | L600 |
| rad_sink | Σ_iZ P_imp(iZ) · weight · Lz(n_e, T_e, n_n) · timesteps | L547–L548 |
| Z_imp | weight · ⟨P_imp, iZ⟩ · timesteps | L535 |
| Z_eff | weight · ⟨P_imp, iZ²⟩ · timesteps | L536 |
| N_imp | weight · timesteps | L537 |

其中 P_imp 是 marker 的内部电荷态分布，由 ADAS 电离/复合系数
（SCD/ACD，在 log n_e、log T_e 查表）逐子步更新（L558–L594）。
沉积到网格：`feedback_rhs(..., 1:5)`（L638–L642），权重为粒子所在单元的 FEM 基函数。

## 3. 与九个状态场的依赖关系

| 投影通道 | 依赖的状态场 |
|---|---|
| 沉积位置 | ψ, Φ, v_par（粒子轨迹） |
| dE_ion, rad_sink | ρ, ρ_imp, T_e（经 n_e/T_e 查 ADAS 表） |
| Z_eff, Z_imp, N_imp | 粒子电荷态分布（同一组速率演化）× 权重 |

注意：投影还依赖 marker 的**内部电荷态分布 P_imp**——这是 9 场之外的隐藏状态，
不完全由 `u` 决定。因此 `S_n` 额外编码了 marker 群体的信息。

## 4. 对训练接线的结论

1. **场模型**：`(u_k, S_k) → u_{k+1}`。旧管线用 `source[:, k+1]`（目标帧索引）是一步未来
   的源（`S_{k+1}` 由 `u_{k+1}` 参与、驱动 `u_{k+1} → u_{k+2}`），属泄漏；旧结果不可作基线。
2. **源模块**：`S_n` 的输入应含窗口两端信息 `(u_{n−1}, u_n, p_{n−1}=part{n})` 或等价记忆；
   旧 source head 的 `(u_n, shard_n) → S_n` 同号瞬时配对（`shard_n` 在窗口末端）也是错的。
3. **restart 边界帧**：3550/4150/4950/5330/6600/7350（run 1）与 5050（v400）需剔除/对齐。
4. **已知近似**：50 帧窗口内只有首尾投影快照，粗模型假设源在窗口内恒定；注入期不成立。
5. **已知近似**：投影隐含 marker 隐藏态，跨 run 的 marker 群体差异带来分布漂移。
6. **新数据导出要求**：每流体步自描述 `(interval_start, interval_end, u_before, S_used,
   u_after, p_start, p_end)`；加载器加时间断言（见 ADR-0019）。
