# Marker 投影场的来源与时序 — 源码存档

本文档记录 Model 502 `use_marker` 模式下五个源项投影场
`[dE_ion, rad_sink, Z_eff, Z_imp, N_imp]` 的生成方式、与九个状态场的依赖关系、
以及与流体场更新的时间耦合，作为数据接线（ADR-0005）的因果依据。

源码基准：

- `/mnt/d/Proj/jorek/particles/examples/marker_spi_source.f90`（粒子主循环）
- `/mnt/d/Proj/jorek/particles/mod_fields.f90`（流体场插值）
- `/mnt/d/Proj/jorek/models/model502/mod_elt_matrix_fft.f90`（投影在流体方程中的消费位置）

## 1. 时间耦合：错步显式（staggered explicit）

粒子模拟是主循环，JOREK 作为事件挂载（`marker_spi_source.f90`）：

1. `do while (.not. sim%stop_now)`（L342）——每个迭代对应一个 JOREK 步；
2. `target_time = next_event_at(...)`（L344）——下一个 JOREK 事件时刻 t_{k+1}；
3. 粒子在**冻结的旧场 F_k** 里走 `n_steps` 个子步（L353），每个子步先采样流体、算速率、
   沉积投影（L492–L642），再推粒子（L657–L658）；
4. 窗口结束时 `jorek_feedback%rhs = feedback_rhs / rhs_gather_time`（L671），
   其中 `rhs_gather_time = n_steps*timesteps`（L443）——投影是**窗口时间平均**；
5. `sim%time = target_time`（L749）后触发事件（L753），JOREK 迈出 F_k → F_{k+1}，
   消耗刚算好的投影。

结论：投影 P_{k+1} 用旧场 F_k 算出、驱动产生 F_{k+1}，时间戳落在**输出帧** t_{k+1}。
粗采样（每 50 JOREK 步一帧）下，P_i 由 t_i−Δt（≈当前帧）的场算出，
即 **P_i ≈ f(x_i)，且 P_i 驱动产生 x_i**。

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
不完全由 x_i 决定。因此 P_i 额外编码了 marker 群体的信息。

## 4. 对训练接线的结论

1. `(x_i, P_i) → x_{i+1}` 因果正确：P_i ≈ f(x_i) 是当前场的即时反馈，驱动产生 x_i。
   不应改用 P_{i+1}（P_{i+1} ≈ f(x_{i+1})，为目标泄漏）。
2. 已知近似一：50 帧窗口内只有首尾两个投影快照，模型假设源项在窗口内恒定；
   注入事件期源项在几个 JOREK 步内大幅变化，此假设失效。
3. 已知近似二：投影隐含 marker 隐藏态，跨 run 的 marker 群体差异会带来分布漂移。

数据侧改进方向：让 JOREK 输出窗口累积/平均投影（`rhs_gather_time` 扩到 50 步），
使粗模型的强迫成为窗口正确的。
