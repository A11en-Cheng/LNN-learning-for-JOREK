# Model 502 参量面审计

审计日期：2026-09-06。依据 `models/model502/mod_elt_matrix(_fft).f90`、`mod_sources.f90`、
`models/diffusivities.f90`、`communication/mod_export_restart.f90`。机器可读清单见
[model502-parameter-surface.json](model502-parameter-surface.json)，由 `tests/test_parameter_surface.py`
与真实 `jorek*.h5` 逐键核对。

## 目标与结论

给定当前 9 场原生状态 + 方程所需的零维参数，下一步状态可唯一推出（马尔可夫闭包）。装配方程读取的
零维参数分四类：已随 h5 导出（可直接做条件）、需在 `mod_export_restart.f90` 补导出（#0004 清单）、
由网络隐式吸收（不导出、不辨识）、源项通道（走 #0002/#0010）。重点结论：**让网络吸收高阶截断与
状态依赖闭包后，条件集收敛到很小的一组基底标量。**

## 高阶截断的定位

方程里的高阶数值耗散由三个超项构成，全部带温度依赖因子，是网络应当隐式学习替代的部分：

- `eta_num_T`（超电阻）：`mod_elt_matrix.f90:1469`，`eta_num * (T/T0)^-1.5` 截断（:749-757）。
- `visco_num_T`（超粘性）：`mod_elt_matrix.f90:1488`，`visco_num * (T/T0)^-1.5` 截断（:760-768）。
- `ZK_par_num`（超平行热导）：`mod_elt_matrix.f90:1592`，`_fft.f90:1623,2373,2375`。

这些项连同状态依赖闭包（`eta_T_ohm(T,Zeff)`、`visco_T(T)`、`ZK_par_T`（即 `ZK_i/e_par_T ∝ (T/T0)^2.5`）、
`coronal_radiation`（ADAS Lz）、`bootstrap_coefficients`（新经典 bootstrap 电流））都标注为 `absorbed`：
不作为条件输入，也不作为参数辨识目标（ADR-0005/0006）。

## 已随 h5 导出（可直接进条件向量）

`eta, visco, visco_par, tstep, t_now, central_density, central_mass, F0, R_domm,
sqrt_mu0_rho0, t_norm, n_spi_tot, n_inj`。每帧 `jorek*.h5` 均含这些标量；parser 参数通道（#0005）
直接读取即可，无缺失。

## 需补导出（#0004 清单）

| 参数 | JOREK 变量 | 说明 |
|---|---|---|
| ZK_i_par | `ZK_i_par` | 离子平行热导基底 |
| ZK_e_par | `ZK_e_par` | 电子平行热导基底 |
| D_perp | `D_perp(1:6)` | 垂直粒子扩散剖面系数（`get_dperp1`） |
| ZK_i_perp | `ZK_i_perp(1:6)` | 离子垂直热扩散剖面系数 |
| ZK_e_perp | `ZK_e_perp(1:6)` | 电子垂直热扩散剖面系数 |
| visco_par_par | `visco_par_par` | 平行粘性（平行流） |
| time_evol_theta | `time_evol_theta` | 时间积分 theta（:222） |
| time_evol_zeta | `time_evol_zeta` | 时间积分 zeta（变 dt 重标定 :225） |
| T_0/Ti_0/Te_0 | `T_0, Ti_0, Te_0` | 闭包参考温度（写出路径存在，当前帧未含） |

每个标量在 `mod_export_restart.f90` 中即一行 `HDF5_real_saving(file_id, <var>, '<name>'//char(0))`；
剖面系数为 6 元素一维数组，用 `HDF5_array1D_saving`。若 `num_d_perp/num_zk_i_perp/num_zk_e_perp`
开启，则对应的数值剖面数组 `num_*_x/y` 也需一并导出。

## 源项通道（#0002/#0010，不进入条件向量）

- `background_sources`：mod_sources 的 `(particle_source, heat_i_source, heat_e_source)` 背景源，作为可选第二控制通道。
- `pellet_impurity_sources`：`source_pellet/source_bg/source_imp`，由 marker_spi_source 粒子投影喂入。
- `marker_projections`：`projections*.h5` 五场投影 `[dE_ion, rad_sink, Z_eff, Z_imp, N_imp]`，是含源训练的主控制通道。

## 与 h5 实况的核对

`tests/test_parameter_surface.py::test_exported_keys_exist_in_real_frame` 用
`data/EHL-2/scan_1d22_ang12_massloss_neon/jorek002710.h5` 验证每个 `exported` 键真实存在；
该帧共 40 个标量数据集，`ZK_*`、`D_perp`、`visco_par_par`、`theta/zeta` 确实缺席，与"需补导出"
标注一致。`dcoef` 已导出但为 Dommaschk 平衡系数，明确不属于输运系数。
