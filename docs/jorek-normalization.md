# JOREK Model 502 归一化规范（代码核实版，2026-10-04）

**状态**：current / canonical。适用对象：本仓库使用的 EHL-2 Model 502 二维
（`ntor=1`，n=0 分量）数据与物理模块。所有条目均以
`/home/ac_desktop/JOREK_code/jorek` 的源码为准，冲突时以源码与本文件为准。

> 背景：仓库原有的 `src/physics/definitions.md` §5.1–5.3 写的是通用 JOREK
> R0/B0/vA 归一化模板，与本仓库实际数据路径（B、ψ 直存 SI；时间用 sqrt(mu0 rho0)）
> 不一致；§5.4/§8 才接近实际。本文把两套叙述彻底分开，并补上 B2 stencil 稳定性
> 所需的"输运系数单位"一节。

## 0. 一句话结论

本路径是**混合单位制**：`R/Z` 用米、`B` 用特斯拉、`ψ` 用 T·m²（三者均 SI，不按
R0/B0 无量纲化）；`rho/T/u/v_par` 等按 `n0/rho0/mu0` 无量纲化；时间单位是
**Alfvén 时间 `t_norm = sqrt(mu0*rho0)` 秒**。输运系数（`D`、`visco`、`ZK` 等）
是"米² 每 JOREK 时间"的归一化扩散率，**直接与 `tstep`（JOREK 时间）相乘**，
不需要再乘 `t_norm`；`t_norm` 只用于把 JOREK 时间换算成秒。

## 1. 参考量（reference quantities）

代码依据：`models/mod_plasma_functions.f90:22-28`、
`models/model502/mod_elt_matrix.f90:1305`、
`communication/mod_export_restart.f90:716-718`。

| 量 | 定义 | 说明 |
|---|---|---|
| `n0` | `central_density * 1e20` [m^-3] | `central_density` 来自 namelist |
| `rho0` | `central_mass * ATOMIC_MASS_UNIT * n0` [kg/m^3] | `central_mass` 来自 namelist |
| `t_norm` | `sqrt(mu0 * rho0)` [s] | H5 同时写为 `sqrt_mu0_rho0` 与 `t_norm` |
| `sqrt_mu0_over_rho0` | `sqrt(mu0 / rho0)` | H5 写入；用于 `eta` 的 SI 换算 |
| `F0` | 真空环向场函数 `R*B_phi` [T·m] | 直存 SI，namelist 值 ≈ -3.3087 |

实测锚点（`scan_1d22_ang12_massloss_neon`）：`t_norm = 4.085775226e-7 s`，
`sqrt_mu0_over_rho0 = 3.075639241`。

> **不要**使用通用模板里的 `vA = B0/sqrt(mu0 rho0)`、`tauA = R0/vA`、
> `psi_norm = R0^2 B0`——本路径的 `B`、`ψ` 是 SI 值，`B0` 不参与归一化。

## 2. 状态与派生量的换算

代码依据：`particles/mod_fields.f90:235-295`（`calc_VBpsiU`）、
`particles/mod_fields.f90:300-333`（`calc_NeTe`）、
`models/mod_plasma_functions.f90:32-45`。

| SI 量 | 由 JOREK 值换算 | 代码依据 |
|---|---|---|
| `R, Z [m]` | `= R, Z`（直存米） | `BigR = x_g`（`mod_elt_matrix.f90:363`） |
| `B [T]` | `= [psi_Z, -psi_R, F0]/R`（直存 T） | `mod_fields.f90:283` |
| `psi [T·m^2]` | `= psi`（直存 T·m²，保证 B 为 T） | `mod_fields.f90:283` |
| `n [m^-3]` | `= n0 * rho` | `mod_fields.f90:320`（`central_density*1e20`） |
| `rho [kg/m^3]` | `= rho0 * rho` | 同上 + `rho0` 定义 |
| `T_e [K]` | `= T_e / (k_B * mu0 * n0)`（with_TiTe） | `mod_fields.f90:321-325`、`mod_plasma_functions.f90:33-34` |
| `T_e [eV]` | `= T_e / (e * mu0 * n0)` | `mod_plasma_functions.f90:33`（`Te0_keV`） |
| 流函数 `U` | `U = P_u / t_norm` | `mod_fields.f90:279` |
| 平行速度因子 | `Vpar = P_vpar / t_norm`，最终 `v_par,SI = Vpar * B_SI` | `mod_fields.f90:286-290`；与 §5.4 表 `v_par,SI = v_par*B_SI/t_norm` 一致 |
| 速度场 `V [m/s]` | `V = [-U_Z, U_R, 0]*R/t_norm + Vpar*B`（`B` 为特斯拉） | `mod_fields.f90:288-290` |
| `j_phi [A/m^2]` | `= -Z_j / (R * mu0)` | `definitions.md` §5.4 表（与 `Z_j=Δ*ψ` 一致） |
| `omega_phi [1/(m·s)]` | `= omega / t_norm` | `definitions.md` §5.4 表 |
| `p [Pa]` | `= rho * T / mu0` | `definitions.md` §5.4 表 |
| `E [V/m]` | `= E / t_norm` | `definitions.md` §5.4 表 |
| `eta [Ω·m]` | `= eta * sqrt(mu0/rho0)` | `mod_plasma_functions.f90:44`（`eta_Spitzer`） |

**温度因子 2 的注意点**：单温闭合（`WITH_TiTe` 未开）时 `T` 代表总压力温度，
代码用 `T_norm = 1/(k_B*2*mu0*n0)`；本仓库 Model 502 数据是 `with_TiTe`，
用 `T_e = P * 1/(k_B*mu0*n0)`（无因子 2）。见 `mod_fields.f90:321-324`。

## 3. 输运系数与时间推进（B2 stencil 稳定性直接相关）

### 3.1 系数在方程中的角色

JOREK 的弱形式矩阵把输运系数直接与 `tstep`（JOREK 时间）和几何项相乘，例如：

- 平行热传导：`(ZK_i_par_T-ZK_i_prof) * BigR/BB2 * Bgrad_T_star * Bgrad_Ti * xjac * tstep`
  （`models/model502/mod_elt_matrix.f90:1595`）；
- 平行粒子扩散：`(D_par_local-D_prof) * BigR/BB2 * Bgrad_rho_star * ... * xjac * tstep`
  （`mod_elt_matrix.f90:1546-1547`）；
- 粘性：`visco_T * BigR * (v_x*w0_x + v_y*w0_y) * xjac * tstep`
  （`mod_elt_matrix.f90:1492`）。

因此这些系数是**归一化扩散率**（对 `D`/`visco` 即"米² 每 JOREK 时间"；
`ZK`/`K` 为对应的归一化热导系数），单位换算为：

**逐项度量权重（2026-10-04 代码核对）**：JOREK 的弱形式并非统一权重——
平行项用 `BigR/BB2`（如 `mod_elt_matrix.f90:1546-1547,1595,1642`），
垂直/粘性项用 `BigR`（如 `:1492,1565,1598,1641,1649`），
数值粘性 `visco_num` 甚至不带 `BigR`（`:1496,2031`）。因此 B2 的
`metric_weight` 必须是**逐项**的（平行 `BigR/BB2`、垂直/粘性 `BigR`、
`visco_num` 不带权重），不能用一个全局开关近似。

**通道-项映射（代码行号）**：`ZK_i_par`→T_i（`:1595`）、`ZK_e_par`→T_e（`:1757` 附近）、
`ZK_perp_num`→T_i/T_e（`:1598,1757`，B2 当前未纳入）、`D_par_local*`→rho/rho_imp
（`:1546-1547`）、`D_perp_num`→rho（`:1565`）、`visco*`→u/omega（`:1492,2029`）、
`visco_par*`→v_par（`:1641-1649`）。

> **2026-10-04 B2.1 勘误**：以上两条待办均已落地——`src/models/implicit_stencil.py`
> 的 `TERM_METRIC_WEIGHT`/`DEFAULT_CHANNEL_TERMS` 逐项实现，并补入
> `ZK_perp_num`/`D_perp_num`/`visco_num`/profile 垂直项（`D_perp`/`D_perp_imp`/
> `ZK_i_perp`/`ZK_e_perp`）。注意：在**单位方向平行核**上，字面因子
> `BigR/BB2` 与 B 量级核相消，等价权重是 `BigR`。逐项权重后的真实帧 λ_max 与
> 稳定界见 `report/phase-b2.1-quasi-steady-2026-10-04.md` §4
> （径向 6.097e6、平行 4.561e6；§3.2 下表为 B2 无权重谱）。

| 系数 | SI 值 | 归一化值（namelist） |
|---|---|---|
| `D` | `D_SI = D / t_norm` [m²/s] | `D`（见 §5.4 表） |
| `nu`（运动粘度） | `nu_SI = mu_SI/rho_SI` | `visco`，ρ≈1 时 `nu_SI ≈ visco/t_norm` |
| `K` | `K_SI = K * sqrt(rho0/mu0)/(gamma-1)` | `ZK_*` |
| `eta` | `eta_SI = eta * sqrt(mu0/rho0)` | `eta`（namelist） |

**结论**：显式稳定性判据用 JOREK 时间单位直接算

```
dt_term = 2 / (α_term * λ_geom)，α_term = namelist 系数
```

**不需要**再乘或除 `t_norm`；`t_norm` 只用于换算到秒。

### 3.2 逐项显式稳定界（B2.1 逐项权重后的精确谱）

B0 的 `1/λ_max ≈ 1.74e-7` 是 α=1 的几何界；B2.1 已按 JOREK 逐项度量权重装配算子
并求精确谱（`report/phase-b2.1-quasi-steady-2026-10-04.md` §4）：
径向（各向同性）`λ_max=6.097275e6`、平行 `λ_max=4.561210e6`（B2 无权重谱为
5.0230e6/3.7604e6；B0 行和代理 5.7527e6/4.3363e6）。按本 run
（`scan_1d22_ang15_5_massloss_neon_v400`）折算：

| 项 | α | `dt = 2/(α·λ_max)`，JOREK 时间 | dt=0.05 显式稳定？ |
|---|---:|---:|---|
| η（磁扩散） | 1.443e-10 | 2.273e3 | 是 |
| D_perp | 5e-6 | 6.56e-2 | 是（临界） |
| ZK_perp_num | 1e-12 | 3.28e5 | 是 |
| visco | 2.4e-3 | 1.367e-4 | 否 |
| visco_num | 9e-4 | 4.42e-4 | 否 |
| visco_par | 5e-4 | 6.56e-4 | 否 |
| visco_par_num | 1e-4 | 3.28e-3 | 否 |
| D_par | 5e-2 | 8.77e-6 | 否 |
| ZK_i_par | 1.67e5 | 2.63e-12 | 否（刚性） |
| ZK_e_par | 4.10e6 | 1.07e-13 | 否（刚性） |

注：`ZK_par_max=1e4` 把温度依赖的平行导热截断后，`dt_limit=2/(1e4·4.56e6)=4.38e-11`，
仍然刚性；基础值（表中 1.67e5/4.10e6）为未截断上界。

**系数取值链（2026-10-04 核实）**：`jorek003000.h5` 直存
`visco=2.4e-3`、`visco_par=5e-4`、`eta=1.443e-10`、`central_density/central_mass/t_norm`，
与生产 namelist **`jorek_namelist_spitest`** 一致；`D_par=5e-2`、`ZK_par_max=1e4`
取该 namelist。run 目录中的 `jorek_namelist_2d`（visco=1e-5、D_par=1e-2、
`ZK_par_max=5e4`）、`lp_2d/wallgrid/xpt_test/postproc` 不是该 run 的有效输入，
**不要**用于 α 或稳定性计算。生成侧仍应最终确认有效值。

### 3.3 为什么"归一化做对了"仍然会刚性（常见疑问）

归一化后的显式稳定判据是 `dt ≤ 2/(α·λ_max)`；刚度比（explicit stability margin）
为 `dt / dt_limit = α·λ_max·dt / 2`。以 dt=0.05 JOREK 时间、B2 精确 λ_max 计算：

| 项 | α | α·λ_max·dt | dt_limit | 刚度比 |
|---|---:|---:|---:|---:|
| η | 1.443e-10 | 4.4e-5 | 2.273e3 | 2.2e-5 |
| D_perp | 5e-6 | 1.52 | 6.56e-2 | 0.76 |
| visco | 2.4e-3 | 7.3e2 | 1.37e-4 | 3.7e2 |
| visco_par | 5e-4 | 1.5e2 | 6.56e-4 | 76 |
| D_par | 5e-2 | 1.1e4 | 8.77e-6 | 5.7e3 |
| ZK∥（cap=1e4） | 1e4 | 2.3e9 | 4.38e-11 | 1.1e9 |

结论：**正确归一化不会消除刚性**。原因是平行热传导系数本身就很大（`ZK_par_max=1e4`
上限），且网格在 X 点/轴附近有很小的单元（`λ_max≈3.8e6 m^-2`）。用 SI 系数或漏乘
`t_norm` 只会让数值差 O(1)–1e6，不改变"物理 dt=0.05 下显式推进必然不稳定"这一结论；
JOREK 每步解隐式矩阵正是为此。B2.1 已用 direct LU 隐式求解把保真度补上
（真实帧 T_i 一步 m-范数修正 1.42e-1，旧阻尼迭代仅 9.69e-7，差 1.5e5 倍；
backward error ≤1.1e-12）；剩余是求解器可扩展性（GPU 无 LU 时的场向预条件/多重网格）
与真实帧剖面需 A′ 细粒度数据对照，不是稳定性。

注意：

- `λ_max` 用的是 B0 的 `A/m`（弱形式拉普拉斯）谱半径代理；**逐项精确界**应对每个
  弱形式算子单独装配求谱半径（B2 的组件任务），但量级与结论（η 不刚性、平行输运刚性）
  不变。
- 系数是温度相关的：`ZK ∝ T^{2.5}`（有 `ZK_par_max` 上限 1e4）、`visco ∝ T^{-1.5}`、
  `eta ∝ T^{-1.5}`（`mod_elt_matrix.f90:691-737`）。稳定性判定要用**当前状态的系数**，
  不能只用 namelist 基值。

## 4. 常见误区

1. **不要混用 §5.1–5.3 的 R0/B0 归一化**：本路径 `B`、`ψ` 是 SI；`psi_norm=R0²B0`
   等公式不适用。
2. **不要重复乘 `t_norm`**：系数已经在归一化单位；只有转成秒时才乘 `t_norm`。
3. **`B_norm` 不是归一化常数**：粒子代码里的 `B_norm` 是单位向量 `B/|B|`
   （`mod_fields.f90:742`），不是"B 的归一化因子"。
4. **温度因子 2 只属于单温闭合**；Model 502 数据（`with_TiTe`）用无因子 2 的公式。
5. **COCOS 符号约定只作用于 IMAS 导出**（`mod_jorek2IMAS.f90` 的
   `fact_psi/fact_w/fact_phi/fact_zj`），不影响内部九场数值。

## 5. 代码证据索引

| 主题 | 位置 |
|---|---|
| `rho0 / sqrt_mu0_rho0 / sqrt_mu0_over_rho0` | `models/mod_plasma_functions.f90:22-28` |
| `t_norm` | `models/model502/mod_elt_matrix.f90:1305`；`particles/mod_fields.f90:263` |
| H5 写出 `sqrt_mu0_rho0`/`t_norm`/`sqrt_mu0_over_rho0` | `communication/mod_export_restart.f90:716-718` |
| `B = [psi_Z, -psi_R, F0]/R`（T）、`u/t_norm`、`v_par/t_norm` | `particles/mod_fields.f90:279-290` |
| `n_e`、`T_e` 更新与温度因子 | `particles/mod_fields.f90:320-325` |
| `Te0_keV`、`eta_Spitzer` | `models/mod_plasma_functions.f90:32-45` |
| 系数×弱形式×`tstep`（D/粘性/ZK） | `models/model502/mod_elt_matrix.f90:1492,1546-1547,1595` |
| 系数温度依赖与上限 | `models/model502/mod_elt_matrix.f90:691-737` |
| 网格坐标 `BigR = x_g`（米） | `models/model502/mod_elt_matrix.f90:363` |

## 6. 待确认/待补

- 逐项弱形式算子的精确谱半径已由 B2 完成；**逐项度量权重与通道-项映射已由 B2.1
  落地**（`report/phase-b2.1-quasi-steady-2026-10-04.md` §2/§3/§4）。
- `ZK_par_max` 在实际 run 中的生效值（T 依赖分支）与 Mach-1/鞘层线性化项的尺度
  （`use_simple_bnd_types`/`no_mach1_bc` 开关；见 B0 报告 §6）。
- `freeboundary=T` 真空耦合的归一化（本数据为 F，未触发）。
