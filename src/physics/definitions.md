# JOREK Reduced MHD Physics Definitions & Normalization

本文件定义 HyPhy-GLNN 在 JOREK 数据上的物理规范。当前项目统一采用 `Model 502` 作为控制方程来源；`Model 500/501` 仅保留历史对照，不再作为主规范。

## 0. 适用范围与执行规则
- 适用数据：本仓库后续训练与评估数据集统一按 `Model 502` 理解。
- 约束优先级：`Model 502` 主方程 > 本文件旧版条目 > 代码实现中的临时简化。
- 网络硬约束：仍遵循 Reduced MHD Ansatz，网络不直接预测 `B` 与 `v`，而由 `psi/Phi/v_par` 重建。

## 1. Model 502 的核心物理定义（新增）

### 1.1 磁场与速度场 Ansatz
在柱坐标 `(R, Z, phi)`：
- 磁场：`B = (F0/R) e_phi + (1/R) (grad psi x e_phi)`
- 速度场：`v = -R (grad u x e_phi) + v_par B`

说明：在本仓库代码中，`u` 对应网络第二通道 `Phi`（流函数/电势流函数变量名沿用历史写法）。

### 1.2 Model 502 压力闭合（双温度，替代旧单温）
`Model 502` 压力定义（来自 note502 Eq.(1)-(2)）：

`P = (rho + xi_i rho_imp) Ti + (rho + xi_e rho_imp) Te`

其中：
- `Ti` 为离子温度，`Te` 为电子温度。
- `rho` 为背景主离子密度（或总密度中的背景分量，具体以 JOREK 变量定义为准）。
- `rho_imp` 为杂质总密度（含各电荷态与中性态总和）。
- `xi_i, xi_e` 为由杂质平均电荷态和质量比给出的系数（note502 中由 `<Z>_imp, m_bg, m_imp` 定义）。

### 1.3 Model 502 主方程组（工程实现采用）
以下写成与 note502 一致的 Reduced MHD 结构（符号保持工程可读性）：

1. 感应方程与安培关系（与 501 相同）
- `dpsi/dt = eta(Te) Delta* psi - R {u, psi} - F0 du/dphi`
- `j = Delta* psi`, `j_phi = j / R`

2. 涡度方程（与 501 同型，仅压力项换为 502 压力闭合）
- `R div[ R^2 ( rho grad_pol(du/dt) + grad_pol(u) drho/dt ) ]`
- `= 0.5 { R^2 j |grad_pol u|^2, R^2 rho } + { R^4 rho omega, u }`
- `- R div[ R^2 grad_pol(u) div(rho v) ] + {psi, j} - (F0/R) dj/dphi + {P, R^2} + R eta(Te) Delta_pol omega`
- `omega = (1/R) d/dR (R du/dR) + d^2u/dZ^2`

3. 总密度连续方程
- `drho/dt = -div(rho v) + div[ D_D grad(rho - rho_imp) ] + div( D_imp grad rho_imp ) + S_D + S_imp`

4. 离子压强方程（对应 note502 的 ion pressure equation）
- `d[(n_bg + n_imp)Ti]/dt = -v·grad[(n_bg + n_imp)Ti] - (n_bg + n_imp)Ti div(v)`
- `+ div(kappa_perp grad_perp Ti + kappa_par grad_par Ti)`
- `+ 0.5 (v·v)(S_D + S_imp) + n_i (dTi/dt)_e`

5. 电子压强方程（对应 note502 Eq.(45)/(46)）
- `d(ne Te)/dt = -v·grad(ne Te) - ne Te div(v) + eta(Te) j^2 / R^2`
- `+ div(kappa_perp grad_perp Te + kappa_par grad_par Te)`
- `- (gamma-1) ne n_imp P_rad(Te) + ne (dTe/dt)_i + S_ion (optional)`

6. 平行动量方程（与 501 同型）
- `B^2 d(rho v_par)/dt = -0.5 rho F0/R^2 d[(v_par B)^2]/dphi - (rho/(2R)){B^2 v_par^2, psi}`
- `- F0/R^2 dP/dphi + (1/R){psi, P} - B^2 div(rho v) v_par + B^2 mu_par(Te) Delta_pol v_par`

7. 杂质连续方程
- `drho_imp/dt = -div(rho_imp v) + div(D_imp grad rho_imp) + S_imp`

### 1.4 电子-离子能量交换项（Model 502 新增关键）
`Model 502` 显式区分电子与离子能量交换：
- `ne (dTe/dt)_i = ne (nu_e_imp + nu_e_bg) (Ti - Te)`
- `(dTi/dt)_e = [ne/(nimp + nbg)] (dTe/dt)_i`

其中 `nu_e_imp, nu_e_bg` 分别为电子与杂质/背景离子碰撞频率，依赖温度、密度、有效电荷与库仑对数。数值离散时需要其对 `Ti/Te/rho/rho_imp` 的导数（note502 Eq.(24)-(43)）。

### 1.5 可选项
- 电离势能耦合：可在电子方程加入 `S_ion`，来自 `dE_ion/dt + div(Gamma_ion) = S_ion`。
- Taylor-Galerkin 稳定化：在密度、垂直动量、平行动量的对流项引入二阶稳定项（note502 Eq.(51)-(58)）。
- Memory array（热电子长尾）：属于喷丸/碎片烧蚀子模型扩展，不改变主 RMHD 方程形式。

## 2. 旧版定义冲突与弃用说明（500/501 -> 502）

| 主题 | 旧版定义（已弃用） | Model 502 定义（现行） |
| :--- | :--- | :--- |
| 热力学闭合 | 单温近似（常见写法 `Te = Ti`，或 `P = rho T`） | 双温闭合：`P = (rho + xi_i rho_imp)Ti + (rho + xi_e rho_imp)Te` |
| 能量方程 | 单一压力方程 | 拆分为 `ion pressure` 与 `electron pressure` 两条方程，并含 `e-i` 能量交换 |
| 辐射/欧姆热作用对象 | 常与单温压力方程合并处理 | 欧姆热与辐射主要进入电子压强方程 |
| 组分追踪 | 500 中可显式中性粒子方程 | 502 主线为总密度 + 杂质总密度（含电荷态总和） |
| 稳定化项 | 可无或较弱处理 | 502 文档明确给出 Taylor-Galerkin 对流稳定项 |

执行结论：凡与上表冲突的旧描述，一律按 `Model 502` 替换并弃用。

## 3. 与当前 5 通道学习变量的对应关系
当前 GNN/LNN 管线输入输出仍为 5 通道：`[psi, Phi, rho, T, v_par]`。

在当前代码 `src/data/jorek_h5_parser.py` 中，`T` 取自 `values_raw[5]`（即 `Ti`），而 `Te` 位于 `values_raw[8]` 未并入 5 通道主状态。由此得到：
- 训练主状态可解释为：`[psi, Phi(=u), rho, Ti, v_par]`。
- `Model 502` 的完整双温闭合在纯 5 通道端到端训练中属于降阶近似；若要严格满足 502，需引入 `Te`（以及必要时 `rho_imp`）作为显式状态或外部闭合输入。

## 4. 物理约束 (Hard Constraints)
模型**绝不**直接预测磁场 `B` 或速度场 `v`。必须通过以下 Ansatz 计算：
- 磁场：`B = (F0/R) e_phi + (1/R) (grad psi x e_phi)`，其中 `F0 = R0 B0`
- 速度场：`v = -R (grad Phi x e_phi) + v_par B`

## 5. 归一化标准 (保留原定义)
本节保留原有 JOREK 无量纲化内容，作为 SI 与代码变量的换算基准。

### 5.1 基础参考量 (Reference Quantities)
所有物理量均基于以下三个基本参考量进行无量纲化：
- 真空环向磁场 (`B0`)：磁轴处的环向磁场强度 [Tesla]
- 大半径 (`R0`)：磁轴处的几何大半径 [m]
- 中心密度 (`rho0`)：初始时刻的中心质量密度 [kg/m^3]

### 5.2 导出参考量 (Derived Quantities)
- 真空磁导率 (`mu0`)：`4*pi*10^-7` [H/m]
- Alfvén 速度 (`vA`)：`vA = B0 / sqrt(mu0 rho0)` [m/s]
- Alfvén 时间 (`tauA`)：`tauA = R0 / vA = R0 sqrt(mu0 rho0) / B0` [s]
- 参考压力 (`p0`)：`p0 = B0^2 / mu0` [Pa]
- 参考温度 (`T0`)：`T0 = p0 / rho0`

### 5.3 核心状态变量归一化 (Primary State Variables)
1. 极向磁通 (`psi`)：`psi_norm = R0^2 B0`
2. 流函数 (`Phi`)：`Phi_norm = R0 vA = R0^2 / tauA`
3. 密度 (`rho`)：`rho_norm = rho0`
4. 温度 (`T`)：`T_norm = T0`
5. 平行速度 (`v_par`)：`v_norm = vA`

### 5.4 物理量汇总表 (JOREK Official Normalization)
- 下标 `SI` 表示 SI 单位制下的物理量；无下标表示 JOREK 内部无量纲单位。
- 重要归一化因子 `sqrt(mu0/rho0)` 与 `sqrt(mu0 rho0)` 会写入 JOREK logfile；需在输入中正确设置 `central_density` 与 `central_mass`。

| SI 单位 | 归一化单位 (JOREK) | 物理量描述 |
| :--- | :--- | :--- |
| `R_SI [m]` | `= R` | 大半径 (Major radius) |
| `Z_SI [m]` | `= Z` | 垂直坐标 (Vertical coordinate) |
| `B_SI [T]` | `= B` | 磁场矢量 (Magnetic field vector) |
| `E_SI [V m^-1]` | `= E / sqrt(mu0 rho0)` | 电场矢量 (Electric field vector) |
| `Psi_SI [T m^2]` | `= Psi` | 极向磁通 (Poloidal magnetic flux) |
| `j_phi,SI [A m^-2]` | `= -j/(R mu0)` | 环向电流密度 |
| `n_SI [m^-3]` | `= rho n0` | 粒子密度 |
| `rho_SI [kg m^-3]` | `= rho rho0` | 质量密度 |
| `T_SI [K]` | `= T/(k_B mu0 n0)` | 温度 |
| `T_eV [eV]` | `= T/(e mu0 n0)` | eV 温度 |
| `p_SI [N m^-2]` | `= rho T / mu0` | 等离子体压力 |
| `v_SI [m s^-1]` | `= v / sqrt(mu0 rho0)` | 速度矢量 |
| `v_par,SI [m s^-1]` | `= v_par * B_SI / sqrt(mu0 rho0)` | 平行速度分量 |
| `u_SI [m s^-1]` | `= u / sqrt(mu0 rho0)` | `R u` 为速度流函数，`F0 u` 为势 |
| `omega_phi,SI [m^-1 s^-1]` | `= omega / sqrt(mu0 rho0)` | 环向涡度 |
| `t_SI [s]` | `= t * sqrt(mu0 rho0)` | 时间 |
| `gamma_SI [s^-1]` | `= gamma / sqrt(mu0 rho0)` | 增长率 |
| `eta_SI [Ohm m]` | `= eta * sqrt(mu0/rho0)` | 电阻率 |
| `mu_SI [kg m^-1 s^-1]` | `= mu * sqrt(rho0/mu0)` | 动力粘度 |
| `nu_SI [m^2 s^-1]` | `= mu_SI / rho_SI` | 运动粘度 |
| `D_SI [m^2 s^-1]` | `= D / sqrt(mu0 rho0)` | 粒子扩散率 |
| `K_SI [kg m^-1 s^-1]` | `= K * sqrt(rho0/mu0)/(gamma-1)` | 热扩散率 |

### 5.5 基础定义 (Base Definitions)
| 变量 | 定义 | 备注 |
| :--- | :--- | :--- |
| `n0 [m^-3]` | `= central_density * 1e20` | `central_density` 在输入文件中指定 |
| `rho0 [kg m^-3]` | `= central_mass * n0 * m_AMU` | `central_mass` 在输入文件中指定 |
| `gamma` | `= GAMMA` | 绝热指数，默认 `5/3` |

### 5.6 常用物理常数 (Useful Constants)
- `m_e0 = 0.911e-30 kg`
- `m_AMU = 1.661e-27 kg`
- `m_deuterium = 2.014101777811 AMU`
- `m_tritium = 5.007e-27 kg`
- `mu0 = 4*pi*1e-7 Vs/(A m)`
- `e = 1.602176565e-19 C`

## 6. 当前待确认项（请你确认）
为确保与你的 `note502` 版本逐字一致，我需要你确认两点：
1. 本项目文档里 `rho` 是否按 `n_bg` 解释，且 `rho_imp` 是否可由现有数据稳定获得（若不可，需要在训练输入中显式新增该通道或由外部闭合提供）。
    
    A: 在 JOREK Model 502 中，rho 代表背景主等离子体质量密度（Background Mass Density），而 HDF5 中 index 为 7 的 N_density，在 Model 501/502 中实际上代表的是杂质总密度 rho_imp（包含了所有电荷态及中性态）。为了代码兼容性，我们在代码变量命名上可以暂用 rho_n，但在写物理 Loss 时，必须按 note502 中的 rho_imp 来做双温压力闭合。

2. 电子压强方程是否默认启用 `S_ion`（电离势能交换），还是按“可选项”关闭后再逐步打开。

    A: 目前我们先按“可选项”处理，默认关闭 `S_ion`。后续如果需要引入电离势能耦合，可以在电子压强方程中添加该项，并根据 JOREK 输出的电离源项数据进行训练。

## 7. 全通道顺序约定 (Full Channel Mapping):
* 0: Psi (极向磁通) —— Linear
* 1: Phi / u (速度流函数) —— Linear
* 2: Z_j (环向电流密度) —— Linear
* 3: omega (环向涡度) —— Linear
* 4: rho (背景质量密度) —— Log-Space $\to$ log1p
* 5: T_i (离子温度) —— Log-Space $\to$ log1p
* 6: v_par (平行速度) —— Linear
* 7: rho_n / rho_imp (杂质密度) —— Log-Space $\to$ log1p
* 8: T_e (电子温度) —— Log-Space $\to$ log1p

过往训练采用5通道数据进行训练；但根据 `Model 502` 的物理定义，完整的状态空间应包含上述8个通道。未来训练计划逐步引入 `rho_imp` 和 `T_e` 以满足 `Model 502` 的双温闭合要求。

## 8. Model 502 归一化（JOREK 源码依据，2026-09-07 核实）

以下关系取自 JOREK 源码 `models/mod_atomic_coeff_deuterium.f90:65-66` 与
`communication/mod_export_restart.f90`，是 pipeline 内部单位换算的权威来源：

- 密度：`rho_norm = central_density * 1e20 * central_mass * ATOMIC_MASS_UNIT` [kg/m^3]；
  `rho_SI = rho_norm * rho_JOREK`。
- 时间：`t_norm = sqrt(mu0 * rho_norm)` [s]；`t_SI = t_norm * t_now`。
- 速度：`v_norm = 1 / t_norm`（marker_spi_source.f90 的换算约定）。
- 每帧 HDF5 同时写出 `sqrt_mu0_rho0`（= t_norm）与 `sqrt_mu0_over_rho0`（= 1/t_norm），
  以及 `central_density/central_mass/F0/R_domm`，可用于量纲重建。
- 模型内部时间统一用 JOREK 单位（`dt = t_now_{n+1} - t_now_n`）；物理秒只在投影时间对齐、
  PDE 残差与外部报告处使用（乘 `t_norm`）。
- COCOS 符号约定只作用于 IMAS 导出接口（`communication/IMAS/mod_jorek2IMAS.f90` 的
  `fact_psi/fact_w/fact_phi/fact_zj`），不影响内部九场与源项的数值。
