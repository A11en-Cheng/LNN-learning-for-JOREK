# JOREK Reduced MHD Physics Definitions & Normalization

本文件定义了 HyPhy-GLNN 模型中使用的所有物理量的归一化标准。这些标准严格对应 JOREK 内部的无量纲化方案（主要基于 Alfvén 标度）。

## 1. 基础参考量 (Reference Quantities)
所有物理量均基于以下三个基本参考量进行无量纲化：
- **真空环向磁场 ($B_0$)**: 磁轴处的环向磁场强度 [Tesla]
- **大半径 ($R_0$)**: 磁轴处的几何大半径 [m]
- **中心密度 ($\rho_0$)**: 初始时刻的中心质量密度 [kg/m³]

## 2. 导出参考量 (Derived Quantities)
- **真空磁导率 ($\mu_0$)**: $4\pi \times 10^{-7}$ [H/m]
- **Alfvén 速度 ($v_A$)**: $v_A = \frac{B_0}{\sqrt{\mu_0 \rho_0}}$ [m/s]
- **Alfvén 时间 ($\tau_A$)**: $\tau_A = \frac{R_0}{v_A} = \frac{R_0 \sqrt{\mu_0 \rho_0}}{B_0}$ [s]
- **参考压力 ($p_0$)**: $p_0 = \frac{B_0^2}{\mu_0}$ [Pa]
- **参考温度 ($T_0$)**: $T_0 = \frac{p_0}{\rho_0}$ (假设理想气体)

## 3. 核心状态变量 (Primary State Variables)
网络输入和输出的 5 个核心变量（遵循 Reduced MHD Ansatz）：
1. **极向磁通 ($\psi$)**: 归一化因子 $\psi_{norm} = R_0^2 B_0$
2. **流函数 ($\Phi$)**: 归一化因子 $\Phi_{norm} = R_0 v_A = \frac{R_0^2}{\tau_A}$
3. **密度 ($\rho$)**: 归一化因子 $\rho_{norm} = \rho_0$
4. **温度 ($T$)**: 归一化因子 $T_{norm} = T_0$
5. **平行速度 ($v_{\parallel}$)**: 归一化因子 $v_{norm} = v_A$

## 4. 物理约束 (Hard Constraints)
模型**绝不**直接预测磁场 $\mathbf{B}$ 或速度场 $\mathbf{v}$。必须通过以下 Ansatz 计算：
- **磁场**: $\mathbf{B} = \frac{F_0}{R}\hat{\phi} + \frac{1}{R}\nabla \psi \times \hat{\phi}$ (其中 $F_0 = R_0 B_0$)
- **速度场**: $\mathbf{v} = -R \nabla \Phi \times \hat{\phi} + v_{\parallel} \mathbf{B}$

## 5. 物理量汇总表 (JOREK Official Normalization)

*   下标 "$_{SI}$" 表示 SI 单位制下的物理量，而没有下标的物理量表示 JOREK 内部的无量纲化单位。
*   注意：重要的归一化因子 $\sqrt{\mu_0/\rho_0}$ 和 $\sqrt{\mu_0\rho_0}$ 会被写入 JOREK 的 logfile 中。为了使这些值正确，需要在输入参数中设置 `central_density` 和 `central_mass`。

| SI 单位 | 归一化单位 (JOREK) | 物理量描述 |
| :--- | :--- | :--- |
| $R_\text{SI}~[m]$ | $=R$ | 大半径 (Major radius) |
| $Z_\text{SI}~[m]$ | $=Z$ | 垂直坐标 (Vertical coordinate) |
| $\mathbf{B}_\text{SI}~[T]$ | $=\mathbf{B}$ | 磁场矢量 (Magnetic field vector) |
| $\mathbf{E}_\text{SI}~[V m^{-1}]$ | $=\mathbf{E}/\sqrt{\mu_0\rho_0}$ | 电场矢量 (Electric field vector) |
| $\Psi_\text{SI}~[T m^2]$ | $=\Psi$ | 极向磁通 (Poloidal magnetic flux) |
| $j_{\phi, \text{SI}}~[A m^{-2}]$ | $=-j/(R\;\mu_0)$ | 环向电流密度 (Toroidal current density); $j_{\phi, \text{SI}}=\mathbf{j}_{\text{SI}}\cdot\hat{e}_\phi$ |
| $n_\text{SI}~[m^{-3}]$ | $=\rho\;n_0$ | 粒子密度 (Particle density) |
| $\rho_\text{SI}~[kg~m^{-3}]$ | $=\rho\;\rho_0$ | 质量密度 (Mass density) = 离子质量 $\times$ 粒子密度 |
| $T_\text{SI}~[K]$ | $=T/(k_B\;\mu_0\;n_0)$ | 温度 (Temperature) = 电子温度 + 离子温度 |
| $T_{eV}~[eV]$ | $=T/(e\;\mu_0\;n_0)$ | 以 eV 为单位的温度 |
| $p_\text{SI}~[N m^{-2}]$ | $=\rho\;T/\mu_0$ | 等离子体压力 (Plasma pressure) |
| $\mathbf{v}_\text{SI}~[m s^{-1}]$ | $=\mathbf{v}/\sqrt{\mu_0\rho_0}$ | 速度矢量 (Velocity vector) |
| $v_{\parallel, \text{SI}}~[m s^{-1}]$ | $=v_{\parallel}\cdot B_\text{SI}/\sqrt{\mu_0\rho_0}$ | 平行速度分量 (Parallel velocity component), 其中 $B_\text{SI}=|\mathbf{B}_\text{SI}|$ |
| $u_\text{SI}~[m s^{-1}]$ | $=u/\sqrt{\mu_0\rho_0}$ | $R u$ 是速度流函数, $F_0 u$ 是势 |
| $\omega_{\phi, \text{SI}}~[m^{-1} s^{-1}]$ | $=\omega/\sqrt{\mu_0\rho_0}$ | 环向涡度 (Toroidal vorticity) |
| $t_\text{SI}~[s]$ | $=t\cdot\sqrt{\mu_0\rho_0}$ | 时间 (Time) |
| $\gamma_\text{SI}~[s^{-1}]$ | $=\gamma/\sqrt{\mu_0\rho_0}$ | 增长率 (Growth rate) |
| $\eta_\text{SI}~[\Omega m]$ | $=\eta\cdot\sqrt{\mu_0/\rho_0}$ | 电阻率 (Resistivity) |
| $\mu_\text{SI}~[kg~m^{-1} s^{-1}]$ | $=\mu\cdot\sqrt{\rho_0/\mu_0}$ | 动力粘度 (Dynamic viscosity) |
| $\nu_\text{SI}~[m^2 s^{-1}]$ | $=\mu_\text{SI}/\rho_{SI}$ | 运动粘度 (Kinematic viscosity) |
| $D_\text{SI}~[m^2 s^{-1}]$ | $=D/\sqrt{\mu_0\rho_0}$ | 粒子扩散率 (Particle diffusivity) |
| $K_\text{SI}~[kg~m^{-1} s^{-1}]$ | $=K\cdot\sqrt{\rho_0/\mu_0}/\left(\gamma-1\right)$ | 热扩散率 (Heat diffusivity) |

### 基础定义 (Base Definitions)
| 变量 | 定义 | 备注 |
| :--- | :--- | :--- |
| $n_0~[m^{-3}]$ | $= \text{central\_density} \cdot 10^{20}$ | `central_density` 在输入文件中指定 |
| $\rho_0~[kg\;m^{-3}]$ | $= \text{central\_mass} \cdot n_0 \cdot m_\text{AMU}$ | `central_mass` 在输入文件中指定 |
| $\gamma$ | $= \text{GAMMA}$ | 绝热指数，默认值为 $5/3$ |

### 常用物理常数 (Useful Constants)
\begin{align*}
m_{e0}        &= 0.911\cdot10^{-30}kg \\
m_{AMU}       &= 1.661\cdot10^{-27}kg \\
m_{deuterium} &= 2.014101777811 AMU \\
m_{tritium}   &= 5.007\cdot10^{-27}kg \\
\mu_0         &= 4\cdot\pi\cdot10^{-7}Vs/(Am) \\
e             &= 1.602176565\cdot10^{-19}C
\end{align*}
