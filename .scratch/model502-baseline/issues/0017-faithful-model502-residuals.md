# 0017 保真 Model 502 七方程残差（蓝本：mod_elt_matrix + marker_spi_source）

Blocked by: #0009, #0011

目标：把 `src/physics/model502_physics.py` 的约化残差替换为保真的七个预后场方程，
在 **JOREK 归一化单位**下逐项构造，作为 physics-informed loss 的主项。

## 蓝本与方程（definitions.md §1.3，对照 mod_elt_matrix(_fft).f90 与 marker_spi_source.f90）

状态先 destandardize + inverse_transform 回 JOREK 单位（`stats.mean/std`）。
导出量 `Z_j = Δ*ψ`、`omega = Δu` 由 `reconstruct_derived` 提供。

1. 感应（ψ）：`dψ/dt = η(Te)·Δ*ψ − R{u,ψ} − F0 ∂u/∂φ`（轴对称 ∂/∂φ=0）。
   项：Δ*ψ、Poisson bracket {u,ψ}、电阻 η（条件通道第 0 位）。
2. 涡度（ω=Δu）：definitions §1.3 eq.2（含 {j,ψ}、{P,R²}、粘性 η∇²ω、F0∂j/∂φ）。
   注意 ω 本身由 u 重建，残差作用于 u 的演化。
3. 密度（ρ）：`dρ/dt = −∇·(ρv) + ∇·(D_⊥ ∇ρ) + S_D + S_imp`。
   源：mod_sources 背景 particle_source + marker 投影。
4. 离子压强（Ti）：`d[(n_bg+n_imp)Ti]/dt = 对流 − p·∇·v + ∇·(κ_∥∇_∥Ti + κ_⊥∇_⊥Ti)
   + 0.5|v|²(S_D+S_imp) + n_i(dTi/dt)_e`。
5. 电子压强（Te）：`d(ne Te)/dt = 对流 − p·∇·v + η j²/R² + ∇·(κ∇Te)
   − (γ−1) ne n_imp P_rad + n_e(dTe/dt)_i + S_ion`。
6. 平行动量（v_par）：`B² d(ρ v_par)/dt = ... − (1/R){ψ,P} − B² ∇·(ρv) v_par + B² μ_∥ Δ_⊥ v_par`。
7. 杂质（ρ_imp）：`dρ_imp/dt = −∇·(ρ_imp v) + ∇·(D_imp ∇ρ_imp) + S_imp`。

## 算子清单

- `Δ*ψ = R² ∇·(R⁻²∇ψ)`、`Δu = ∇·(∇u)`（已有 grad_ops）。
- Poisson bracket `{A,B} = (1/R)(A_R B_Z − A_Z B_R)`（cocos8.poisson_bracket）。
- `B`（get_magnetic_field）、`v`（get_velocity_field）、∇、∇·（grad_ops）。
- 平行/垂直梯度、P_rad（ADAS，来自 #0014 表）、κ_∥/κ_⊥、μ_∥、D_⊥ 由条件通道/参数表取。

## 验收

- 逐方程单测：在合成/真实帧上残差有限；对静止/解析解残差趋零（构造可解析的简化场验证算子）。
- 与 JOREK 源码逐项对照表落盘（report/model502-residual-map.md）。
- 重跑 λ_phys 扫描，物理项开始随 λ_phys 单调影响 val loss。

