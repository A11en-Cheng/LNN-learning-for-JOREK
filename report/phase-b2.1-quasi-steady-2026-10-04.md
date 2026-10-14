# Phase B2.1 交付报告：逐项度量权重 + 通道映射 + 边界复刻 + 准稳态保真（2026-10-04）

依据：ADR-0018（决定 5/6/7/11/12/13，尤其决定 13"稳定性 ≠ 保真度"）、
ADR-0019（配对语义）、`roadmap_v2.md` §4 Phase B2、§5 第 4/5/10/11 条、§6 B2 行、
§7 风险表（刚性模态被冻结 / 系数取值链 / 平行项度量权重与通道映射 / Mach-1 与角点项）、
`docs/jorek-normalization.md` §3.1–3.3、B0 报告 §9（边界语义确认表）、
B2 报告 §11（开放问题 #1–#6）。JOREK 源码基线：
`models/model502/mod_elt_matrix.f90`、`models/model502/mod_boundary_matrix_open.f90`、
`models/model502/mod_boundary_conditions.f90`、`matrix/construct_matrix_mod.f90`、
`grids/grid_utils/update_boundary_types.f90`、`models/preset_parameters.f90`、
`models/model502/initialise_parameters.f90`。

**未做**：训练/长窗口/闭环（Phase C）、A′ 细粒度数据接入、GNN/谱分支、全集 concat、
"无 condition"对照臂、legacy 测试改动、roadmap/ADR 改动、数据删除、git 操作。
`gh` 在本环境缺失（B0 已记录），未走 issue 流程。

## 0. 一句话结论

B2.1 把九场推进器从"隐式稳定"推进到"隐式稳定 + 准稳态保真"：**逐项度量权重**
（平行 `BigR/BB2`、垂直/粘性 `BigR`、`visco_num` 无权重）与 **`ZK_perp_num→T_i/T_e`
在内的通道-项映射**已按 `mod_elt_matrix.f90` 行号落地；时间积分默认改为
**逐通道稀疏直接 LU 的精确隐式解**（自定义 autograd，隐函数定理反传），
ZK∥/D∥ 等刚性模态在物理 `dt=0.05` 下被**真正求解**而不是冻结
（真实帧一步里 T_i 的 m-范数修正 1.42e-1，B2 阻尼迭代只有 9.69e-7，相差 1.5e5 倍）；
JOREK 单步弱形式对照、准稳态剖面速率、field-line 平均守恒、前向/反向稳定、
确定性与成本记录全部有测试；边界层补上 **corner-pair 跳过** 与
**Mach-1/鞘层线性化主要项**（`density_reflection`/`imp_reflection`/`gamma_sheath_*`/
`vpar_smoothing`/`mach_one_bnd_integral`/`visco_par_heating`/`rhs_ij_5..9`），
未复刻项逐条登记。全量 pytest **186 passed**（B2 基线 177，无回退）。

## 1. 改动文件

| 文件 | 内容 |
|---|---|
| `src/models/implicit_stencil.py` | 逐项度量权重（`TERM_METRIC_WEIGHT`）；17 项系数表和逐项 provenance（`ALPHA_PROVENANCE`）；逐行号通道-项映射（`DEFAULT_CHANNEL_TERMS`、`CHANNEL_TERM_PROVENANCE`）；`ZK_perp_num` 等新项装配；`dt=2/(αλ)` 表带 metric 列；求解器 `solver ∈ {direct, cg, richardson}`（默认 `direct`，稀疏 LU + 隐函数定理反传；质量内积 Jacobi-PCG；B2 阻尼迭代保留为对照）；`relax` 学习旋钮（恒等初始化）；成本记录 `last_solve_stats` |
| `src/models/jorek_boundary.py` | `SheathParameters`（preset/namelist 取值链 + `gamma_*_stangeby` 推导）；`boundary_edge_topology` + `edge_wall_side`（`construct_matrix_mod.f90:110-196`）；corner-pair 跳过；`_apply_open_boundary_terms`（`rhs_ij_5..9` + `amat_55/66/88/99` 对角线性化 + Mach-1 目标含 `1/Btot` 与极向流修正）；`_mach1_target` 与 BC 一致性报告更新；`BoundaryContext` 增 edge/sheath/pos/psi_gradient 字段 |
| `src/models/field_propagator.py` | `solver/cg_tol/max_cg_iters` 透传；`read_namelist_sheath_parameters`；`read_namelist_bc_flags` 增 `grid_to_wall`/`mach_one_bnd_integral`；`build_propagator_from_frame` 接线 sheath/F0/grid_to_wall；`LoadedPropagator.sheath`；参数量/条件深度报告增 solver |
| `tests/test_implicit_stencil.py` | 重写（20 用例）：逐项权重（含 `BigR/BB2 ≡ BigR` 代数等价）、通道映射（含 `ZK_perp_num`）、精确 λ/稳定界、三种求解器、真解判据（backward error）、准稳态剖面（速率/投影/守恒）、JOREK 单步对照、显式 cap 组件、真实帧 smoke |
| `tests/test_field_propagator.py` | 增 5 用例：solver 选择与成本记录、namelist 鞘层参数与推导、corner-pair 跳过、Mach-1 `1/Btot` 关系、全栈准稳态剖面 |
| `tests/README.md` | B2/B2.1 行更新（186 passed） |
| `docs/jorek-normalization.md` | §3.2/§6 增 2026-10-04 B2.1 勘误指针（逐项权重已落地、λ 表更新指向本报告） |
| `report/implementation-progress.md` | 本条目摘要 |

## 2. 逐项度量权重（替代全局 `metric_weight`）

JOREK 弱形式不是单一几何权重。**平行项**（`BigR/BB2 * Bgrad_a * Bgrad_b`）与
**垂直/粘性项**（`BigR * ∇·∇`）差一个 `BigR/BB2` 因子；`visco_num` 连 `BigR` 都不带。
B2.1 的装配约定（`src/models/implicit_stencil.py` 模块 docstring）：

```
BigR/BB2 * (B·∇N_i)(B·∇N_j) = BigR * (b̂·∇N_i)(b̂·∇N_j)      # Bgrad 用特斯拉量级的 B
```

即：平行项在**单位方向核**上取 `BigR` 权重（`BB2` 与 B 量级核抵消），垂直/粘性项取
`BigR`，`visco_num` 取 1。`R` 以 `R_REF = 1 m` 归一，权重为 O(0.46–2.0)。
测试 `test_parallel_weight_is_bigR_over_BB2_on_the_B_kernel` 用"B 量级核 × 字面
`BigR/BB2` 因子"独立复算并逐元素比对（1e-12）。

| 项 | 类别 | JOREK 源（`mod_elt_matrix.f90`） | α（JOREK 单位） | α 取值链 |
|---|---|---|---|---|
| `eta` | 无权重 | :1946-1947（`eta_num` 无权重；`eta_T` 含圆柱 `1/BigR`，未复刻，见 §7） | 1.443e-10 | namelist:391（=H5） |
| `visco` | `BigR` | :1492（u）、:2029（ω） | 2.4e-3 | namelist:381（=H5） |
| `visco_num` | 无权重 | :1496（u）、:2031（ω） | 9.0e-4 | namelist:385 |
| `D_perp` | `BigR` | :1551（`D_prof`，rho 物理垂直扩散） | 5.0e-6 | namelist:433 `D_perp(1)` |
| `D_perp_imp` | `BigR` | :1552（`D_prof_imp`，杂质） | 5.0e-6 | namelist:430 `D_perp_imp(1)` |
| `D_perp_num` | `BigR` | :1565（rho 数值垂直扩散） | 1.0e-12 | namelist:395 `d_perp_num` |
| `Dn_perp_num` | `BigR` | :1720（杂质数值垂直扩散） | 0.0 | preset:297（namelist 未设） |
| `ZK_i_perp` | `BigR` | :1596（`ZK_i_prof`，T_i 物理垂直热导） | 2.0e-6 | namelist:467 `ZK_i_perp(1)` |
| `ZK_e_perp` | `BigR` | :1755（`ZK_e_prof`，T_e） | 2.0e-6 | namelist:460 `ZK_e_perp(1)` |
| `ZK_perp_num` | `BigR` | :1598（T_i）、:1757（T_e） | 1.0e-12 | namelist:396 `zk_perp_num` |
| `visco_par` | `BigR` | :1641（v_par 垂直粘性） | 5.0e-4 | namelist:383（=H5） |
| `visco_par_num` | `BigR` | :1649（v_par 数值粘性） | 1.0e-4 | namelist:386 |
| `visco_par_par` | `BigR/BB2` | :1642（v_par 平行粘性，`r0_corr` 因子未复刻） | 0.0 | preset:49（namelist 未设） |
| `D_par` | `BigR/BB2` | :1546（rho） | 5.0e-2 | namelist:422 |
| `D_par_imp` | `BigR/BB2` | :1547/:1702（rho_imp） | 5.0e-2 | namelist:429 |
| `ZK_i_par` | `BigR/BB2` | :1595（T_i） | 1.6666e5（有效值受 `ZK_par_max=1e4` 上限） | namelist:456 |
| `ZK_e_par` | `BigR/BB2` | :1754（T_e） | 4.1037e6（同上） | namelist:457 |

**登记的口径差异**（不改变稳定性结论，见 §7）：① `ZK_*_par` 用未取上限的 namelist 值
（`ZK_par_max` 会把它压到 1e4）；② JOREK 平行项用 `(ZK_i_par_T − ZK_i_prof)`，
B2.1 不扣 profile 项；③ 温度依赖（`ZK ∝ T^{2.5}`、`visco/eta ∝ T^{-1.5}`）未随状态更新；
④ `visco_par_par` 的 `r0_corr` 密度因子未复刻。

## 3. 通道-项映射修订（补 `ZK_perp_num`）

`DEFAULT_CHANNEL_TERMS`（B2 缺 `ZK_perp_num`、`D_perp_num`、`visco_num`、profile 项）：

| 通道 | 项 | 代码行号（`mod_elt_matrix.f90`） |
|---|---|---|
| `psi` / `Z_j` | `eta` | :1946-1947（另加 `eta_num` :1946，ρ 与 Z_j 方程） |
| `Phi` | `visco`, `visco_num` | :1492, :1496（eq2 垂直动量） |
| `omega` | `visco`, `visco_num` | :2029, :2031（eq4 涡量定义） |
| `rho` | `D_perp`, `D_perp_num`, `D_par` | :1551（`D_prof`）, :1565, :1546 |
| `T_i` | `ZK_i_perp`, `ZK_perp_num`, `ZK_i_par` | :1596（`ZK_i_prof`）, :1598, :1595 |
| `v_par` | `visco_par`, `visco_par_num`, `visco_par_par` | :1641, :1649, :1642 |
| `rho_imp` | `D_perp_imp`, `Dn_perp_num`, `D_par_imp` | :1552, :1720, :1702 |
| `T_e` | `ZK_e_perp`, `ZK_perp_num`, `ZK_e_par` | :1755（`ZK_e_prof`）, :1757, :1754 |

（表中行号在代码里由 `CHANNEL_TERM_PROVENANCE` 逐条登记，
`test_channel_term_map_covers_all_nine_fields_and_matches_jorek` 断言存在性。）

## 4. 更新后的逐项 λ_max 与显式稳定界

**装配与谱**：各向同性核沿用 B0（与 `fem_weak.build_p1_laplacian` 逐元素相等），
平行核 `∫(b̂·∇N_i)(b̂·∇N_j)`；逐项乘度量权重后求 `λ_max(diag(m)^{-1}A_t)` 的
稀疏广义特征值（内部节点子集）。真实帧
`scan_1d22_ang15_5_massloss_neon_v400/jorek003000.h5`（4125 节点，198 边界节点，
内部 3927）：

| 项 | metric | α | λ_max（B2.1 逐项权重） | λ_max（B2 无权重，参考） | dt=2/(αλ) | dt=0.05 显式稳定 |
|---|---|---:|---:|---:|---:|---|
| `eta` | none | 1.443e-10 | 5.022959e6 | 5.0230e6 | 2.7593e3 | ✅ |
| `visco` | radius | 2.4e-3 | 6.097275e6 | 5.0230e6 | 1.3667e-4 | ❌ |
| `visco_num` | none | 9.0e-4 | 5.022959e6 | —（B2 未纳入） | 4.4241e-4 | ❌ |
| `D_perp` | radius | 5.0e-6 | 6.097275e6 | 5.0230e6（旧名） | 6.5603e-2 | ✅（临界） |
| `D_perp_imp` | radius | 5.0e-6 | 6.097275e6 | — | 6.5603e-2 | ✅（临界） |
| `D_perp_num` | radius | 1.0e-12 | 6.097275e6 | — | 3.2802e5 | ✅ |
| `Dn_perp_num` | radius | 0 | 6.097275e6 | — | ∞ | ✅（无贡献） |
| `ZK_i_perp` | radius | 2.0e-6 | 6.097275e6 | — | 1.6401e-1 | ✅ |
| `ZK_e_perp` | radius | 2.0e-6 | 6.097275e6 | — | 1.6401e-1 | ✅ |
| `ZK_perp_num` | radius | 1.0e-12 | 6.097275e6 | —（B2 未纳入） | 3.2802e5 | ✅ |
| `visco_par` | radius | 5.0e-4 | 6.097275e6 | 5.0230e6 | 6.5603e-4 | ❌ |
| `visco_par_num` | radius | 1.0e-4 | 6.097275e6 | — | 3.2802e-3 | ❌ |
| `visco_par_par` | parallel | 0 | 4.561210e6 | — | ∞ | ✅（无贡献） |
| `D_par` | parallel | 5.0e-2 | 4.561210e6 | 3.7604e6 | 8.7696e-6 | ❌ |
| `D_par_imp` | parallel | 5.0e-2 | 4.561210e6 | — | 8.7696e-6 | ❌ |
| `ZK_i_par` | parallel | 1.6666e5 | 4.561210e6 | 3.7604e6 | 2.6310e-12 | ❌（刚性） |
| `ZK_e_par` | parallel | 4.1037e6 | 4.561210e6 | 3.7604e6 | 1.0685e-13 | ❌（刚性） |

结论：`BigR`/`BigR/BB2` 权重使径向/平行谱相对 B2 的无权重代理各升约 21%
（5.023e6→6.097e6、3.760e6→4.561e6），**不改变任何稳定性结论**：
η 显式宽松（2.76e3），D_perp/profile 临界但可显式（6.56e-2 > 0.05），
visco/visco_par/D_par/ZK∥ 在物理 dt=0.05 下必须隐式。
与 `docs/jorek-normalization.md` §3.2 的差异仅为逐项权重与新增项。

## 5. 隐式求解器与准稳态保真（核心）

### 5.1 三种求解器

| `solver` | 机制 | 可微性 | 状态 |
|---|---|---|---|
| `direct`（默认） | 逐通道稀疏 LU（SuperLU），行替换式 Dirichlet；自定义 `torch.autograd.Function`，反传用 `M^T` 伴随解 + 隐函数定理给出 `∂L/∂α`、`∂L/∂rhs`、`∂L/∂ref` | 一阶可微、确定性 | 交付默认 |
| `cg` | 质量内积下 Jacobi 预条件 CG（stencil matvec，torch 图内） | 完全在图中 | 备选；真实帧 ZK∥ 下 Jacobi 预条件数 ~1e9：12 迭代残差 3.4e-2、200 迭代 9.3e-3（1.9 s，仍不收敛），仅合成/各向同性场景可用 |
| `richardson` | B2 对称主元阻尼迭代（`1−ω/D_c` 收缩、任意 dt 稳定） | 是 | 对照/稳定性测试；ZK∥ 收缩率 1−8.5e-13，实测"冻结" |

学习参数：逐项/逐通道 `alpha_scale`（恒等初始化）+ 逐通道 `relax`（0 初始化，界 0.5，
对三种求解器都给出可训练但初始恒等的过/欠松弛）；`damping` 只在 `richardson` 模式存在
（避免未使用参数）。

### 5.2 准稳态验收（测试证据）

| 验收 | 测试 | 结果 |
|---|---|---|
| a. 沿磁力线剖面按正确速率松弛（合成细网格，b̂∥Z，ZK_i_par） | `test_parallel_profile_relaxes_at_the_correct_rate` | 模态衰减 = `1/(1+dt·α·λ)`（rel 1e-8），与稠密隐式解逐元素一致（rtol 1e-10）；`decay < 0.1`（非冻结） |
| a′. 刚性极限 = 磁场线不变子空间投影（不零化、不冻结） | `test_stiff_parallel_profile_projects_to_the_null_space_not_frozen` | `‖x−P x0‖/‖x0‖ < 1e-3`，近零空间分量保留、扰动衰减；质量加权平均守恒（<1e-8·std） |
| a″. B2 阻尼迭代"能稳不能保真"（对照） | `test_richardson_solver_is_stable_but_frozen_on_the_parallel_profile` | k=1 本征模：direct 衰减至 1/(1+dtαλ)，richardson 12 步后 >100× 更大；真实帧 T_i 一步 m-范数修正 direct 1.42e-1 vs richardson 9.69e-7（**1.5e5 倍**） |
| b. 与 JOREK 单步隐式解对照（弱形式直接求解） | `test_jorek_single_step_reference_matches_scipy_direct_solve`、`test_direct_solver_matches_the_dense_exact_solution`、真实帧 `test_real_frame_quasi_steady_solve_and_rollout_smoke` | rho/T_i/T_e 与 SciPy `spsolve`/稠密解 rtol 1e-10；真实帧 backward error：rho 1.1e-16、T_i 3.7e-14、T_e 1.1e-12（稠密参考 1.1e-12） |
| c. 前向/反向稳定、确定性、梯度有限 | `test_solver_determinism_finite_gradients_and_cost_record`、`test_propagator_is_deterministic_and_differentiable`、`test_real_frame_*` | 三种 solver 同输入同输出（`torch.equal`）；逐参数梯度有限（direct 全体参数非 None 且有限）；4 层真实帧 rollout 有界、Dirichlet 行逐元素保持 |
| d. 成本记录（无 performance gate） | `test_propagator_solver_selection_and_cost_record`、`last_solve_stats` | direct：`matrix_nnz=608566`、factorization+solve ≈ 6.1e-2 s/次（4052 cluster、9 通道，CPU）、反传 ≈ 1.1e-2 s；同一 `(dt, α)` 的重复求解命中因子缓存后 ≈ 1.6e-4 s（38×，rollout/推理路径）；cg：`iterations`/`residual_relative`/`converged`；richardson：`iterations`/`residual_relative` |

**为什么"稳定 ≠ 保真"**：B2 的阻尼迭代收缩率由最大特征值 `D_c≈dt·α·λ_max` 决定，
近零空间模（λ 小）的收缩率 ≈ `1−ω λ/D_c → 1`，在有限迭代下**原样保留**；真实帧
`contraction_factor(T_e) = 1−8.5e-13`。直接 LU 求解没有这个截断，一步就把
ZK∥ 模态解到位（见上表 a″）。这就是 ADR-0018 决定 13 要求升级求解器的原因。

## 6. 边界层复核：corner-pair 与 Mach-1/鞘层复刻状态

### 6.1 分支与角点（按 B0 §9 确认表 + 源码）

- `1/11/5/15` TARGET、`9/19` CORNER free → `apply_cs`；`2/12` TANGENT、
  `20/21` CORNER fixed/inverted → `apply_dirichlet_all`；`3/4` 旧码按
  `mod_boundary_conditions.f90:227-248` 分支处理（3 同时进两分支）。
  映射表与逐场施加沿用 B0，未变。
- **corner-pair 跳过**：新增 `boundary_edge_topology` / `edge_wall_side`，
  按 `matrix/construct_matrix_mod.f90:99-196` 复刻：边界边 = 两端 boundary≠0 的
  单元边；`grid_to_wall` 分支下角点对（两端都在 `{3,9,19,20,21}`）→ `cycle`；
  非 `grid_to_wall` 分支用 `else` 的组合白名单（(1|3,1|3)、(1|9,1|9)、(4|9,4|9)、
  (1|4,4|1)、(5|9,5|9)、(2|3,2|3)），其余同 `cycle`。
  生产 namelist `grid_to_wall = .t.`（`jorek_namelist_spitest:142`），
  v400 真实帧 198 条边界边中 2 条被跳过；合成全角点网格 16/16 全部跳过
  （`test_corner_pair_edges_skip_the_sheath_integration`）。
- 未登记 boundary 码仍然 fail-fast；`{9,19,20,21}` 需显式 acknowledge 并登记。
  **保留 pending 的理由**：B2.1 已按 §9 表逐条复核并实现分支，"待确认"仅剩两处无帧级
  交叉核对的对象——`direction=sign(B·n)` 的节点级符号（B0 §6 code 1/11/5/15）
  与 20/21 的 `grid_to_wall` side 指派（无 `grid_to_wall` 输出可核对）。

### 6.2 Mach-1 / 鞘层（`bc_natural_open` 分支）逐项状态

| JOREK 项 | 源码行 | B2.1 状态 |
|---|---|---|
| 1° 角粒子/能量通量 `c_angle`（`c_angle=1°=0.0174524`） | `mod_boundary_matrix_open.f90:62,288-305` | ✅ 复刻；`min_sheath_angle` 从 namelist 读入（生产 1°） |
| `density_reflection` / `imp_reflection`（守恒 + 对角线性化） | :288,292-294,299-305,362-416 | ✅ 复刻（RHS 源 + 对角项，比值限幅保证不翻号） |
| `gamma_sheath_i` / `gamma_sheath_e`（`(γ−1)` 能量权重） | :292-295,303-305,374-387,404-416 | ✅ 复刻；由 `gamma_*_stangeby` 推导（`initialise_parameters.f90:241-251`；生产 8.0 → γ_i=3.556、γ_e=4.667） |
| `−r·R²·u0_s·normal_sign3`（垂直流反射） | :290,294,301,305 | ✅ 复刻（节点对边平均，1/m 归一） |
| Mach-1：`v_par = dir·factor·c_s/Btot + factor·R²·U_b/ps0_b` | `mod_boundary_conditions.f90:448,511-528` | ✅ 主项 `1/Btot` 复刻（修正 B0 的 `direction*c_s`）；极向流修正项复刻并**限幅到声速项量级**（B2.1 组件级 boundedness guard）；无 ψ 梯度（合成态）时修正项置 0 并登记 |
| `vpar_smoothing`（tanh 因子） | `mod_boundary_conditions.f90:466-486`、open:268-272 | ✅ 复刻（`c1/c2/c3` 从 namelist；节点级近似"启用即平滑"，元素级 `bn_1·bn_2<0` 判定未复刻） |
| `mach_one_bnd_integral` 开关 | open:274-275,317,440-441；`mod_boundary_conditions.f90:370` | ✅ 门控复刻：`=T` 时 open 分支的 `rhs_ij_7/amat_76/77` 生效；`=F`（生产预设）时由 `mod_boundary_conditions.f90` 的罚行给出——两分支收敛到同一关系，本实现用同一目标函数并登记分支 |
| `visco_par_heating`（eq6 源项） | :295,386-387 | ✅ 复刻（生产 =0；法向导数用单侧邻居近似，符号约定登记） |
| `neutral_source` | open:195-202 | ⚪ **JOREK 源码中计算后未被 `rhs_ij_5..9` 使用**（且 `neutral_line_source` 默认 0，preset:356）⇒ 无对应项，登记为 no-op |
| 鞘层通量弱形式归一化（`ws`/基函数积分 vs 节点集总） | open:277-319 | ⚠️ 节点级近似（`Σ_edges ½dl_e`），量级/符号一致，未逐高斯点装配 |
| 离对角线性化耦合 `amat_51/52/56/57/61/62/65/67/76/77/81/82/86/87/91/92/95/96/97` | :362-416 | ⚠️ **未复刻**（跨场/跨节点 Jacobian）；已实现对 `amat_55/66/88/99` 的**对角部分** + RHS。影响：边界节点的隐式耦合弱化，可能影响边界层的收敛速度与剪切响应，不影响内部 9 场演化 |
| `freeboundary=T` 真空耦合 | `mod_vacuum_fields.f90` | ⚠️ 本数据 F，未触发（沿用 B0 单侧近似） |
| `use_simple_bnd_types` 折叠 | `update_boundary_types.f90:665-676` | ✅ 沿用 B2（11→1、15→5、19→9、12→4、20/21→9），从 namelist 读入 |

### 6.3 未复刻项的影响评估

1. **`amat` 离对角耦合**（中）：角落/靶板节点的隐式块被对角化，边界值仍满足
   Dirichlet/Mach-1/通量关系（有断言），但边界与内部的隐式耦合响应偏差未量化；
   对内部演化影响有限（边界行不参与内部写入），对边界层精细结构的保真度未知。
2. **极向流修正限幅**（中低）：限幅到 ±|c_s/Btot| 是组件级 boundedness guard；
   真实 JOREK 的修正由完整矩阵隐式解出，可能超出该范围。
3. **节点级 vs 高斯点弱形式**（中低）：鞘层 RHS 的能量/粒子通量系数一致，
   但基函数积分精度未逐点复刻。
4. **`vpar_smoothing` 元素级判定**（低）：只按节点启用，未做 `bn_1·bn_2<0` 与
   导数项（`factor_b/factor_bb`）。

## 7. 开放问题（按风险排序）

1. **真实帧/生产系数下的准稳态剖面直接验证仍缺**（中高）：B2.1 的准稳态剖面
   验收在合成细网格（解析/半解析参考 + 稠密对照）完成；真实帧因 T_i/T_e 已
   99.97% 落在平行近零空间，一步"剖面松弛"不可分辨，只能用 backward error +
   修正量对照（direct 1.42e-1 vs richardson 9.7e-7）。建议用 Phase A 的细粒度
   `(u_before, S_used, u_after)` 帧做"JOREK 真值单步 vs 本求解器"的逐通道对照。
2. **`ZK_par_max` 上限与温度依赖未进模型**（中）：α 表用未取上限的 namelist 值，
   实际有效 `ZK∥ ≤ 1e4`；`ZK ∝ T^{2.5}`、`visco/eta ∝ T^{-1.5}` 需在 Phase C
   前后接状态。当前 λ 表结论不受影响（上限只会降低刚度）。
3. **CG 未收敛**（中）：真实帧并行算子的近零空间（112 维 / 4052）使 Jacobi
   预条件 CG 200 迭代后残差仍 9.3e-3（1.9 s，比精确 LU 的 78 ms 慢 24×）。
   若要在 GPU 上避免 LU，需场向线预条件/多重网格；当前 direct 为 CPU 稀疏 LU，
   已是 JOREK 同类的精确解。
4. **Mach-1 离对角线性化未复刻**（中，见 §6.3）：建议后续按 `amat_*` 逐项装配
   边界元块（或最小二乘近似），并补边界层对照。
5. **η 的圆柱度量与 `eta_num`**（低中）：B2 保留无权重 P1 Laplacian；
   `:1947` 的 `1/BigR` 与 `:1946` 的 `eta_num=5e-10` 未复刻（相对权重 O(1/R)，
   不改变稳定性结论）。`D_prof`/`ZK_*_prof` 已按行 1551/1552/1596/1755 首次纳入，
   但 profile 形状用 namelist 首值近似。
6. **真实帧默认 n_iters 与 direct 无关**（低）：`n_iters` 只作用于 cg/richardson；
   报告 `parameter_report()["n_iters"]` 在 direct 下仅作历史字段保留。
7. **freeboundary=T**（低）：本数据 F，未触发；真空耦合仍未实现（同 B0 §6）。
8. **`bc_natural_open=F` 分支未实现**（低）：B2.1 的 BC 层实现的是开放磁力线
   自然/鞘层分支（生产 `bc_natural_open=T`）；`=F` 时 JOREK 走固定边界逻辑，
   本层未复刻，遇到该配置应显式拒绝/登记（后续补）。
9. **`B2.1 组件级限幅`**（低）：鞘层源/极向流修正的 boundedness guard 是组件级
   约定，接入 Phase C 闭环时需复核是否引入偏差。

## 8. 测试命令与结果

```bash
.torch-cuda-env/bin/python -m pytest tests/test_implicit_stencil.py tests/test_field_propagator.py -q
# 42 passed
.torch-cuda-env/bin/python -m pytest tests/test_stencil_operator.py tests/test_stencil_boundary.py -q
# 16 passed
.torch-cuda-env/bin/python -m pytest -q
# 186 passed, 2 warnings in 84.1s   （B2 基线 177 → +9，无回退）
```

真实帧测试在 `data/EHL-2/scan_1d22_ang15_5_massloss_neon_v400/jorek003000.h5`
存在时执行（本环境存在），否则 skip-if-absent。帧级 rollout 仍只是 **smoke**：
记录帧间距 ~10 个流体步，不是基线也不是 gate（ADR-0018 决定 4/10）。

## 9. 与 B2 报告的差异速查

| B2 报告 §11 | B2.1 状态 |
|---|---|
| #1 边界码确认/角点处理 | ✅ corner-pair 跳过 + `edge_wall_side` 白名单复刻（§6.1） |
| #2 刚性极限精度（准稳态保真） | ✅ direct LU 精确解；准稳态剖面/投影/守恒测试通过（§5.2） |
| #3 系数取值链 | ✅ 以 H5 + `jorek_namelist_spitest` 为准；`d_perp_num/zk_perp_num/visco_num/visco_par_num` 进 `from_h5_and_namelist` |
| #4 逐项算子口径（度量权重） | ✅ `TERM_METRIC_WEIGHT` + `BigR/BB2 ≡ BigR` 等价测试（§2） |
| #5 通道-项映射（`ZK_perp_num`） | ✅ `DEFAULT_CHANNEL_TERMS` + 行号 provenance（§3） |
| #6 Mach-1/鞘层线性化 | 🟡 主要项复刻、离对角耦合登记缺口（§6.2/§6.3） |
| #7 FiLM per-channel | ⏸ 未启用（默认全局标量），保持 B2 结论 |
| #8 LTC target 有界性 | ⏸ 未变（训练期约束待 Phase C） |
| #9 freeboundary=T | ⏸ 未触发 |
| #10 帧级 smoke 定位 | 同 B2（仍非基线） |
