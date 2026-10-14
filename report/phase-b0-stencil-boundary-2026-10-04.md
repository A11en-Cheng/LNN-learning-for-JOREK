# Phase B0 交付报告：stencil-conv 算子 + JOREK 边界条件层（2026-10-04）

依据：ADR-0018 决定 5/11、`roadmap_v2.md` §4 Phase B0/B2、§5 第 4/10 条、
§6 B0 行、§7 "边界 BC 缺失/错配"风险行；JOREK 源码
`models/model502/mod_boundary_conditions.f90`、
`models/model502/mod_boundary_matrix_open.f90`、
`grids/grid_utils/update_boundary_types.f90`、
`communication/mod_import_gvec.f90`。

未做：训练、GNN/谱分支、细粒度数据 loader（Phase A 另派）、legacy 测试改动、
roadmap/ADR 改动、数据删除。`gh` CLI 本环境缺失（`which gh` → not found），
未认领 B0 issue，按实测记录跳过。

真实帧：`data/EHL-2/scan_1d22_ang15_5_massloss_neon_v400/jorek003000.h5`（4125 节点 /
3986 单元），对比帧 2750/3700/4650；namelist `jorek_namelist_spitest` 中
`freeboundary = .f.`、`bc_natural_open = .true.`（开放磁力线 sheath 走
`mod_boundary_matrix_open.f90`）。

## 1. 改动文件

| 文件 | 内容 |
|---|---|
| `src/models/stencil_conv.py` | 新增：FEM 几何装配（`size`/`vertex`/`contain_node`/`neighbours`/`x`）、P1 局部 4×4 stencil matmul、稀疏弱形式算子、`delta_star`/`delta_plus`、`StencilConv`/`StencilConvStack`、显式稳定步长界、几何/邻接自检 |
| `src/models/jorek_boundary.py` | 新增：boundary 代码→BC 映射表、逐节点 mask/normal/单侧邻居、Dirichlet hold、Neumann/free 单侧、Mach-1 sonic + 鞘层通量汇、NaN 预处理、BC 一致性报告 |
| `tests/test_stencil_operator.py` | 新增：合成 mesh 几何/线性场零空间/守恒/确定性/稳定 cap；真实帧离散关系与 JOREK 邻接一致性（skip-if-absent） |
| `tests/test_stencil_boundary.py` | 新增：映射表钉死、合成 mask/normal/单侧邻居、Dirichlet hold、free/`no_mach1_bc`、60 层边界一致、边界 NaN 不向内传播；真实帧 198 节点/60 层/NaN（skip-if-absent） |
| `tests/stencil_fixtures.py` | 新增：合成四边形 mesh、边界代码分配、磁轴塌缩 fan mesh、确定性状态 |
| `tests/README.md` | 测试分组表新增 B0 两行 |
| `report/implementation-progress.md` | 新增本条目摘要 |

## 2. stencil 算子口径与证据

装配与 `src/physics/fem_weak.py` 完全同源：每个四边形拆两个 P1 三角形
（`(0,1,2)`/`(0,2,3)`），共点节点合并（= JOREK 轴节点共享 dof，
`communication/mod_import_gvec.f90:387-404`），`|J|≈0` 三角形丢弃。单元贡献嵌入
4×4 局部核 `element_kernel`，`StencilConv` 对 9 场做
`msg[e,i] = Σ_j K[i,j] x[vertex_j(e)]`、按集总质量散射回节点：
恒等初始化时严格等于已验证弱形式 `-A x / m`（保守、常场零空间）。

真实帧 jorek003000（`size`/`vertex` 同帧校验）：

| 检验 | 结果 |
|---|---|
| 节点 / 单元 / cluster | 4125 / 3986 / 4052（68 个共点节点合并） |
| 本模块 A/m/cluster vs `fem_weak.build_p1_laplacian` | 逐元素相等（max abs diff **0.0 / 0.0**，cluster 全等） |
| 单元核行和 / 对称性 | 7.1e-15 / 0.0 |
| 整体刚度对称性 / nnz / mass>0 | 0.0 / 27962 / 是（min 3.03e-5） |
| `neighbours` vs `vertex` 共享边邻接 | 15844/15876 匹配（99.80%）；68 个共点边另计；32 处失配全部在 X 点节点（见 §6） |

离散关系（值 dof 节点值 + Hermite dof 梯度，边界码非零节点剔除后 n=3927，
中心化后 corr/relL2；口径与 `check_fem_second_order.py` 相同）：

| 帧 | `Z_j ≈ Δ*ψ` | `ω ≈ Δ+Φ` |
|---|---|---|
| 2750 | corr 0.9921 / relL2 0.1326 | corr 0.9979 / relL2 0.0647 |
| **3000** | **corr 0.9921 / relL2 0.1326**（rms 2.263 vs 2.382） | **corr 0.9975 / relL2 0.0702**（rms 7.20e-5 vs 7.26e-5） |
| 3700 | corr 0.9920 / relL2 0.1326 | corr 0.9963 / relL2 0.0885 |
| 4650 | corr 0.9927 / relL2 0.1257 | corr 0.9970 / relL2 0.0804 |

参照口径（roadmap B0/2026-09-30 条目）为 corr 0.994/0.997、relL2 0.113/0.080；
同帧同口径的图 LSQ 算子是 corr 0.9872/0.9931、relL2 0.1740/0.1206 ⇒
本 FEM 算子不劣于图 LSQ，差异为帧与离散误差。测试阈值取 corr>0.98、relL2<0.15。

确定性/守恒：同输入同输出（`torch.equal`）；恒等核下单层 cluster 空间质量漂移
5.2e-15（合成 mesh −3.0e-17）、常场严格保持；无任何随机初始化。
显式稳定界：Gershgorin `dt ≤ 2/λ_max`，jorek003000 上 λ 上界 5.75e6（单位 1/m²）。
**注意单位**：`1.74e-7` 是算子系数 α=1（单位 m²/JOREK 时间）时的**几何界**（代码取
`dt_eff = 0.5·(2/λ_max) = 1/λ_max`），不能直接和 JOREK 时间步 0.05 比较。物理显式界
应为 `dt_term = 2/(α_term·λ_max)`，α 取该方程的输运系数（namelist，单位 m²/JOREK 时间）。
用本 run 的系数折算：

| 项 | α（namelist） | dt=2/(α·λ_max) | dt=0.05 显式稳定？ |
|---|---:|---:|---|
| η（磁扩散） | 1.443e-10 | 2.4e3 | 是 |
| D_perp | 5e-6 | 7.0e-2 | 是（临界） |
| visco_par | 5e-4 | 7.0e-4 | 否 |
| visco | 2.4e-3 | 1.4e-4 | 否 |
| D_par | 5e-2 | 7.0e-6 | 否 |
| ZK_par_max / ZK_i_par / ZK_e_par | 1e4 / 1.67e5 / 4.10e6 | 3.5e-11 / 2.1e-12 / 8.5e-14 | 否（刚性） |

结论不变但必须按项陈述：**η 扩散显式可稳定；平行热传导/输运（ZK/D_par/visco）
在物理 dt=0.05 下远超显式稳定界，必须隐式/半隐式**。B0 的 cap 只限制显式层间推进，
不改稳态；B2 的时间积分决策见 roadmap B2 与 ADR-0018 决定 13。
完整换算链（含输运系数的 SI 换算与代码行号）见 `docs/jorek-normalization.md`。

## 3. BC 映射表（代码 → 物理边界段 → 施加方式）

| code | 名称 | side/方向 | Mach-1(apply_cs) | Dirichlet-all | 来源 |
|---|---|---|---|---|---|
| 0 | interior | − | F | F | `update_boundary_types.f90:60` |
| 1 | TARGET side2（inward） | 2 / inward | T | F | `update_boundary_types.f90:636`；`mod_boundary_conditions.f90:226-245` |
| 2 | TANGENT side3（tangent field） | 3 / tangent | F | T | `update_boundary_types.f90:642`；`mod_boundary_conditions.f90:226-233` |
| 3 | CORNER 1-2（旧 grid） | corner / inward | T | T | `update_boundary_types.f90:647`；`mod_boundary_conditions.f90:226-245` |
| 4 | TARGET side2（旧 grid，同 1） | 2 / inward | T | F | `update_boundary_types.f90:648` |
| 5 | TARGET side3（inward） | 3 / inward | T | F | `update_boundary_types.f90:638`；`mod_boundary_conditions.f90:234-245` |
| 9 | CORNER free（inward） | corner / inward | T | F | `update_boundary_types.f90:640` |
| 11 | TARGET side2（outward） | 2 / outward | T | F | `update_boundary_types.f90:637` |
| 12 | TANGENT side2（tangent field） | 2 / tangent | F | T | `update_boundary_types.f90:643` |
| 15 | TARGET side3（outward） | 3 / outward | T | F | `update_boundary_types.f90:639` |
| 19 | CORNER free（outward） | corner / outward | T | F | `update_boundary_types.f90:641` |
| 20 | CORNER fixed（tangent） | corner / tangent | F | T | `update_boundary_types.f90:644`；`mod_boundary_conditions.f90:226-233` |
| 21 | CORNER inverted（3 单元） | corner / − | F | T | `update_boundary_types.f90:645`；`mod_boundary_conditions.f90:226-233` |

逐场施加（fixed boundary，`freeboundary=F`；本数据 198 个非零码节点）：

| 场 | 任意 boundary≠0 | apply_dirichlet_all 码（2/3/12/20/21） | apply_cs 码（1/3/4/5/9/11/15/19） |
|---|---|---|---|
| `psi`(0) | Dirichlet（`apply_psi_BC`；freeboundary=T 时 Free/单侧） | Dirichlet | Dirichlet |
| `Phi`(1) | Dirichlet（`apply_dirichlet_1234`） | Dirichlet | Dirichlet |
| `Z_j`(2) | Dirichlet（`apply_current_BC`；free 时 Free/单侧） | Dirichlet | Dirichlet |
| `omega`(3) | Dirichlet（`apply_dirichlet_1234`） | Dirichlet | Dirichlet |
| `rho`(4) | 无 BC（自然，单侧） | Dirichlet | **鞘层通量汇** |
| `T_i`(5) | 无 BC（自然，单侧） | Dirichlet | **鞘层通量汇** |
| `v_par`(6) | 无 BC（自然，单侧） | Dirichlet | **Mach-1：`v_par = sign(B·n)·c_s`** |
| `rho_imp`(7) | 无 BC（自然，单侧） | Dirichlet | **鞘层通量汇** |
| `T_e`(8) | 无 BC（自然，单侧） | Dirichlet | **鞘层通量汇** |

实现语义：

- **Dirichlet = hold**：与 JOREK 矩阵行 `zbig·I` + RHS `zbig·u_old` 等价，边界行不参与
  内部 stencil 写入（roadmap §5.10 "边界节点不参与内部 stencil 写入"按此实现）。
- **Neumann / free（开磁力线）**：用单侧口径——取沿内法向最近的内部节点值
  （零法向通量的单侧实现）；freeboundary=T 的 `psi`/`Z_j` 走同一路径并登记为近似
  （真正的真空耦合未实现，见 §6）。
- **Mach-1（`apply_cs`）**：`v_par = direction·c_s`，`direction = sign(∇ψ_Z·n_R − ∇ψ_R·n_Z)`
  （JOREK `dot((psi_y,-psi_x), normal)`，`mod_boundary_conditions.f90:340-360`）；
  方向由同帧 Hermite dof 梯度算出（本帧 ±各 99 个节点）。
- **鞘层通量汇**：`rho/T_i/rho_imp/T_e` 按
  `rate = c_angle·c_s·R·dl/m`（`c_angle=0.0174524` = 1°，
  `mod_boundary_matrix_open.f90:62,288-305`）衰减，clamp 到 `[0, 1/dt]`
  保证不翻号、不产生负值。
- **磁轴/R→0**：共点节点合并 + 塌缩四边形按三角形处理 + `|J|` 退化丢弃 +
  `1/R` 用 `R_MIN=1e-8` clamp（`delta_plus`/`delta_star`）。

## 4. 多层 / NaN / 一致性实测

真实帧 jorek003000（60 层，dt 请求 0.05 → 有效 1.74e-7）：

| 检验 | 结果 |
|---|---|
| 60 层后有限 / 内部 max | 是 / 9.29（初始状态同量级） |
| Dirichlet 行 60 层后 | 与初始逐元素相等（`torch.equal`） |
| BC 一致性报告 | `dirichlet_ok=True`（max dev 0.0）、`sonic_ok=True`（max dev 0.0） |
| 边界 NaN → 3 层后 | 全网格有限（边界 NaN 在进入 stencil 前被复位） |
| 内部 NaN → Dirichlet 边界 | 边界保持有限、不被污染 |
| BC 类型统计 | psi/Phi/Z_j/omega 各 198 Dirichlet；rho/Ti/Te/rho_imp/v_par 各 104 Dirichlet + 94 鞘层 |

合成 mesh（30 节点/20 单元，码 1/2/11/20）60 层：有限、Dirichlet 行保持、
`dirichlet_ok/sonic_ok` 全 True；边界 NaN 3 层后全有限。
磁轴塌缩 fan（16 节点 → 9 cluster）：mass>0、无 NaN、常场保持、`delta_plus` 有限。

## 5. 测试命令与结果

```bash
.torch-cuda-env/bin/python -m pytest tests/test_stencil_operator.py tests/test_stencil_boundary.py -q
# 16 passed in 1.58s
.torch-cuda-env/bin/python -m pytest -q
# 144 passed, 2 warnings in 85.72s   （基线 128 passed → +16 新增，无回退）
```

## 6. 不确定性与待 JOREK 侧确认（按风险排序）

**仍需 JOREK 侧确认的具体代码映射**：

| code | 已实现映射 | 待确认点 |
|---|---|---|
| 1 / 11 / 5 / 15 | 目标靶板（inward/outward）× side2/side3，Mach-1 | 本帧数据里 1/11 各 32、5/15 各 10，成对出现；但 `direction=sign(B·n)` 用的是本层从 ψ 梯度算的符号，需与 JOREK 内部 `bn` 逐节点比对 |
| 9 / 19 | free corner，Mach-1 + psi/Z_j Dirichlet（非自由边界） | "CORNER free"（RECAP）是否意味着 ψ/Z_j 也应自由（当前按 `apply_psi_BC` 非 free 分支做 Dirichlet） |
| 20 | corner fixed（两侧不一致），Dirichlet-all | 映射依赖 `grid_to_wall` 的 side 指派（`construct_matrix_mod.f90:116-140`）；本 run 无 `grid_to_wall` 输出可核对 |
| 21 | inverted corner（3 单元特殊情形），Dirichlet-all | Fortran 注释 "there we need to let is free, or fix it???"（`update_boundary_types.f90:470`）表明原作者也不确定；本层按 `apply_dirichlet_all` 实现 |
| 3 / 4 / 12 | 旧 grid 类型（3=Dirichlet-all+Mach-1、4=target、12=tangent side2） | 本帧数据未出现，仅按源码表登记，未做数据核对 |
| 全部代码 | `use_simple_bnd_types`、`no_mach1_bc` 未从 namelist 取值链读取 | 若目标 run 置 `.t.`，code 11/15/19/12/20/21 会先被折叠（见 §6.4） |

1. **freeboundary=T 的真空耦合未实现**（中高风险）：本数据 `freeboundary=F`，未触发；
   实现为单侧自然条件近似。若后续用自由边界 run，需按
   `mod_vacuum_fields.f90` 的 `is_freebound` 耦合补真空响应矩阵。
2. **X 点节点的 `boundary=0` 语义**（中风险）：由 `neighbours<=0` 定义的几何边界节点有
   266 个，其中 198 个有非零 BC 码，68 个（含 X 点相邻节点）码为 0；本层按 JOREK
   语义对码 0 不做 BC 复位、正常参与 stencil。X 点处 `neighbours` 有 32/15876 条与
   共享边不匹配（0.20%，全部在含 X 点节点 2/6/7 的单元；另有 68 条共点边不参与比较），
   算子做通量只用 `vertex`、不用 `neighbours`，但邻接一致性检查须排除这些单元。
3. **Mach-1 只用 `c_s` 与鞘层通量汇的近似**（中风险）：
   `density_reflection/imp_reflection/gamma_sheath_i/e/vpar_smoothing/
   mach_one_bnd_integral/neutral_source` 与 `rhs_ij_5..9` 的线性化项未进层；
   方向 `direction` 依赖调用者提供的 ψ 梯度（不提供时默认 +1 并登记）。
   需要 JOREK 侧确认：这些系数在目标 run 的取值（namelist 默认）与是否要逐项复刻
   `amat_*`（当前只复刻 Dirichlet/自然/声速三类行）。
4. **`use_simple_bnd_types` 未从 namelist 读取**（中风险）：若为 `.t.`，
   11→1、15→5、19→9、12→4、20/21→9（`update_boundary_types.f90:665-676`）。
   本 run 的 `jorek_namelist_spitest` 未设该项（默认 F），且 h5 保留 11/15/19 ⇒
   按原始码处理正确。后续应从 namelist 读入并登记。
5. **稳定步长 cap**（中风险）：`dt_effective` 由 Gershgorin 界给出（本帧 1.74e-7），
   显式 60 层几乎等于恒等推进；B2 若要在物理 dt 上推进，需要隐式解
   `(I - dt L) x_new = x`（JOREK 的做法）或学习型稳定格式。当前 cap 已防止爆炸，
   但"多层推进的物理时间尺度"未验证。
6. **鞘层通量汇的 `dl`/`m` 归一化**（低中风险）：Fortran 是边界积分
   `∫ N_i (…) R dl` 的弱形式装配，本层用节点集总量 `c_angle·c_s·R·dl/m` 近似；
   系数数量级一致，但未逐项对 `ws`/基函数积分。
7. **磁轴数据未覆盖**（低风险）：本 run `axis_node=F`、`treat_axis=F`、R 范围
   0.463–2.02，无真实磁轴；轴塌缩只由合成 fan mesh 验证。
8. **`contain_node`**：本帧全 0（JOREK 只在部分输出写它），加载器在非零时校验
   与 `vertex` 一致；真实连通性以 `vertex` 为准。

## 7. 开放问题（按风险排序）

1. B2：把 stencil 层的显式 cap 换成隐式/半隐式局部 matmul（保持 FEM 邻接、禁 GNN/谱），
   并在物理 dt 上验证多层边界不变量；显式界按 `dt_term = 2/(α_term·λ_max)` 逐项计算
   （η 稳定、ZK/D_par/visco 刚性，见 §2 单位修正表）。
2. 逐项复刻 `mod_boundary_matrix_open.f90` 的 `amat_*`/`rhs_ij_*`（含 reflection、
   gamma_sheath、vpar_smoothing、neutral_source），替换当前"声速 + 通量汇"近似。
3. 与 JOREK 侧核对：X 点/free 节点（boundary=0）在 B2 推进器里是否应保持纯内部演化；
   `freeboundary=T` run 的真空响应是否需要最小复刻。
4. namelist→BC 的取值链（`use_simple_bnd_types`、`no_mach1_bc`、
   `mach_one_bnd_integral`）进 context 构造，避免按默认假设。

## 8. 2026-10-04 复核修正（B2 发现）

B2 复核发现 `StencilConv` 的 element matmul 路径把 `[4, E]` 连接矩阵按行主序展平
（`vertex_cluster.reshape(-1)`），得到"连续 4 个单元的同一条边"而不是"同一单元的
4 个角点"，与 element-major 语义不符；已在 `src/models/stencil_conv.py` 修复
（`transpose(0,1).reshape(-1)`）并新增回归测试
`test_element_matmul_path_matches_the_assembled_weak_form_operator`（element 路径
vs 稀疏 `-Ax/m`，1e-8 容差）。本报告原 16 个测试未覆盖该置换（守恒/常场在该置换下
不变、线性场检验走稀疏路径），修复后仍全绿；稀疏矩阵路径与 `fem_weak` 的逐元素
一致结论不受影响。

## 9. 2026-10-04 边界语义确认（用户 + JOREK 源码核对）

代码标签（`grids/grid_utils/update_boundary_types.f90:630-702` 注释与赋值）：

- `1` TARGET side2 (inward)、`11` TARGET side2 (outward)、`5` TARGET side3 (inward)、
  `15` TARGET side3 (outward)；`9` CORNER free (inward)、`19` CORNER free (outward)；
  `2` TANGENT side3、`12` TANGENT side2；`20` CORNER fixed (tangent)；
  `21` CORNER inverted (3 elements, special case)。
- model502 的实际施加（`models/model502/mod_boundary_conditions.f90:227-248`）：
  `2/3/12/20/21 → apply_dirichlet_all`；`1/3/4/5/9/11/15/19 → apply_cs`
  （Mach-1/鞘层）。
- 开关默认：`no_mach1_bc=F`（`models/preset_parameters.f90:679`）、
  `mach_one_bnd_integral=F`（:91）、`use_simple_bnd_types=F`（:741）；
  生产 namelist 覆盖为 `bc_natural_open=.true.`、`freeboundary=.false.`
  （`jorek_namelist_spitest:398/131`）。
- 角点对（`3/9/19/20/21` 相邻）在 `matrix/construct_matrix_mod.f90:124-130` 以
  `is_corner_pair → cycle` **跳过**（与用户"构建时跳过角点"的记忆一致）；
  不满足允许组合的边界边也会 `cycle` 并打印警告（:110-175）。

对 B2 的要求：按本表复核 BC 层映射（尤其 corner-pair 的鞘层积分跳过与
`apply_cs`/`apply_dirichlet_all` 的分支）；Mach-1 的 reflection/gamma_sheath/
vpar_smoothing/`rhs_ij_*` 线性化项仍属未复刻部分。
