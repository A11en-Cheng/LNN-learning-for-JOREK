# Marker → projection 配方（Phase 1.5.1 契约参考）

目的：说明 `projections*.h5` 到底是怎么从 `part*.h5` 的 marker 状态算出来的，以及
Phase 1.5.1 的"机器精度反向组装 gate"需要复刻哪些步骤。所有引用都来自 JOREK 工作树
`/home/ac_desktop/JOREK_code/jorek`。

## 1. 整条链路（四步）

```
part*.h5 的 marker 状态（weight, P_imp, i_elm, st, x(3)）
   └─(A) 逐粒子、逐子步累积 RHS        [marker_spi_source.f90 ~615-650]
        └─(B) 除以 gather time 归一化   [marker_spi_source.f90 ~670-676]
             └─(C) FEM 投影求解 M x = b [mod_project_particles.f90::project]
                  └─(D) 写 node%values → projections*.h5 [save_to_h5]
```

**关键结论：`projections*.h5` 的 `values` 是第 (C) 步解出来的 FEM 节点系数，不是原始 RHS。**
把 marker 存量直接和它比总量必然差若干数量级。

## 2. (A) RHS 累积（use_marker 分支）

对每个粒子、每个子步（`particles/examples/marker_spi_source.f90`）：

```fortran
call basisfunctions(st(1), st(2), HH, ...)       ! s,t 在 [0,1]，bicubic Hermite
v_1 = HH(l,m) * size(l,m,e_old) * dEion_dt * t_norm / E_norm   ! dE_ion
v_2 = HH(l,m) * size(l,m,e_old) * rad_sink * t_norm / E_norm   ! rad_sink
v_3 = HH(l,m) * size(l,m,e_old) * Z_eff / n_norm               ! Z_eff
v_4 = HH(l,m) * size(l,m,e_old) * Z_imp / n_norm               ! Z_imp
v_5 = HH(l,m) * size(l,m,e_old) * N_imp / n_norm               ! N_imp
feedback_rhs(m,l,e_old,i_tor,ch) += HZ(i_tor) * v_ch
```

通道量的定义（同一文件，粒子循环里）：

| 通道 | 量 | 单位/说明 |
|---|---|---|
| 1 dE_ion | `Σ_iZ (P_new(iZ)-P_old(iZ)) · E_ion(iZ) · w`，再乘 `EL_CHG` | eV→J；需要 ADAS 电离/复合率做电荷态更新 |
| 2 rad_sink | `w · Lrad(n_e,T_e) · n_e · timesteps` | 需要 ADAS 辐射函数 |
| 3 Z_eff | `w · Σ_iZ iZ² P(iZ)` | 只用 marker 电荷态分布 |
| 4 Z_imp | `w · Σ_iZ iZ P(iZ)` | 同上 |
| 5 N_imp | `w` | 同上 |

通用路径（`particles/diagnostics/mod_project_particles.f90::sample_rhs`）是同一形式
`HH · size · f(particle) · weight`，只是 `f` 由 `proj_f` 列表给出。

## 3. (B) 归一化

```fortran
jorek_feedback%rhs_gather_time = n_steps * timesteps     ! 秒
jorek_feedback%rhs = feedback_rhs / jorek_feedback%rhs_gather_time
```

即记录的是**帧区间内的时间平均**（源项速率），不是区间累计量。logfile 里
`GATHER TIME : 4.0857752259518330E-007` 就是这一步打印的值（等于该次粒子步的
`n_steps*timesteps`）。

## 4. (C)(D) 投影求解与写出

- `project()`（`mod_project_particles.f90`）先 `sample_rhs`，再 `project_only`
  解投影系统（质量矩阵 + namelist 里的 filter/hyper-filter/parallel-filter），
  解落在 `node%values`；随后 `save_to_h5` 调
  `write_particle_distribution_to_h5`，把 `node%values(:,:,1:n_fields)` 写成
  `projections*.h5` 的 `values`（`n_tor, n_degrees, n_fields, n_nodes` 顺序）。
- 单元矩阵在 `models/element_matrix_projection.f90`（4×4 dof 的高斯积分质量矩阵，
  含滤波项）。

## 5. 本仓库已实现/未实现

| 步骤 | 状态 |
|---|---|
| (A) 三密度通道 RHS（Z_eff/Z_imp/N_imp） | **已实现** `src/data/marker_projection.py`（含 bicubic 基函数、`HH·size`、`1/n_norm`、无效粒子跳过），单测 `tests/test_marker_projection.py` |
| (A) 两能量通道 RHS | 未实现（需要 ADAS 电荷态更新与辐射函数） |
| (B) gather-time 归一化 | 已实现 `gather_time(n_steps, timesteps)`；`n_steps`/`timesteps` 需从 namelist/日志取值 |
| (C) 质量矩阵 + 滤波 + 求解 | **未实现**（下一步） |
| (D) 与 `projections*.h5` 对比 | 待 (C) 完成后逐帧做机器精度比对 |

## 6. 数值证据（为什么旧检查必然失败）

以 `scan_1d22_ang15_5_massloss_neon_v400/jorek006000.h5` 为例（`n_norm = 0.8e20`）：

| 量 | 数值 |
|---|---|
| 本仓库组装的 RHS 和（Z_eff / Z_imp / N_imp） | 3.78e2 / 4.72e1 / 6.51e0 |
| 记录的节点系数和（项目文件同一帧） | -1.389e6 / -1.604e5 / -1.837e4 |
| 旧检查（全部存活 marker 加权求和） | 与记录相差 ~1e17 |

差异来源：RHS 是"源项在单元上的加权积分"，而记录值是"用质量矩阵解出的节点系数"；
局域源在 FEM 基下会产生大量符号相反的系数，因此在解空间里量级大得多。

## 7. 下一步

1. 移植 `element_matrix_projection.f90` 的质量矩阵（bicubic × 高斯积分 × R 权重），
   读 namelist 的 filter 系数（默认 0 时矩阵就是纯质量矩阵）。
2. 用 `M x_recorded` 与 (A)+(B) 的 RHS 逐帧比对——先验证组装，再验证求解；
   两者都过才达到 Phase 1.5.1 的机器精度 gate。
3. 需要时再补能量通道（ADAS 表已在仓库/运行目录可用）。

## 8. 投影系统的规模与求解成本（效率参考）

来源：`mod_project_particles.f90`（`new_projection` 里 `assemble_system`）、
`particles/diagnostics/mod_uncoupled_projection.f90`（`assemble_projection_matrix_n0`，
725-1148 行）、`solvers/solve_mumps_all.f90`。

- **未知量**：`system_size = 2 × n_nodes × n_degrees`（n_tor_local = 1 的 n=0 谐波）。
  本数据集 `n_nodes = 4125`、`n_degrees = 4` → **33,000 个未知量/谐波**，
  右端列数 = 投影通道数（本次 5 列）。2D run 只有 n=0 一个谐波。
- **稀疏度**：单元刚度/质量块 16×16 → 结构非零约
  `n_elements × 16 × 16 ≈ 3986 × 256 ≈ 1.0e6`（再加 Dirichlet 边界行）。
- **组装**：每个 run 只组装一次（`new_projection` 构造时），与帧数无关。
- **求解**：`solve_mumps_all`（或 PaStiX / Strumpack）带
  `initialized / analyzed / factorized` 与 `solve_only` 缓存标志——
  **分解一次，之后每帧只做回代**。
- **量级**：33k 未知量的 2D FEM 稀疏直接分解在单核上是秒级、几核更快；
  每帧回代（5 列）毫秒级。相对 marker 循环（1e6 粒子 × 每个流体步的子步）
  与流体矩阵求解可以忽略。
- **对本 gate 的含义**：验证阶段**不需要求解**。只要用记录的解 `x_recorded` 计算
  `M · x_recorded`（稀疏矩阵-向量乘，~1e6 次浮点运算，毫秒级）并与第 (A)+(B) 步
  组装的 RHS 比对，就能验证装配；只有需要"从 marker 生成投影"时才需要真求解。
  滤波项（namelist `filter_perp=1e-4`、`filter_par=1e-2` 等）会进入 `M`，比对时
  必须带上，否则残差会出现 1e-4~1e-2 量级的结构化偏差。

## 9. Gate 现状（2026-09-29，未通过）

已实现并跑通：

- 单元质量矩阵基项（`tests/check_projection_gate.py::element_mass_matrix`，
  4×4 高斯积分 + `HH·size` 双线性 + `xjac·R`）；
- 三密度通道 RHS 与 `M·x_recorded` 的逐单元残差统计。

结果与已知差异：

| 检查 | 结果 |
|---|---|
| 记录值来源 | 已确认：`project_only` 把求解向量 `y_tmp` 写进 `node%values`（解，不是 RHS） |
| 单元几何 | 顶点 1 的角点重建与节点坐标一致；其余顶点需确认 JOREK 的顶点↔(s,t) 顺序 |
| `element%size` | 4×4 有符号张量：值 dof = 1，导数 dof ~1e-2..1e-4（不是面积） |
| `M x_recorded` vs `b` | 当前 `M x` 比 `b` 小约 2 个数量级 → **未达机器精度** |

待排除项（下一步按序做）：

1. 矩阵拷贝时的 `× TWOPI / n_plane`（`mod_uncoupled_projection.f90:~653`）——
   若 RHS 不带同一因子，需在比对里补上（本数据 `n_plane=1`，因子 2π）；
2. 顶点↔(s,t) 顺序与局部 dof 抽取顺序（本脚本用 `vertex` 顺序 + `[dof, vertex]` 转置）；
3. `rhs_vec(1:n) -= scaling_integral_weights * integral_weights`（n=0 零积分约束，
   见 `project_only` 的 RHS 填充段）；
4. 帧内 81 个粒子子步的累积语义：RHS 用每个子步的 `i_elm_old` 与逐步电荷态，
   而 `part*.h5` 只保存帧末快照——若前 3 项排除后仍有残差，则这一项是**数据侧**的
   机器精度上限，需要 JOREK 侧导出子步累计量（P2 类改动）。

### 9.1 复查结果（2026-09-29 第二轮）

已排除 / 已修：

| 项 | 结论 |
|---|---|
| `scaling_integral_weights × integral_weights` | **排除**：该修正只在 `if (do_zonal)` 分支里执行（`mod_uncoupled_projection.f90:213-218`），本 run 未启用 |
| `TWOPI / n_plane` | `n_plane = 1`，矩阵拷贝时乘 `2π`；RHS **不带**该因子（`!* TWOPI` 被注释）。已加入比对 |
| 顶点↔(s,t) 顺序 | 标准顺序：顶点 1/2/3/4 ↔ (0,0)/(1,0)/(1,1)/(0,1)，值 dof 重建与节点坐标逐位一致 |
| 坐标轴序 bug | **已修**：HDF5 的 `x` 是 `[dim, dof, tor, node]`，局部单元需转成 `[vertex, dof, dim]`；此前把 dof 当成了 vertex |

剩余差异：对源区单元 `||M x_recorded|| / ||b|| ≈ 1.2e-5`，即组装的 RHS 比"记录解 × M"
大 ~1e5。用**不依赖 FEM 的标量锚点**交叉检查（logfile 的
`Total generated electrons`，即电子源速率）：

| 量 | 数值 |
|---|---|
| logfile `Total generated electrons`（速率，例如 index 5052） | 3.51e21 s^-1 |
| 本仓库 `Σ w⟨Z⟩`（marker 存量，t=6000 帧）除以 gather time | 1.67e28 s^-1 |

两者差 ~4.8e6 —— 与 gate 里缺的 ~1e5 同量级。结论：**缺口在 marker 侧归一化链**
（`weight` 的含义/单位、或它在投影累积里的用法），不在 FEM 组装。
下一步先用这个标量锚点把 `weight`/`n_norm` 的归一化定死（单标量、无需 FEM），
再回到逐单元 gate。

### 9.2 归一化定死（2026-09-29 第三轮）

**marker 权重是正确的**。JOREK 在 `marker_spi_source.f90` 里定义

```fortran
Z_imp = weight * dot(P_imp, P_Z) * timesteps      ! 密度通道带 timesteps
Z_eff = weight * dot(P_imp, P_Z**2) * timesteps
N_imp = weight * timesteps
Ne_tot = Ne_tot + Z_imp                            ! 在子步循环里累加
... print "Total generated electrons", Ne_tot_all/(n_steps*timesteps)
```

即打印值 = `Σ_particles weight·⟨Z⟩`（`n_steps` 与 `timesteps` 全部抵消）。
实测同一帧（index 6000）：logfile `6.8384e21` vs 本仓库 `Σ w⟨Z⟩ = 6.843e21`
—— 差 0.07%，**吻合**。

由此得到密度通道 RHS 的正确形式（**不除 gather time**）：

```
b_ch = Σ_particles HH·size · w·(1 or ⟨Z⟩ or ⟨Z²⟩) / n_norm
```

因为 `Z_imp = w⟨Z⟩·timesteps` 的 `timesteps` 与 `1/(n_steps·timesteps)` 相消。
（先前脚本多除了一次 `gather_time`，是 2.45e6 倍的高估；已修。）

修完后的逐单元结果：`median ||M x_recorded|| / ||b|| ≈ 1.4e3`，且对 16 维局部系统
做最小二乘标定时 `x_recorded` 与 `M⁻¹b` 的相关性 ≈ 0（不是单一全局因子），说明
**不只是归一化问题，矩阵或 RHS 的结构仍有差异**。

剩余怀疑点（按优先序）：

1. 全局系统每 node-dof 有 2 个未知量（cos/sin 对），`project_only` 对 n=0 的第二分量
   置零并施加 Dirichlet；**局部 16×16 检查忽略了这一配对与边界消元**；
2. `element%size` 在矩阵与 RHS 中的用法需要逐项对齐（矩阵里 `size` 出现两次、
   RHS 里一次，若有 dof 相关的附加因子会被放大）；
3. 最直接的办法：用 run 目录里现成的 `jorek2_diagno_spi` 二进制在拷贝出来的
   `jorek_restart.h5/part_restart.h5` 上跑一次，得到"受控"的投影输出，
   再与磁盘记录/我们的 Python 三方对照。
