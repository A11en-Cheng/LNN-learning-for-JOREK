# Phase A′ 数据审计：逐步导出契约核对与生成侧补字段清单（2026-10-04）

**状态**：current。**依据**：ADR-0018/0019、`roadmap_v2.md` Phase A、
`report/data-contract-v3.md`（冻结）、`report/data-export-spec-v3.md`（冻结）。
ADR-0001–0017 为 v1 历史，未作为依据、gate 或对照。

**结论（一句话）**：扫描范围内**没有已交付的 A′ 逐步导出 run**
（无 `<run>/export_v3/manifest.json`、无 `step_*.h5`），因此本报告对生成侧的
*来源文件*（`jorek{n}.h5` / `projections{n}.h5` / `part{n}.h5` /
`macroscopic_vars.dat` / `abl_history.dat` / 生产 namelist）做逐字段清点，
并给出 A′ 文件的精确补字段清单（§9）。`spi_*_arr` 16 个数组在 SPI 开启后的
原始 `jorek` 中**全部存在**（`jorek007360`：`n_spi_tot=251`；注入前 55 帧没有，
见 §2.1），缺的不是字段本身，而是**每流体步双端（`shard_start/shard_end`）导出**
与区间元数据。loader + 验收 CLI 已合入（§10），
smoke 数据一落地即可用同一命令出验收报告。

---

## 1. A′ 导出搜索证据（只读）

2026-10-04 扫描（全部只读）：

| 范围 | 命令/方式 | 结果 |
|---|---|---|
| 全部 EHL-2 run 目录（`/mnt/d/EHL-2/*`） | `find … -maxdepth 5 -name step_*.h5 -o -iname '*export*v3*'` | 0 命中 |
| 仓库与 `.scratch` | `rg -n "export_v3" repo/`、`find .scratch` | 仅规范/进度文本，无数据 |
| Windows 用户目录（浅层） | `find /mnt/c/Users -maxdepth 4 -iname '*export_v3*'` | 无 A′ 导出 |
| JOREK 源码树 `/mnt/d/Proj/jorek` | `rg -l -i "export_v3\|shard_start"` | 无 A′ 导出代码/字符串 |
| 全盘浅层 | `find / -maxdepth 4 -name export_v3` | 无 |

> 记录：A′ 生成尚未在可见文件系统中交付。本报告中的“存在”均指**来源文件**
> （`scan_1d22_ang12_massloss_neon`，帧 `jorek007360/7350`、`part07360`、
> `projections007360`、`jorek008900`（预算用最末帧）、生产 namelist
> `jorek_namelist_spitest`），不是 A′ 文件。

---

## 2. 每步 `jorek{n}.h5` 字段审计（A′ 主路径的字段来源）

帧 `jorek007360.h5`（`n_nodes=4125`，`n_elements=3986`，`index_now=7360`，
`t_now=10474.75`，`t_norm=4.085775226e-7 s`，`tstep=0.05`）：

| A′ 需要 | 原始字段 | 形状 / dtype | 状态与备注 |
|---|---|---|---|
| `u_before/after` | `values` | `[9,4,1,4125]` float64（另有 `aux_values`、`deltas` 同形） | ✅ 通道顺序 `psi,Phi,Z_j,omega,rho,T_i,v_par,rho_imp,T_e`（schema §1 恒等映射） |
| shard 状态 | `spi_*_arr`（16 个） | `[n_spi_tot]` float64，`n_spi_tot=251` | ✅ 字段齐（明细 §3）；❌ 没有“窗口起点”快照，需生成侧缓存上一步 |
| mesh | `x` | `[2,4,1,4125]` float64 | ✅ 几何米，Bezier 控制点 |
| | `size` | `[4,4,3986]` float64 | ✅ 单元几何权重 |
| | `vertex` | `[4,3986]` int32（1-based） | ✅ 需在 manifest 写明 1-based |
| | `contain_node` | `[5,3986]` int32 | ✅ 原始形状即 `[5,n_elem]` |
| | `neighbours` | `[4,3986]` int32 | ✅ X 点 32/15876 条失配（B0 §6，算子只用 `vertex`） |
| | `parent_elem` / `parents` / `index` | `[4125]` / `[2,4125]` / `[4,4125]` int32 | 可选，A′ 未要求 |
| 边界 | `boundary` | `[4125]` int32 | ✅ 本帧码分布：0×3927、1×32、2×94、5×10、9×6、11×32、15×10、19×4、20×4、21×6（全部在 B0 注册表内） |
| | `freeboundary` | `(1,)` `|S1` = `b'F'` | ❌ **全局 flag，不是逐节点**；A′ 要 `[nodes] int8`（缺失=warning，已预留） |
| | `axis_node` | `[4125]` `|S1`（`T`×68） | ❌ **逐节点 flag，不是索引表**；A′ 要 `int64 [n_axis]`（68 个节点） |
| 时间 | `t_now` / `t_norm` / `index_now` / `tstep` | 标量 | ✅ 双单位导出的换算源 |
| `grid_to_wall` | —— | —— | ❌ 原始 h5 无此字段；**用户侧调查项**，loader 只登记 warning（§10） |
| 预算系列 | `xtime*`（12 个） | `[index_now]` / `[index_now,1]` float64 | ✅ **完整历史**（非当前步切片），见 §8 |

**时间语义复核（本帧）**：

| 量 | 值 | 与 `T_n` 关系 |
|---|---|---|
| `T_7360`（`macroscopic_vars @times`） | 10474.750000 | 基准 |
| `jorek007360.t_now` | 10474.750000 | = `T_7360` |
| `projections007360.t_now ÷ t_norm` | 10474.750000（差 5.5e-12） | = `T_7360`（窗口 `[T_7359,T_7360]` 平均，ADR-0019） |
| `part07360.time ÷ t_norm` | 10474.700000（差 7.9e-10） | = `T_7359 = T_7360 − dt_source(0.05)` |

### 2.1 字段逐帧存在性（675 个 `jorek*.h5` 全扫描）

| 字段 | 缺失帧数 | 缺失范围 | 说明 |
|---|---:|---|---|
| `values`、`boundary`、`freeboundary`、`t_now/t_norm`、`size/vertex/contain_node/neighbours/x`、`n_spi_tot` | 0 | — | 每帧都有 |
| `spi_*_arr`（16 个）、`xtime_spi_ablation*`、`xtime_Ne_imp` | **55** | 索引 `0,50,…,2700`（`T ≤ 8367.0`） | 注入前阶段：`n_spi_tot=0`，数组**不存在**（不是空数组） |
| `xtime_E_ion*`、`xtime_radiation*`、`xtime_rad_power` | 1 | 索引 0（`T=0`） | marker 模块从 index 50 起写出 |

首个含 `spi_*_arr` 的帧：**index 2710，`T=8412.0`，`n_spi_tot=251`**（生产
`t_ns` 门限落在 2700 与 2710 之间）。`projections*` 621 帧（2700–8900，步长 10）；
`part*` 620 帧（2710–8900）；索引 2700 只有 projection、没有 part（已登记的覆盖边缘）。

---

## 3. `spi_*_arr` 明细（A′ shard 主路径）

16 个数组**从 index 2710（`T=8412.0`）起**全部存在（`jorek007360.h5`，
`n_spi_tot=251`，float64，长度 251）；index 0–2700 的 55 帧没有这些数组
（`n_spi_tot=0`，注入未开始）：

| 组 | 键 | 形状 | A′ 目标 |
|---|---|---|---|
| 位置 | `spi_R_arr`、`spi_Z_arr`、`spi_phi_arr` | `[251]` | `/shards/{start,end}/…` |
| 速度 | `spi_Vel_R_arr`、`spi_Vel_Z_arr`、`spi_Vel_RxZ_arr` | `[251]` | 同上 |
| 标量 | `spi_radius_arr`、`spi_abl_arr`、`spi_species_arr`、`spi_vol_arr` | `[251]` | 同上 |
| 磁通/初始化 | `spi_psi_arr`、`spi_grad_psi_arr`、`spi_phi_init_arr` | `[251]` | 同上 |
| drift 分量 | `spi_psi_arr_drift`、`spi_grad_psi_arr_drift`、`spi_vol_arr_drift` | `[251]` | 同上 |

**缺字段（P0）**：A′ 每步要求 `/shards/start/<name>` 与 `/shards/end/<name>`
**同时**存在。原始 `jorek{n}` 只有 `T_n` 末端状态（对应 `shard_end`）；`shard_start`
= 上一步 `jorek{n−1}` 的同一批数组（`T_{n−1}`），生成侧必须在导出 n 时缓存 n−1
（或直接写 `part`/窗口数据）。loader 对缺失/形状不符 **fail-fast**（已实现并测试）。
**注入边界裁决（P0）**：首个 SPI 帧 2710 的 `shard_start = T_2700` 属于注入前阶段
（`n_spi_tot=0`、无数组）⇒ 要么对 2710 显式导出空 shard（`n_spi_tot_start=0`，
数组长度 0，**不得省略 key**），要么 A′ 导出窗口从 2720 开始（其
`shard_start = T_2710` 已有 251 个碎片）。两者都要写进 manifest。

---

## 4. `projections{n}.h5` 审计

| 项 | 值 |
|---|---|
| `values` | `[5,4,1,4125]` float64；5 通道顺序 `dE_ion, rad_sink, Z_eff, Z_imp, N_imp`（schema §3.1） |
| `t_now` | 物理秒；`÷ t_norm` = `T_n`（窗口末端，本帧差 5.5e-12） |
| 网格字段 | `x`/`vertex`/`size`/`neighbours` 同帧同形状（可校验同源） |
| provenance | `jorek_model=-1`、`RCS_version`；无 `source_object` 字段 → A′ 必须显式写 `/provenance/source_object` |

**与 A′ 的关系**：`S_used` 若采用投影通道，`source_object` 必须写 `"projection"`
（ADR-0016/0018：投影≠内部 `inj_source`，只作监督/交叉校验）；若生成侧能导出内部
`inj_source`，写 `"inj_source"`。**禁止**两种语义混用或静默互换。

---

## 5. `part{n}.h5` 审计（可选诊断流，非主路径）

`part07360.h5`：`/time = 0.00427973 s`（= `T_7359`）；`groups/001/`：

| 字段 | 形状 / dtype | 过滤/量化规则（实测） |
|---|---|---|
| `weight` | `[1000000]` float64 | 死亡槽位 `weight≈1.0`；**存活过滤必须用 `i_life>0`**（等价 `i_elm>0`，181990/1e6），不能用 `weight!=0` |
| `P_imp` | `[1000000,11]` float64 | 每行和=1（实测精确 1.0）；规范建议 uint8 量化（A′ 可选流） |
| `x`、`v` | `[1e6,3]` float64 | 死槽位全 0 |
| `i_life`、`i_elm` | `[1e6]` int32 | `i_life∈{0,1,2,3}`；`i_life=0` 为死槽位 |
| `st` | `[1e6,2]` float64 | 可选 |
| `t_birth` | `[1e6]` float32 | 本帧**全 0**（疑似未填充，登记） |
| 组属性 | `Z=10`、`mass=20.17970085`、`type='particle_marker'`、`adas_suffix='96_ne'` | 需在 manifest 登记 |

`weight` 全量求和 = `1.609298e21`（与 ADR-0019 的日志核对值一致）；存活
（`i_life>0`）求和同为 `1.609298e21`（死槽位权重≈1，可忽略）。

---

## 6. `condition`（61 维）审计

用现有 `src/data/parameter_channel.py` 对生产 namelist
`jorek_namelist_spitest` + `jorek007360.h5` 实测：

| 项 | 结果 |
|---|---|
| 维数 | `CONDITION_DIM=61`，names/vector 等长 61 |
| provenance 分布 | namelist 42、h5 12、derived 2、default 4 |
| default 填充项 | `visco_par_par=0`、`T_0=1e-6`、`Ti_0=5e-7`、`Te_0=5e-7`（JOREK preset 默认） |

用户点名的参数：

| 参数 | 结果 | 来源 |
|---|---|---|
| `D_par` | 0.01 | namelist（`:426`） |
| `ZK_i_par` | 1.6666e5 | namelist（`:462`） |
| `ZK_e_par` | 4.1037e6 | namelist（`:463`） |
| `ZK_par_max` | 1e4 | namelist（`:464`） |
| `D_perp` | 6 分量 `[5e-6,0,0.13,0.01,0.92,0]` | namelist（`:437-442`，展开为 `D_perp[1..6]`） |
| `d_perp_num` | 1e-12，**不在 61 维向量内** | namelist（`:399`） |
| `visco_num` | 7e-4，**不在 61 维向量内** | namelist（`:389`） |
| `eta_num` | 1e-9，**不在 61 维向量内** | namelist（`:396`） |

结论：点名的参数**全部能从生产 namelist 读到**；`d_perp_num`/`visco_num`/`eta_num`
属于 stencil 刚度/数值参数，当前 61 维 condition 未收录（也不应进入 FiLM 的
`excluded_*` 之外路径）。生成侧若需登记，应在 A′ 的 `/condition` 中新增
`group="excluded_transport"`、`film_eligible=0` 条目，或在 manifest 单列
transport 参数块。`freeboundary`/`bc_natural_open` 已按 `boundary_flags` 组解析
（H5 全局 flag + namelist）。

**建议的 61 维分组映射**（A′ `/condition/groups` 用；按冻结词表自动落组）：

| group | 条目数 | 条目 |
|---|---:|---|
| `excluded_transport` | 42 | `eta, visco, visco_par, visco_par_par, ZK_i_par, ZK_e_par, D_perp[1..6], ZK_i_perp[1..6], ZK_e_perp[1..6], central_density, central_mass, F0, R_domm, sqrt_mu0_rho0, t_norm, ZK_par_max, ZKpar_T_dependent, Ti_min_ZKpar, Te_min_ZKpar, T_min, T_min_neg, rho_min, rho_min_neg, ZK_e_par_neg, ZK_i_par_neg, D_par, D_par_imp` |
| `excluded_numeric` | 4 | `time_evol_theta, time_evol_zeta, tstep, t_now` |
| `boundary_flags` | 8（含重复 1） | `gamma_i_stangeby, gamma_e_stangeby, bc_natural_open, freeboundary, resistive_wall, bc_natural_open, n_open, n_limiter` |
| `background_plasma` | 5 | `T_0, Ti_0, Te_0, particlesource, ellip` |
| `pellet_mass` | 2 | `n_spi, pellet_density` |

**重复名（生成侧需处理）**：`bc_natural_open` 在 61 维向量里出现两次（0-based
索引 43 与 58，值同为 1.0）。A′ `/condition/names` 若直接照搬会破坏“名字与
values/groups 对齐”的唯一性；建议去重，或写成 `bc_natural_open#2` 并在
provenance 登记来源（两条实际来自不同 `_SPEC` 条目）。

---

## 7. 预算字段布局：完整历史 vs 当前步切片

**实测结论：原始 `jorek{n}.h5` 里的 `xtime*` 是完整历史，不是切片。**

证据（`jorek007350` vs `jorek007360`，同 run）：

| 系列 | 长度 7350 → 7360 | 共享前缀 |
|---|---|---|
| `xtime`、`xtime_spi_ablation`、`_rate`、`_bg`、`_bg_rate` | 7350 → 7360 | 逐位相等 ✅ |
| `xtime_Ne_imp`、`xtime_E_ion`、`_power`、`xtime_radiation`、`xtime_rad_power` | 7350 → 7360 | 逐位相等 ✅ |

即数组长度 = `index_now`，第 i 个元素 = 第 i 个流体步的记录值；新帧只 append。
A′ 导出两种布局都必须支持（loader 按 manifest `budget_layout ∈ {history,slice,absent}`
读取，未写时自动探测长度）；**manifest 必须写清**。

来源语义（代码行号）：

| 系列 | 语义 / 单位 | 源码 |
|---|---|---|
| `xtime_spi_ablation` | 累计烧蚀原子（species 加权），`+= t_norm*tstep*spi_abl*species` | `models/pellet_module.f90:319-321` |
| `xtime_spi_ablation_rate` | `+= spi_abl*species`，原子/秒 | `models/pellet_module.f90:767-768` |
| `xtime_Ne_imp` | 生成电子 / JOREK 时间（`Ne_tot/(n_steps*timesteps)`） | `particles/examples/marker_spi_source.f90:735-739` |
| `xtime_E_ion` / `xtime_E_ion_power` | 累计电离损失能量 / 功率（JOREK 能量、能量/JOREK 时间） | 同上 |
| `xtime_radiation` / `xtime_rad_power` | 累计辐射损失能量 / 功率 | 同上 |
| `abl_history.dat` | 列：`i, time, rate_inj(10), abl_inj(10), bg_rate_inj(10), bg_inj(10), rate_tot, abl_tot, bg_rate_tot, bg_tot`（46 列，`n_inj` 数组维=10） | `diagnostics/jorek2_diagno_spi.f90:185-205` |
| `macroscopic_vars.dat` | 块格式；`energy_conservation`、`dparticles_dt`、`particlecontent`、`bnd_particle_fluxes` 等 | `communication/live_data.f90:591,650-680,845-866` |

Run 目录存在性（`scan_1d22_ang12_massloss_neon`）：

| 文件 | 状态 |
|---|---|
| `macroscopic_vars.dat` | ✅ 294501 行；`times`/`energy_conservation`/`dparticles_dt`/`particlecontent`/`bnd_particle_fluxes` 各 8908 行数据（`dissipative_terms` 8907） |
| `abl_history.dat` | ✅ 8900 行 + 5 行表头；总烧蚀列末值 `8.9726e19` |

---

## 8. 预算基础读取 summary（本 run，无 gate；单位与口径登记）

来源：`jorek008900.h5` 完整历史 + `abl_history.dat` + `macroscopic_vars.dat`。

| 项 | 数值 | 口径/备注 |
|---|---:|---|
| 烧蚀原子（累计末值） | `8.9726e19` atoms | `xtime_spi_ablation` 末值 − 首值；与 `abl_history.dat` `total_abl_number` 一致（差 `1.06e13`，≈0.012%） |
| 烧蚀率（末步） | `3.635e21` atoms/s | `xtime_spi_ablation_rate` 末值 |
| 率积分 vs 累计 | 比值 ≈ `19.8` | **口径差异已登记**：率序列按 “atoms/s × dt×t_norm” 积分不复现累计序列的逐索引对应；累计序列与 `abl_history` 一致。需生成侧确认率的索引/时间步口径，不做 gate |
| 电子注入（率积分） | `2.572e25` electrons | `xtime_Ne_imp`（电子/JOREK 时间）× `dt×t_norm` |
| 电离能量（累计） | `2.424e4`（JOREK 能量） | `xtime_E_ion` 末−首；率积分 `6.595e10`（同口径登记） |
| 辐射能量（累计） | `3.965e6`（JOREK 能量） | `xtime_radiation` 末−首；率积分 `9.742e12` |
| 全局能量闭合（末行残差） | `-8.988e15` | `-dEtotdt − sum(bnd fluxes+sources+dissipative)`；8908 行中 **2 行非有限**（记 `n_nonfinite`，不计入 max/median）；`median|res|=9.09e8` |
| `dparticles_dt` 闭合 | 末行 0（中位 0） | `Total − (Ions+Neutrals)`；另有 2 行非有限 |
| `particlecontent` | 总离子 `1.2138e21`、杂质 `1.6048e21`；inside+outside 残差 `5.0e11`（≈4e-10 相对） | 末行 |
| 边界粒子流 | `sum_last = -1.1188e28` | `bnd_particle_fluxes` 5 分量之和（末行，JOREK 粒子单位/JOREK 时间） |
| 场空间交叉校验（可选） | `dE_ion=-5.87e-5`、`rad_sink=2.47e-4`、`N_imp=2.97` | `projections007360` 的 P1 集总 FEM 积分（4125 节点），**只是投影通道交叉校验**，≠内部 `inj_source` |
| 杂质粒子预算 | **未闭合** | 需 `rho_imp` FEM 体积积分 + 边界通量 → **Phase B1 做**（已写入 CLI summary） |

---

## 9. 生成侧精确补字段清单（按优先级）

### P0（缺了就 fail-fast，A′ 不成立）

1. **`manifest.json`**（`<run>/export_v3/`）：`layout ∈ {node_dof0,full_dof}`、
   `cadence ∈ {fluid_step,substep_group}`、`source_object ∈ {inj_source,projection}`、
   `n_nodes`、`run`、`jorek_version`、`model_id`、`restart_indices`、`gap_indices`、
   `budget_layout`（本次应为 `history`）。
2. **每步 `/shards/start|end/<16 个 spi_*_arr>` + `n_spi_tot_start|end`**：
   数组已存在于每帧；生成侧需在导出 n 时同时写 `jorek{n}`（`shard_end=T_n`）与
   缓存的 `jorek{n−1}`（`shard_start=T_{n−1}`）。**这是本任务确认的第一优先缺项。**
   注意注入前 55 帧（索引 0–2700）没有 `spi_*_arr`：2710 帧必须显式写空数组
   （长度 0 + `n_spi_tot=0`）或从 2720 起导出（§2.1/§3）。
3. **区间元数据（双单位）**：`/time/{interval_start,end}_jorek`、
   `/time/{source_interval_start,end}_jorek`、四个 `*_seconds`、`t_norm`、`substeps`、
   单步时 `substep_group_id=n`；并带 `@step_index`、`@cadence` 属性。
4. **`/state/u_before`、`/state/u_after`**：同一文件的相邻两帧九场
   （`u_before=jorek{n}`、`u_after=jorek{n+1}`），布局按 manifest；`/source/S_used`
   写 `projections{n}`（或内部 `inj_source`），5 通道。
5. **`/provenance/source_object`**：`"inj_source"` 或 `"projection"` 二选一。

### P1（有则完整，缺则 warning/登记）

6. `/mesh/*`：`x/size/vertex/contain_node/neighbours` 可直接来自 `jorek{n}`；
   `boundary`（逐节点，已注册码 0/1/2/3/4/5/9/11/12/15/19/20/21）；`freeboundary`
   需**逐节点 `int8[nodes]`**（原始是全局 `'F'`）；`axis_node` 需**索引表
   `int64[n_axis]`**（原始是逐节点 flag，本帧 68 个 `'T'`）。`vertex` 需在
   manifest 写明 1-based。
7. `/condition/*`：61 维 `names/values/units/groups/provenance/film_eligible`；
   或用冻结词表分组的物理子集（`pellet_*`/`background_plasma`/`boundary_flags`）；
   `excluded_transport`（含 `d_perp_num`/`visco_num`/`eta_num` 若导出）必须
   `film_eligible=0`。
8. `/time/interval_*` 的 restart 帧必须**无隐式偏移**；若生成侧仍提前一步写出，
   在 manifest `restart_indices` 登记（loader/CLI 按 `drop|error` 处理）。

### P2（预算/诊断，不影响主路径加载）

9. 预算系列若进 step 文件：`/budget/xtime*`（12 个键，见 §7），并写
   `budget_layout`；否则验收 CLI 用 `--raw-run-dir` 从原始 `jorek*.h5` 读
   完整历史（当前路径）。
10. `grid_to_wall`、`bn/法向`、`c_s`、`T_e/n_e` 边界诊断：**用户侧调查**，
    不入 loader 硬校验；出现则登记，缺失仅 warning（已实现）。
11. 低分辨率 marker 流（可选，1 个 run）：`weight/P_imp/x/v/i_life` + 组元数据，
    过滤 `i_life>0`、`P_imp` 量化规则写入 manifest（注意本帧 `t_birth` 全 0）。

---

## 10. 下游接口（本次交付）与验收命令

| 文件 | 作用 |
|---|---|
| `src/data/step_dataset.py` | `StepRunDataset`：读 A′ 布局，构造 `(u_k, p_{k−1}=shard_start, S_k) → u_{k+1}`；`require_single_step`（断言 `dt_step==dt_source`，tol 默认 1e-6，manifest 可覆盖）；fail-fast：时间严格递增/区间链接、`source_end==interval_start`、shard 缺失/形状、未登记 boundary 码、condition 组非法；warning：缺 boundary/condition/seconds/预算布局；支持 dof0/full-dof、静态 mesh、manifest restart/gap、raw run 交叉校验 |
| `src/data/budget.py` | 预算基础读取：`xtime_spi_ablation[_rate]`/`abl_history.dat` 逐 step 差值、`xtime_Ne_imp`、`xtime_E_ion[_power]`、`xtime_radiation[_power]`；`macroscopic_vars.dat` 的 `energy_conservation`/`dparticles_dt`/`particlecontent`/`bnd_particle_fluxes`；可选投影 FEM 积分；杂质预算标记 Phase B1 |
| `tools/check_export_v3.py` | 合同 + 配对 + 覆盖统计 CLI：窗口/步数、raw 对齐、restart、shard 分布、boundary 码统计、缺失诊断 warning、condition 组、文件大小、linkage、预算 summary；输出 JSON + 人类摘要 |
| `tests/test_step_dataset.py`（28，含 1 skip-if-absent）、`tests/test_budget.py`（7）、`tests/test_check_export_v3.py`（3） | 合成 fixture 必测 + 真实数据 skip-if-absent |

命令（smoke 数据到位后直接跑）：

```bash
# 合同/配对/单步/覆盖统计 + 预算（用原始 run 的完整历史）
.torch-cuda-env/bin/python tools/check_export_v3.py \
  --run-dir <run> --raw-run-dir <raw-run> \
  --out .scratch/n/export_v3_check.json

# loader 冒烟（require_single_step=True 是 Phase B/C 硬门禁）
.torch-cuda-env/bin/python -c "
from src.data.step_dataset import StepRunDataset
ds = StepRunDataset('<run>/export_v3', raw_run_dir='<raw-run>', require_single_step=True)
print(len(ds), ds.report.to_dict()['dt_step'])
"
```

**验收判定**：合同/配对/单调性 fail-fast 全通过（exit 0）才算“可进 Phase B/C”；
预算残差只登记数值与口径，不设 performance gate。

---

## 11. 开放问题（按风险排序）

1. **A′ 导出未交付（高）**：本报告无法对真实 A′ 文件做逐键核对；loader/CLI 就绪，
   首次 smoke 落盘后需重跑本报告 §2–§7 并更新缺字段状态。
2. **`shard_start` 生成协议（高）**：生成侧必须缓存上一步 `spi_*_arr`；若只导
   `jorek{n}` 单帧，A′ 主路径无法成立（loader fail-fast）。
3. **烧蚀率口径（中）**：`xtime_spi_ablation_rate` 的 dt 加权积分与累计序列
   比值 ≈20（§8）；需生成侧确认率的索引/时间步语义。累计序列与 `abl_history.dat`
   已互证，暂不阻塞。
4. **`freeboundary`/`axis_node` 形状转换（中）**：原始为全局 flag/逐节点 flag，
   A′ 要逐节点数组/索引表；转换规则需随 smoke 一起核对。
5. **restart 帧无偏移导出（中）**：新导出应直接写 `interval_*`；若仍偏移，
   必须登记，loader 默认 drop。
6. **`grid_to_wall` 等边界诊断缺位（中，用户侧）**：loader 预留接口（warning+登记），
   不阻塞合同验收。
7. **`gh` CLI（流程）**：本环境缺失，未执行 issue 认领/关闭；已在实现进度登记。
