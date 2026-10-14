# 细粒度数据导出规范 v3（交付 JOREK 数据生成方）

**状态**：current / frozen（2026-10-04）。语义依据 [data-contract-v3.md](data-contract-v3.md)、
[ADR-0018](../docs/adr/0018-architecture-first-and-physics-structured-build.md)、
[ADR-0019](../docs/adr/0019-recorded-output-timing-contract.md) 与
`roadmap_v2.md` Phase A。ADR-0001–0017 是 v1 历史，不作为本规范依据。

**目标**：让数据生成方按 JOREK 交错格式导出**每流体步自描述区间**，使
`(u_k, p_{k−1}, S_k) → u_{k+1}` 可在无未来帧泄漏的前提下监督与验收。
**起步范围**：先 1–2 个 run（首选 `scan_1d22_ang12_massloss_neon`），跑通后再扩。

**导出档位（A′，2026-10-04 用户确认）**：

- **必需（每流体步）**：九场 `u_before/u_after`、源 `S_used`、**碎片 shard 状态**
  `shard_start/shard_end`（`spi_*` 数组，≤~846 个碎片，约 0.03–0.11 MB/步）、
  区间/单位/边界/condition 元数据。
- **可选（低分辨率诊断）**：完整 marker（1e6 宏粒子）只对 1 个 run 按粗粒度导出
  （每帧或每 5–10 步，存活过滤 + float32 + `P_imp` 量化），用于 ADAS/电荷态/
  解析骨干诊断。主路径（ADR-0018 B1）不把 marker 当输入，S 才是监督目标。

---

## 1. 目录与文件布局

```
<run>/export_v3/
  manifest.json                  # run 级：provenance、cadence 登记、condition 分组、restart 登记
  steps/step_{n:06d}.h5          # 每个导出区间一个文件（默认 n = 区间右端点流体步号）
  steps/step_{n:06d}.json        # 可选：纯文本副本的轻量元数据（不含数组）
```

每个 `step_{n:06d}.h5` 的 HDF5 结构：

| 路径 | 形状 / 类型 | 语义 |
|---|---|---|
| `@step_index` | int64 属性 | 源步号 `n`（与 `projections{n}`/`part{n}` 的编号一致） |
| `@cadence` | 字符串属性 | `"fluid_step"` 或 `"substep_group"` |
| `/time/interval_start_jorek` | float64 | `T_n`：**场推进区间**起点（= `u_before` 时间） |
| `/time/interval_end_jorek` | float64 | `T_{n+1}`：场推进区间终点（= `u_after` 时间） |
| `/time/source_interval_start_jorek` | float64 | `T_{n−1}`：**源采样窗口**起点（= `shard_start` 时间） |
| `/time/source_interval_end_jorek` | float64 | `T_n`：源采样窗口终点（= `shard_end` 时间） |
| `/time/*_seconds` | float64 | 上述各端点的物理秒（`× t_norm`） |
| `/time/t_norm` | float64 | 秒 / JOREK 时间单位（同帧值） |
| `/time/substeps` | int64 | 组内流体步数（`fluid_step` 时为 1） |
| `/time/substep_group_id` | int64 | 子步组编号（`cadence=fluid_step` 时等于 `n`） |
| `/state/u_before` | float64 `[9, 4, 1, nodes]` | `u_n`（时刻 `T_n`，场推进区间起点） |
| `/state/u_after` | float64 `[9, 4, 1, nodes]` | `u_{n+1}`（时刻 `T_{n+1}`） |
| `/source/S_used` | float64 `[5, 4, 1, nodes]` | `S_n`：源窗口 `[T_{n−1}, T_n]` 的时间平均（内部 `inj_source` 语义，见 §3） |
| `/mesh/x` | float64 `[2, 4, 1, nodes]` | 几何米；与场/源同帧同索引 |
| `/mesh/size` | float64 `[4, 4, n_elem]` | 单元几何权重 |
| `/mesh/vertex` | int32 `[4, n_elem]` | 单元-节点连接（JOREK 1-based 约定需在 manifest 写明） |
| `/mesh/contain_node` | int32 | 节点-单元包含关系 |
| `/mesh/neighbours` | int32 | 可选：单元邻接（缺省时由 vertex 重建） |
| `/mesh/boundary` | int32 `[nodes]` | 逐节点 boundary 代码（真实数据中 198/4125 个非零） |
| `/mesh/freeboundary` | int8 `[nodes]` | 逐节点 0/1（与 `boundary` 同源同索引） |
| `/mesh/axis_node` | int64 `[n_axis]` | 磁轴节点（度量奇异单独处理用） |
| `/shards/start/<name>` | float64 `[n_spi_start]` | 源窗口起点的碎片状态（`spi_*_arr`；时刻 `T_{n−1}`），必需 |
| `/shards/end/<name>` | float64 `[n_spi_end]` | 源窗口终点的碎片状态（时刻 `T_n`），必需 |
| `/shards/n_spi_tot_start` / `_end` | int32 | 两端碎片数 |
| `/markers/…` | 见 §2.3b | **可选**：低分辨率 marker 诊断流（仅 1 个 run，粗粒度） |
| `/provenance/source_object` | 字符串 | `"inj_source"`（解析 shard 密度源）、`"marker_feedback"`（记录投影五通道；marker run 的能量/辐射/电荷真实源）或 `"analytic_shard"`（B1 确定性主项）；不得混用 |
| `/provenance/jorek_version` | 字符串 | RCS/版本号，可追溯生成侧代码 |
| `/provenance/model_id` | int32 | `502` |
| `/provenance/run` | 字符串 | run 名 |
| `/provenance/namelist_sha256` | 字符串 | 输入 namelist 摘要（provenance 复现） |
| `/condition/names` | 字符串数组 | 条件名（与 `values` 对齐） |
| `/condition/values` | float64 `[n_cond]` | 数值 |
| `/condition/units` | 字符串数组 | 单位 |
| `/condition/groups` | 字符串数组 | 分组标签（词表见 §4） |
| `/condition/provenance` | 字符串数组 | `h5` / `namelist` / `default` |
| `/condition/film_eligible` | int8 `[n_cond]` | 是否允许进入浅层 FiLM（§4） |

> 时间单位规则（ADR-0019）：JOREK 时间单位与物理秒必须都写出，且用同一 `t_norm`；
> 只写一种单位视为不合格。
>
> **错步语义（必须显式）**：`S_n` 采样在 `[T_{n−1}, T_n]`，被**下一段**场步
> `u_n → u_{n+1}` 消耗（ADR-0019）。因此每条记录同时给出**源窗口**
> `[T_{n−1}, T_n]`（`shard_start → shard_end`）与**场推进区间** `[T_n, T_{n+1}]`
> （`u_before → u_after`）；两个区间的端点必须分别写时间戳，禁止只给一个含糊
> "interval"。

---

## 2. 每流体步最小字段集

### 2.1 区间对象（必需字段）

| 字段 | 语义 | 必需 |
|---|---|---|
| `interval_start` / `interval_end` | 场推进区间 `[T_n, T_{n+1}]`（`u_before`/`u_after` 的时刻，双单位） | ✅ |
| `source_interval_start` / `source_interval_end` | 源采样窗口 `[T_{n−1}, T_n]`（`S_used`、`shard_start`、`shard_end` 的时刻，双单位） | ✅ |
| `u_before` | `u(T_n)` 九场（场推进起点） | ✅ |
| `S_used` | `S_n`：窗口 `[T_{n−1}, T_n]` 的源，驱动 `u_before → u_after` | ✅ |
| `u_after` | `u(T_{n+1})` 九场 | ✅ |
| `shard_start` | `T_{n−1}` 的碎片（`spi_*`）状态 | ✅ |
| `shard_end` | `T_n` 的碎片状态 | ✅ |
| `marker_start` / `marker_end`（如导出） | 低分辨率 marker 诊断流 | ❌（可选） |

**禁止**：用平均值/中心时间含糊代替区间端点；把 `u_before → u_after` 误配 `S_{n−1}`
或 `S_{n+1}`（一步泄漏/滞后）；缺源帧不得零填充（"无源" 与 "零源" 不同）；
`S_used`、场帧与 `boundary` 必须来自同一网格、同一节点顺序。

### 2.2 场与源

* 场通道顺序固定 9 通道（schema §1）；源通道顺序固定 5 通道（schema §3.1）；
* 数组布局沿用 JOREK 原生 `[channels, dof, harmonics, nodes]`（2D run 取 `harmonics=1`）；
  若为省体积只导出 dof-0 节点值，必须在 manifest 写明 `layout="node_dof0"`，并保证
  `boundary`/`freeboundary` 仍按全节点导出；
* 单位：状态场沿用 JOREK 内部单位（与现有 `jorek*.h5` 一致）；源沿用现有
  `projections*.h5` 的归一化口径，并在 manifest 登记换算。

### 2.3 碎片状态 `shard_start` / `shard_end`（必需，每步）

碎片（`type_SPI`）才是主架构里的 `p`。沿用 `jorek*.h5` 的 `spi_*_arr` 字段
（长度 `n_spi_tot`，当前数据 ≤846）：

| 数据集 | 形状 | 说明 |
|---|---|---|
| `n_spi_tot` | 标量 | 本帧碎片数 |
| `spi_R_arr` / `spi_Z_arr` / `spi_phi_arr` | `[n_spi]` | 位置 |
| `spi_Vel_R_arr` / `spi_Vel_Z_arr` / `spi_Vel_RxZ_arr` | `[n_spi]` | 速度 |
| `spi_radius_arr` / `spi_abl_arr` / `spi_species_arr` / `spi_vol_arr` | `[n_spi]` | 半径、烧蚀率、species、体积 |
| `spi_psi_arr` / `spi_grad_psi_arr` / `spi_phi_init_arr` | `[n_spi]` | 磁通与初始化量 |
| `spi_psi_arr_drift` / `spi_grad_psi_arr_drift` / `spi_vol_arr_drift` | `[n_spi]` | drift 分量 |
| 组级元数据 | 标量 | 单位、species 表、时间戳 |

碎片数组必须与同一 step 的 `u_before/after`、`S_used`、`boundary` 同网格、
同节点顺序、同时间语义。

### 2.3b 低分辨率 marker 诊断流（可选，仅 1 个 run）

主路径不需要完整 1e6 marker。若需诊断 ADAS/电荷态/解析骨干，对 1 个 run 按每帧
或每 5–10 步导出一次，存活过滤 + float32 + `P_imp` 量化（uint8，行和=1）。
字段：`weight`、`P_imp`、`x`、`v`、`i_life`、组级 `Z/type/adas_suffix`；
manifest 必须写明 cadence、过滤与量化规则。**禁止**把它当作主路径输入。
若以后启用"学习 marker→projection 映射"回退路线（ADR-0016 候选 c），再对该 run
加密导出。

### 2.4 元数据（每文件必需）

`step_index`、`cadence`、`interval_start/end`（双单位）、`t_norm`、`source_object`、
`jorek_version`、`run`、`model_id`、`namelist_sha256`。
文件名编号不得作为隐式时序约定（ADR-0019 决定 3）。

---

## 3. 源对象 provenance（必填）

（ADR-0020 决定 1 修订：marker run 里记录投影就是真实场源，不再是"仅监督"）

| 取值 | 含义 | 允许用途 |
|---|---|---|
| `inj_source` | model502 内部解析 shard 密度源：Gauss 点沉积进 RHS、对场显式（杂质密度通道） | 解析 shard 源与其交叉校验 |
| `marker_feedback` | 记录投影五通道（`dE_ion/rad_sink/Z_eff/Z_imp/N_imp`）：**marker run 的能量/辐射/电荷真实场源**（`mod_elt_matrix.f90:989-1003,1129,1728`） | B1/B2/C 的共享 S 对象（训练目标 = 记录值；闭环推理 = B1 自生成值） |
| `analytic_shard` | B1 的确定性解析 shard 沉积主项（`inj_source` 语义） | 交叉校验与消融 |

导出 `S_used` 时必须明确其一并登记；若两者都要，写成两个数据集/两个文件，
并在 manifest 标明对应关系与截断误差评估计划。marker run（`use_marker=.t.`）
的主监督对象是 `marker_feedback`。

---

## 4. 逐节点 boundary + condition 分组（两条 review 字段）

### 4.1 boundary / freeboundary（stencil 边界执行必需）

* 每个 step 文件都带逐节点 `boundary` 代码与 `freeboundary`；
* 必须与同文件的 `u_before/u_after/S_used` **同源、同索引、同节点顺序**；
* 不允许重排/压缩节点顺序后不登记（会立刻破坏 stencil 邻接与边界映射）；
* 边界代码到 BC 的映射表在生成侧给出（供 `mod_boundary_conditions.f90` /
  `mod_boundary_matrix_open.f90` 口径对照，ADR-0018 决定 11）。
* **归一化规则（ADR-0020 决定 7）**：原生 `jorek*.h5` 的 `freeboundary` 是全局
  `'F'/'T'` 标志，不是逐节点 int8；导出规范要求逐节点 `int8[nodes]` 时，
  按全局标志展开并在 manifest 登记
  `freeboundary_normalization="global_flag_expanded"`；
  `axis_node` 在当前 2D run 不存在（`axis_node=F`），缺省时不导出/不推断，
  若未来 run 携带磁轴则在 manifest 登记 `axis_node_present=true`。

### 4.2 condition 分组词表（浅层 FiLM 必需）

condition 与场/源在同一文件内导出，每条带 `group` 与 `film_eligible`：

| group | 语义 | 典型条目 | FiLM |
|---|---|---|---|
| `pellet_mass` | 弹丸质量/粒子数 | `n_spi`, `pellet_density`, `mass` | ✅ 仅浅层 |
| `pellet_velocity` | 注入速度 | `v0`（弹丸初速） | ✅ 仅浅层 |
| `pellet_angle` | 注入角度 | `ang` | ✅ 仅浅层 |
| `pellet_species` | 杂质种类/电荷态 | 物种名、`Z`、`adas_suffix` | ✅ 仅浅层 |
| `background_plasma` | 背景等离子体场景 | `T_0/Ti_0/Te_0`、背景密度 | ✅ 仅浅层 |
| `boundary_flags` | 边界口径 | `freeboundary`, `bc_natural_open`, `resistive_wall` | ✅（只影响边界处置） |
| `excluded_transport` | 输运/数值系数、run 指纹 | `eta`, `visco`, `ZK_*`, `D_perp`, `central_density`, `F0`, `R_domm` | ❌ 禁止 |
| `excluded_numeric` | 数值方案 | `time_evol_*`, `tstep` | ❌ 禁止 |

规则（ADR-0018 决定 6/12，roadmap_v2 §5.5、§5.11）：

* FiLM/AdaLN 只允许作用于**最浅编码层**（输入嵌入 + 第一层），恒等初始化（γ=1, β=0）；
* 深层 stencil 层不接受全局 condition；`excluded_*` 组不得进入任何调制路径；
* 分组标签必须能让下游做分组置换检验（打乱某组后局部响应形状不变）。

---

## 5. cadence 与规模（A′ 与可选 marker 流）

### 5.1 实测参照（`scan_1d22_ang12_massloss_neon`）

| 文件 | 实测大小 |
|---|---|
| `jorek007360.h5`（九场全量导出，含大量诊断标量） | 10.1 MB |
| `projections007360.h5`（5 通道全 dof） | 1.57 MB |
| `part07360.h5`（1e6 slots；存活 181,990） | 172 MB |
| 整个 run 目录（675 场 + 620 粒子 + 621 投影） | 107 GB |

按全 dof、float64 估算：场 ~2.4 MB/步 + 源 ~0.66 MB/步 ≈ 3.0 MB/步 ⇒
8900 步 ≈ 27 GB/run；只导 dof-0 节点值则 ≈ 0.76 MB/步 ⇒ ~7 GB/run。
粒子若按存活过滤 + float32（`P_imp` 可 `uint8` 量化，行和=1）约 4–16 MB/步
（峰值），SPI 窗口数千步 ⇒ 数十 GB/run 量级。

### 5.2 A′ 主路径（必需，推荐）

每流体步：场 + 源 + shard + 边界/condition/区间元数据。
按 dof-0 节点值估算：场+源 ≈0.76 MB/步，shard ≈0.03–0.11 MB/步
⇒ 1 个 run（~8900 步）约 **7–9 GB**；全 dof 约 27 GB。
这是新架构 B1/B2 训练与闭环的完整数据集。

### 5.3 可选 marker 诊断流（1 个 run）

* **M1（默认）**：每帧（约 10 步）导出过滤 marker ≈ 4–16 MB/帧 ⇒ 4–14 GB/run；
* **M2**：每 5–10 步导出过滤 marker ≈ 8–30 GB/run；
* **M3**：只导出减量化 Eulerian 矩（逐节点电荷态密度/辐射权重），体积更小但需另定格式。

主路径不受 M1/M2/M3 选择影响；若以后需要 marker 回退路线，再加密导出。
所有档位的 `S_used` 必须是区间源（不是"整段平均"）。

---

## 6. 交付与验收清单

数据生成方交付时逐项自检：

1. **时间戳严格递增**：同一 run 内 `interval_start`/`interval_end` 单调、无重叠、无缺口
   （gap 必须登记到 `manifest.json`；旧 run 的登记 restart 事件按数据契约 v3 §4.3
   处理，新导出走严格去重模式）；
2. **区间对象一致**：同一 step 的 `u_before/after`、`S_used`、`shard_start/end`、
   `boundary/freeboundary` 来自同一窗口、同一网格、同一节点顺序；
3. **配对正确**：`S_used`（窗口 `[T_{n−1}, T_n]`）驱动 `u_before → u_after`
   （区间 `[T_n, T_{n+1}]`）；`shard_start`/`shard_end` 时间 = `source_interval_start/end`
   （可选 marker 流同规则）；
   `interval_start` = `source_interval_end`；restart 帧无隐式偏移；
4. **预算可算**：用同一 FEM 基函数可复算原子/能量/辐射预算（生成侧给出核算脚本或公式）；
5. **condition 合规**：分组标签齐全、`excluded_*` 未进入 FiLM 路径；
6. **provenance 可回溯**：`source_object`、版本、namelist 摘要齐全；
7. 提供 3 帧小型 fixture（可入测试数据区）供下游契约测试使用。

下游（本仓库）验收命令：

```bash
.torch-cuda-env/bin/python -m pytest tests/test_output_contract.py tests/test_paired_dataset.py -q
.torch-cuda-env/bin/python tools/rebuild_ehl2_pairs_v3.py --run-dir <run> --out-dir <new-dir>
```

## 7. 开放问题（需生成方确认）

1. 可选 marker 诊断流的 cadence（M1/M2/M3）与是否纳入本次生成；
2. restart 帧的生成顺序能否直接产出无偏移的 `interval_*`（新导出应能；旧 run 的登记
   事件规则见数据契约 v3 §4.3 与 `data/restart_events.json`）；
3. 若未来启用 marker 回退路线，需要的 marker 密度与组内累计源导出形式；
4. 能量/辐射预算的离散核算基函数与口径（与 JOREK 内部一致的对账脚本）。
