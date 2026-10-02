# SPI analytic baseline — 2026-09-27

Phase-2 gate evidence for the analytic backbone port of the JOREK SPI model.
Question: how close is the ported model 3 (Parks/NGS neon branch) to the
recorded shard ablation before any learning?

## 阅读约定：术语与指标（先读这一节）

**帧（frame）**：JOREK 的一次输出时刻，包含同编号的 `jorek######.h5`（九场状态 +
`spi_*_arr` 碎片状态）、`part######.h5`（marker 快照）与 `projections######.h5`（五场投影）。
本文所有对照都在"帧"这个粒度上做：预测值与同一帧记录值比较。

**碎片（shard / `type_SPI`）**：JOREK 的弹丸碎片，一个碎片一条记录，字段见 `spi_*_arr`。
"逐 shard"= 每个碎片单独比较；"聚合"= 同一帧所有碎片求和。

**表（nonlocal table）**：`nonlocal_array.dat` 的内容，1000 行
`(index, psi_n, <T_e> [JOREK 单位], <n_e> [JOREK 单位])`，`<·>` 为磁通面体积平均。
它只服务于非局域烧蚀分支 `abl = max(local, nonlocal)`。该文件是**调用快照**（每次
`int3d_new` 覆盖），不属于某一帧；本工作改为按帧重建（`src/data/nonlocal_table.py`）。

**指标定义**（`pred` = 本仓库解析模型算出的烧蚀率，`rec` = JOREK 记录值 `spi_abl_arr`，单位 atom/s）：

| 指标 | 定义 | 读法 |
|---|---|---|
| 中位比 median ratio | `median(pred/rec)`，在两边都 >0 的 shard-帧上取 | 1.0 = 无系统偏差；0.5 = 系统性低估一半；2.0 = 高估一倍 |
| log-corr | `ln(pred)` 与 `ln(rec)` 的 Pearson 相关系数 | 幂律形状一致性；>0.98 表示"随参数变化的方式几乎一样" |
| within-2x | `pred/rec ∈ [0.5, 2]` 的 shard 占比 | 工程上常用的"够用"比例，越高越好 |
| 总烧蚀率比 sum ratio | `Σ pred / Σ rec`（同一批帧的全部 shard 求和） | 物质预算；1.0 = 总烧蚀原子数一致 |
| MAE | 平均绝对误差（半径用 m，速度用 m/s） | 绝对误差，不带符号 |

## Method

- Data: `scan_1d22_ang15_5_massloss_neon_v400`, frames at positions 5 / 20 / 40
  of the aligned frame list (all post-`t_ns`).
- Local fields: `rho`, `rho_imp`, `T_e` sampled at every alive shard position
  with inverse-distance k-NN (`k=4`) over the 4125-node poloidal mesh.
- Analytic ablation: `src/physics/spi_model.py::parks_ablation_neon`
  (`spi_abl_model = 3`, neon branch, `use_marker` electron-density convention
  `ne = n - n_imp`, `spi_abl_mag_reduction = false`).
- Reference: the recorded `spi_abl_arr` of the same frame.
- Command: `.torch-cuda-env/bin/python tests/check_analytic_spi.py
  --run scan_1d22_ang15_5_massloss_neon_v400 --position <p> --k 4`.

官方后处理路线（对照，不替代上面的方法）：`util/auto_spi_diagnostics.sh` 用
`convert2diagno_spi.py` + `jorek2_diagno_spi` 逐帧导出碎片轨迹
(`fragments_position_*.csv`)，并在最后一个 restart 上导出 `abl_history.dat`
（逐时间步的注入率与累计烧蚀原子数，直接 dump restart 里的 `xtime_spi_ablation*`）。
它给出的是**聚合、逐时间步**的量，不给出逐 shard 值，也不给出非局域表；逐 shard 与
非局域表的真值仍来自各帧的 `spi_abl_arr` 与按帧重建。

现状提醒：这两个 run 目录里已有的 `abl_history.dat` 是坏文件（除 t=0.02 一行外全为 0），
不能用来校核；可用的逐时间步历史在每个 `jorek*.h5` 的
`xtime_spi_ablation` / `xtime_spi_ablation_rate`（`(index_now, 1)`）里，
`RunFrameStore` 已读取它们（`ablation_cumulative` / `ablation_rate_total`）。

JOREK source: `/home/ac_desktop/JOREK_code/jorek` working tree at
`54c675bcc`; the nonlocal ablation tables (`nl_Psi`, `nl_avg_Te`, `nl_avg_ne`)
are **not** exported in the HDF5 data, so the port implements the local branch
only.

## Results

| frame position | alive shards | median T_e [eV] | median n_e [m^-3] | median analytic/recorded | log-corr | within 2x |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 846 | 14.9 | 1.66e18 | 1.47 | 0.989 | 91.1% |
| 20 | 846 | 24.6 | 8.80e18 | 1.28 | 0.994 | 93.0% |
| 40 | 846 | 174.3 | 3.44e19 | 0.49 | 0.882 | 42.8% |

## Reading

- Early in the event the local Parks branch tracks the JOREK ablation with a
  small multiplicative bias (1.3-1.5x) and a log-correlation of ~0.99; the
  remaining bias is consistent with k-NN interpolation error and with the
  recorded value coming from JOREK's internal sub-steps.
- Late in the event the analytic value underestimates by ~2x (43% within 2x),
  which is the expected signature of the **nonlocal ablation branch**
  (`spi_abl = max(local, nonlocal)`), whose lookup tables are not in the
  exported data.

## Consequences

1. The analytic backbone is a usable baseline: it needs no learning to reach
   ~0.99 log-correlation in the early phase.
2. The learned source/dynamics correction is required for the late phase; this
   is exactly the "analytic baseline + learned correction" design of ADR-0011.
3. If a faithful nonlocal branch is wanted, the JOREK side must export
   `nl_Psi`, `nl_avg_Te`, `nl_avg_ne` (and the index convention) with the frames.

## Update 2026-09-29 — the nonlocal branch is computable offline (late-phase gap closed)

Consequence 3 above is superseded: no JOREK-side export is needed.  Two
findings:

1. `nonlocal_array.dat` is rewritten on every `int3d_new` call, so only the
   last snapshot survives on disk.  That snapshot does **not** belong to the
   saved frames: using it as if it did (JOREK's own lookup + `max`) overshoots
   the recorded ablation by 60-150x on late frames of some runs, and is colder
   than the local value in others.  The file is a diagnostic, not a per-frame
   table.
2. The table is a pure function of a saved frame.  `src/data/nonlocal_table.py`
   rebuilds it exactly as `diagnostics/mod_integrals3D.f90` does:
   `n_e = rho + (m_i/m_imp * Z_imp - 1) rho_imp` with the marker-projected
   `Z_imp = projection Z_imp / projection N_imp`, 40 psi surfaces
   (`<T_e> = sum(n_e T_e dV) / sum(n_e dV)`, `<n_e> = sum(n_e dV) / sum(dV)`)
   and linear interpolation to the 1000-point grid.  Element connectivity comes
   from the `vertex` dataset (`contain_node` is all zeros in these files and
   must not be used).

`tests/check_nonlocal_ablation.py --table frame` samples post-injection frames
per run and applies JOREK's nearest-index lookup + `max(local, nonlocal)` with
that rebuilt table (8 frames per run, `k=4`).  Column meaning: "median" =
中位比 `median(pred/rec)`; "within-2x" = 比例 `pred/rec ∈ [0.5,2]`; "sum ratio"
= `Σpred/Σrec`（物质预算）；每个数字都是在"两边都 >0 的 shard-帧"上统计。

| run | local median | local within-2x | local sum ratio | +nonlocal median | +nonlocal log-corr | +nonlocal within-2x | +nonlocal sum ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| scan_1d21_ang12_massloss_neon | 0.878 | 82.6% | 0.651 | 1.087 | 0.996 | 94.9% | 1.036 |
| scan_1d21_ang12_massloss_neon_v200 | 0.908 | 82.2% | 0.640 | 1.035 | 0.996 | 86.8% | 1.013 |
| scan_1d21_ang12_massloss_neon_v300 | 0.933 | 79.9% | 0.678 | 1.025 | 0.994 | 85.5% | 0.941 |
| scan_1d22_ang12_massloss_neon | 0.601 | 57.2% | 0.539 | 1.016 | 0.997 | 95.7% | 1.013 |
| scan_1d22_ang12_massloss_neon_v200 | 0.643 | 60.4% | 0.627 | 1.006 | 0.993 | 88.8% | 0.996 |
| scan_1d22_ang15_5_massloss_neon_v300 | 0.633 | 57.1% | 0.589 | 1.027 | 0.997 | 93.1% | 1.009 |
| scan_1d22_ang15_5_massloss_neon_v400 | 0.738 | 58.3% | 0.535 | 1.073 | 0.982 | 90.8% | 1.012 |
| scan_2d21_ang12_massloss_neon_v300 | 0.847 | 82.0% | 0.613 | 1.039 | 0.994 | 92.9% | 1.009 |
| scan_2d21_ang15_5_massloss_neon_v300 | 0.771 | 81.0% | 0.635 | 1.029 | 0.996 | 90.5% | 0.972 |

(`scan_1d22_ang12_massloss_neon_v300` is a byte-identical copy of `_v200` and
is omitted; see the progress log for the duplicate-run finding.)

New reading: the local-only branch is not just biased but incomplete — its late
deficit is the missing `max(local, nonlocal)`.  With the rebuilt table the
analytic backbone reproduces the recorded per-shard ablation with median ratio
1.0, log-correlation >= 0.98 and 85-96% of shards within 2x on every run, and
the total ablated-atom rate within ~5%.  The learned correction therefore
starts from a faithful backbone instead of absorbing a structural model error.
