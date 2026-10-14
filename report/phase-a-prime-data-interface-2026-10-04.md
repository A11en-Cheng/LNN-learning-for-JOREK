# Phase A′ 数据接口交付：export_v3 loader + 验收 CLI + 预算基础读取（2026-10-04）

**状态**：current。**依据**：ADR-0018/0019、`roadmap_v2.md` Phase A、
`report/data-contract-v3.md`（冻结）、`report/data-export-spec-v3.md`（冻结）。
ADR-0001–0017 为 v1 历史，未作为依据。

**范围**：只做数据接口（合同核对、loader、验收 CLI、预算基础读取与闭合登记）。
**未做**：训练、闭环、GNN/谱分支、边界诊断采集（用户侧）、JOREK 物理重写、
legacy 测试/roadmap/ADR 改动。`gh` 本环境缺失，未走 issue 流程。

本轮同时产出 **字段审计报告**
[`phase-a-prime-data-audit-2026-10-04.md`](phase-a-prime-data-audit-2026-10-04.md)：
扫描范围内**尚无 A′ 逐步导出 run**（无 `export_v3/`、无 `step_*.h5`），该报告对
生成侧来源文件做了逐字段清点并交付补字段清单（尤其 `spi_*_arr` 双端导出）。

---

## 1. 改动文件清单

| 文件 | 类型 | 内容 |
|---|---|---|
| `src/data/step_dataset.py` | 新增 | A′ `StepRunDataset`：`(u_k, p_{k−1}=shard_start, S_k) → u_{k+1}`；`require_single_step`；fail-fast/warning 规则；dof0/full-dof；静态 mesh；manifest restart/gap；raw run 交叉校验 |
| `src/data/budget.py` | 新增 | 预算基础读取：烧蚀/电子注入/电离/辐射、`abl_history.dat`、`macroscopic_vars.dat` 全局闭合、可选投影 FEM 积分；杂质预算标记 Phase B1 |
| `tools/check_export_v3.py` | 新增 | 验收 CLI：合同 + 配对 + 覆盖统计 + 预算 summary，JSON + 人类摘要 |
| `tests/conftest.py` | 扩展 | A′ 导出 fixture（`write_export_step`/`write_export_v3_run`） |
| `tests/test_step_dataset.py` | 新增 | loader 合成 fixture 必测（28 个，含 real skip-if-absent） |
| `tests/test_budget.py` | 新增 | 预算读取器合成数据必测（7 个） |
| `tests/test_check_export_v3.py` | 新增 | CLI 端到端（3 个） |
| `tests/README.md` | 更新 | 新增 A′ 测试分组与运行命令 |
| `report/README.md` | 更新 | 新增两份 A′ 报告索引行 |
| `report/phase-a-prime-data-audit-2026-10-04.md` | 新增 | 字段审计 + 生成侧补字段清单 |
| `report/phase-a-prime-data-interface-2026-10-04.md` | 新增 | 本报告 |
| `report/implementation-progress.md` | 更新 | 追加本轮交付条目 |

未改动：`tests/legacy/**`、`roadmap_v2.md`、`docs/adr/**`、旧 manifest/checkpoint/
评测 JSON。`src/data/output_contract.py`、`src/data/paired_dataset.py` 保持原样
（legacy 帧级路径语义不变），`step_dataset.py` 复用其 `ContractError`、
`TIME_TOL_JOREK`、`assert_strictly_increasing`、`resolve_step_times`、`TimingIssue`。

---

## 2. loader 语义（`src/data/step_dataset.py`）

### 2.1 记录与配对

每个 `step_{n:06d}.h5` 自描述：

```
u_before = u(T_n)            field interval [T_n, T_{n+1}]
u_after  = u(T_{n+1})
S_used   = S_n               源窗口 [T_{n−1}, T_n]（projections{n} 或内部 inj_source）
shard_start = p_{n−1}        = 源窗口起点（= part{n} 语义）
shard_end   = p_n            = 源窗口终点
```

样本 `(u_k, p_{k−1}=shard_start, S_k) → u_{k+1}=u_after`，与 ADR-0019 唯一配对一致；
禁止 `source[:, k+1]` 式一步未来配对（旧管线错误）。

### 2.2 `require_single_step`

* `require_single_step=True`（Phase B/C 硬门禁）断言
  `|dt_step − dt_source| ≤ single_step_tol`；tol 默认 `1e-6` JOREK 时间单位，
  可由 manifest `single_step_tolerance_jorek` 覆盖并写入报告；
* 帧级旧数据（`dt_step = dump 间隔 >> dt_source = 一个流体步`）会被直接拒绝
  —— 正/反例均有测试；
* 诊断/覆盖场景可显式 `--allow-multi-step` 关闭，报告里记录 `require_single_step=false`。

### 2.3 fail-fast（报错即不可进 Phase B/C）

1. `jorek` 时间（`interval_start`）严格递增、`dt_step/dt_source > 0`；
2. 每步 `source_interval_end == interval_start`（S_n 驱动 `u_n→u_{n+1}`）；
3. 相邻步区间链接 `interval_end[k] == interval_start[k+1]`；索引跳跃必须登记
   `gap_indices`，未登记缺口直接报错；
4. `shards/start|end/<16 个 spi_*_arr>` 缺失或 `n_spi_tot`/长度不一致；
5. `boundary` 出现未登记码（注册表 0/1/2/3/4/5/9/11/12/15/19/20/21，B0 §3/§9）；
6. `condition` 组不在冻结词表、`excluded_*` 进入 FiLM（`film_eligible=1`）、
   names/values/groups 长度不一致；
7. `require_single_step` 违反；
8. manifest 缺 `layout`/`cadence`/`source_object` 或取值非法；
9. 双单位时间戳（若有）与 `jorek×t_norm` 不一致；
10. raw run 交叉校验（可选）：`projections{n}.t_now/t_norm == T_n`、
    `part{n}.time/t_norm == T_{n−1}`；restart 登记帧允许恰为 `−dt_source`
    的偏移并按 `policy=drop|error` 处理，未登记偏移直接报错。

### 2.4 warning + 登记（不 hard fail）

* 缺 `/mesh/boundary`（边界诊断由用户侧调查）；
* 缺 `/condition`（FiLM 不可用；`condition_required=True` 时才报错）；
* 缺 `*_seconds` 时间戳；缺 `gap_indices` 里登记的空缺；
* restart 注册表陈旧（帧实际无偏移）。

### 2.5 布局支持

* `layout="node_dof0"`：`values [9,nodes]` 或 `[9,1,nodes]`（源 `[5,*]`）；
* `layout="full_dof"`：`[9,4,1,nodes]`（2D，`harmonics=1`）；
* 静态 mesh（manifest `mesh_file`）或每步 `/mesh`；
* 预算系列 `budget_layout ∈ {history, slice, absent}`（manifest 必写；未写时
  验收 CLI 自动探测长度），两种布局都支持。

---

## 3. 验收 CLI（`tools/check_export_v3.py`）

```bash
.torch-cuda-env/bin/python tools/check_export_v3.py \
  --run-dir <run-or-export_v3> [--raw-run-dir <raw-run>] \
  [--indices FIRST LAST] [--policy drop|error] \
  [--allow-multi-step] [--no-budget] [--budget-step N] \
  [--max-linkage-checks 200] [--out report.json] [--json] [--quiet]
```

判定：合同/配对/单调性/linkage 全部通过 → exit 0（即使有 warning）；任一无通过
→ exit 1 + JSON `{"ok": false, "error": ...}`。JSON 段：`manifest`、`contract`
（步数/窗口/dt/单步门禁/restart/gap/issues）、`alignment`（raw 交叉校验）、
`shards`（数量分布/直方图）、`boundary`（码统计/缺失/注册表）、`condition`
（组分布/条目数/FiLM 计数）、`files`（大小）、`linkage`、`budget`。

### 3.1 输出示例（合成 5 步导出，含边界码与 condition）

```
run            : a_prime_demo (/tmp/a_prime_demo/export_v3)
steps          : 5 [10..14] windows=4
dt_step        : 0.5 / 0.5 / 0.5 | dt_source: 0.5 / 0.5 / 0.5
single-step    : require=True tol=1e-06
restart dropped: [] | gaps []
issues         : []
shards         : start 3/3/3 | end 4/4/4
boundary       : available=True codes=[0, 1, 2, 5, 9, 11] missing_steps=0
condition      : available=True n_entries=[4] groups=['excluded_transport', 'pellet_angle', 'pellet_mass', 'pellet_velocity']
files          : n=5 total=4.08 MB median=0.816 MB
linkage        : checked=4
budget         : not available (no --raw-run-dir …)
```

（真实 A′ run 到位后，用 `--raw-run-dir` 即可追加完整预算 summary；当前无导出，
§4 的预算数字由 `src/data/budget.py` 直接在原始 run 上读取验证。）

---

## 4. 预算基础读取 summary（真实 run，登记口径，无 gate）

Run：`scan_1d22_ang12_massloss_neon`，预算源 `jorek008900.h5` 完整历史 +
`abl_history.dat`(8900 行) + `macroscopic_vars.dat`(8908 行/块)。

| 项 | 数值 | 口径与代码行号 |
|---|---:|---|
| ablated atoms | `8.9726e19` | `xtime_spi_ablation` 末−首（`pellet_module.f90:319-321`，`t_norm*tstep*spi_abl*species`）；`abl_history.dat` `total_abl_number` 末值一致（差 0.012%） |
| ablation rate（末步） | `3.635e21 atoms/s` | `xtime_spi_ablation_rate`（`pellet_module.f90:767-768`） |
| 率积分/累计 | ≈`19.8` | 口径差异登记：率序列 dt 加权积分不复现逐索引累计；**不做 gate**，待生成侧确认 |
| electrons injected | `2.572e25` | `xtime_Ne_imp`（电子/JOREK 时间）× `dt×t_norm`（`marker_spi_source.f90:735-739`） |
| ionization energy | `2.424e4`（累计） | `xtime_E_ion`；率积分 `6.595e10` |
| radiated energy | `3.965e6`（累计） | `xtime_radiation`；率积分 `9.742e12` |
| energy closure 末行残差 | `-8.988e15` | `-dEtotdt − (sum bnd fluxes + sources + dissipative)`（`live_data.f90:591`）；`median|res|=9.09e8`；2 行非有限（`n_nonfinite`） |
| particle closure | 末行 0；inside+outside 残差 `5.0e11`（相对 4e-10） | `dparticles_dt`/`particlecontent`（`live_data.f90:650-680`） |
| boundary particle flux | `-1.1188e28` | `bnd_particle_fluxes` 5 分量和（`live_data.f90:845-866`） |
| projection FEM 交叉校验（可选） | `dE_ion=-5.87e-5`、`rad_sink=2.47e-4`、`N_imp=2.97` | P1 集总积分；投影≠内部 `inj_source`（ADR-0016/0018） |
| 杂质粒子预算 | 未闭合 | 需 `rho_imp` FEM 体积积分 + 边界通量 → **Phase B1** |

布局结论：原始 `jorek{n}.h5` 的 `xtime*` 是**完整历史**（长度=`index_now`，
7350→7360 帧共享前缀逐位相等），不是当前步切片；A′ manifest 用
`budget_layout="history"` 登记，切片布局同样支持。

---

## 5. 测试与结果

```bash
# 新增测试
.torch-cuda-env/bin/python -m pytest tests/test_step_dataset.py tests/test_budget.py tests/test_check_export_v3.py -q
# 37 passed, 1 skipped (real A' export not delivered yet)

# 全量（不回退基线 186）
.torch-cuda-env/bin/python -m pytest -q
# 223 passed, 1 skipped, 2 warnings (91.8 s)
```

覆盖：`require_single_step` 正/反例、dof0/full-dof、历史/切片预算、shard 缺失与
形状不符、boundary 未登记码、condition 非法组/`excluded_*` FiLM、区间链接/缺口、
restart 登记 drop/报错、raw part/projection 交叉校验、linkage、CLI 退出码与
JSON 输出。

---

## 6. DoD 对照

| DoD | 状态 | 证据 |
|---|---|---|
| smoke 字段审计 + 生成侧缺字段清单 | ✅（无 smoke 交付，已做来源文件审计） | `phase-a-prime-data-audit-2026-10-04.md` §2–§9（P0 清单：shard 双端/区间元数据/manifest） |
| export_v3 loader + require_single_step + 验收 CLI 合入 | ✅ | `src/data/step_dataset.py`、`tools/check_export_v3.py`、测试 |
| 预算基础读取可用（原子/能量/辐射 + 全局闭合残差） | ✅ | `src/data/budget.py`、§4 数字、`tests/test_budget.py` |
| 口径与代码行号登记 | ✅ | 审计报告 §7–§8、本报告 §4（JOREK 行号） |
| 报告与进度条目 | ✅ | 本报告、审核报告、`implementation-progress.md` |
| 未动 legacy 测试/roadmap/ADR | ✅ | 改动文件清单 §1 |

---

## 7. 开放问题（按风险排序）

1. **A′ 数据未交付（高）**：loader/CLI 就绪但无真实钥匙可验；首次 smoke 落盘后
   必须重跑审计并把状态从“来源文件核对”切换为“导出键核对”。
2. **`shard_start` 双端导出（高）**：生成侧缓存上一步 `spi_*_arr` 是第一优先补项；
   注入前 55 帧（0–2700）无 `spi_*_arr`，2710 帧需显式空数组或从 2720 起导出。
3. **烧蚀率口径（中）**：率积分/累计 ≈20 的差异需生成侧确认（不设 gate）。
4. **`freeboundary`（全局→逐节点）与 `axis_node`（flag→索引表）转换（中）**。
5. **restart 帧无隐式偏移（中）**：新导出写显式区间；否则登记并 drop。
6. **边界诊断缺位（中，用户侧）**：`grid_to_wall`/法向/`c_s`/`T_e,n_e` 缺失
   仅 warning + 登记。
7. **`gh` CLI 缺失（流程）**：issue 认领/关闭未执行，待环境恢复补。
