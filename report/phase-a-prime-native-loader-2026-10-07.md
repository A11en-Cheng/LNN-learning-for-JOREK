# Phase A′ 原生逐步布局 loader：数据接口 v2（2026-10-07）

**状态**：current。**依据**：ADR-0018/0019、`roadmap_v2.md` Phase A（§4/§5/§6 A0–A2）、
`report/data-contract-v3.md`（冻结）、`report/data-export-spec-v3.md`（冻结）。
ADR-0001–0017 为 v1 历史，本文未作为依据、gate 或对照。

**范围**：数据接口（原生逐步布局 loader、restart/零步长处理、验收 CLI 原生模式、
原生预算/condition 读取）。**非目标**：训练、闭环、边界拓扑/法向/bn（用户侧调查）、
GNN/谱分支、性能 gate、legacy 测试/roadmap/ADR 改动。`gh` 在本环境缺失，
issue 流程未执行（已登记）。

> **勘误/更新（2026-10-07，ADR-0020 决定 1/5/7）**：
>
> 1. **默认策略变更**：本报告写作时默认 `--zero-dt-policy error`（要求去重重导）。
>    用户随后决定**不重导**；loader 默认改为 `zero_dt_policy="registered"`：
>    登记事件（`data/restart_events.json`）通过，跳过跨界配对与滞后投影，
>    事件帧保留并登记退化源窗口；**未登记** dt<=0 仍 fail-fast。
>    v400 run 的事件为 9 个（2702/4111/5871/6871/8371/9421/10421/11421/11711），
>    本报告的 injInner 6 事件为另一个 run。
> 2. **shard 时间标注修正**：`jorek{k}` 的 `spi_*_arr` 真实时间是 `T_k`
>    （§2 证据 `spi_abl_arr` == `xtime_spi_ablation_rate[k]`）；loader 的
>    `shard_start/end` 标注改为 `T_k`/`T_{k+1}`，源窗口起点 `p_{k-1}` 取上一条
>    记录的 `shard_start`（原报告 §8 #1 的"提前一帧"问题已修）。
> 3. **源对象**：原生 `S_used = projections{k}` 的 `source_object` 由 `projection`
>    改为 `marker_feedback`（marker run 里它就是能量/辐射/电荷真实场源，
>    ADR-0020 决定 1）。
> 4. **预算**：`xtime_spi_ablation` 按 restart 分段（ADR-0020 决定 7）；
>    `native_step_budget` 在事件帧用 rate 序列给出段内 per-step 值。

**对象 run（只读）**：`data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner`
（= `/mnt/d/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner`）。

---

## 1. 改动文件清单

| 文件 | 类型 | 内容 |
|---|---|---|
| `src/data/native_run.py` | 新增 | 原生逐步布局 reader：`jorek*/projections*/part*` → `(u_k, S_used, u_{k+1})`；t_now 推导区间 + provenance；静态 mesh sha256 缓存；restart 注册表解析与报告；零步长策略；条件通道 |
| `src/data/step_dataset.py` | 扩展 | `detect_run_layout()` 自动检测两种布局；`StepRunDataset` 分派到 export_v3 / native 后端（同一公开面）；`StepMeta`/`StepContractReport` 增补 native/restart/zero-dt 登记字段 |
| `src/data/budget.py` | 扩展 | `read_series_tail()`（只读 `dataset[-1]`，避免 O(N²) 历史 I/O）、`native_step_budget()` 逐 step 差值、`summarize_budget()` native 登记段 |
| `tools/check_export_v3.py` | 扩展 | `--run-dir` 直接吃原生 run（自动检测）；`--zero-dt-policy {error,list}`、`--restart-file`/`--restart-indices`、`--gap-indices`、`--namelist`、`--mesh-verify-stride`；输出新增 `layout_kind`/`restart`/`zero_dt`/`provenance` 段 |
| `tests/conftest.py` | 扩展 | `write_native_frame`/`write_native_projection`/`write_native_run` 合成原生 run fixture（可注入零步长帧、restart 偏移、part 偏移、缺口） |
| `tests/test_step_dataset.py` | 扩展 | 原生布局 16 例：配对/区间、condition、零步长 error/list、去重缺口、非零 dt 缺口与多步登记、未登记偏移、登记 restart drop、part 失配、边界帧登记、参数面分组覆盖、真实数据（skip-if-absent） |
| `tests/test_budget.py` | 扩展 | tail-only 读取、逐 step 预算、abl_history 登记、native summarize |
| `tests/test_check_export_v3.py` | 扩展 | CLI 原生模式端到端、零步长 error/list 退出码、原生预算段 |
| `tests/test_data_manifest.py` | 微调 | v1 8-run 网格断言排除新出现的 A′ 原生 run（`phaseA_dataset_*`）；非本任务数据契约改动 |
| `report/phase-a-prime-native-loader-2026-10-07.md` | 新增 | 本报告 |
| `report/README.md`、`report/implementation-progress.md` | 更新 | 索引 + 进展条目 |

未改动：`tests/legacy/**`、`roadmap_v2.md`、`docs/adr/**`、`src/data/output_contract.py`
（其 `ContractError`/`TIME_TOL_JOREK`/`assert_strictly_increasing`/`resolve_step_times` 被复用）、
旧 manifest/checkpoint/评测 JSON。

---

## 2. 布局支持（任务 A）

`StepRunDataset(root, ...)` 现在自动检测：

| 布局 | 判据 | 记录构造 |
|---|---|---|
| (a) `export_v3` | `<root>/export_v3/manifest.json` + `steps/step_*.h5`（或 root 即 export 目录） | 每步自描述文件（原有语义，未变） |
| (b) 原生逐步 run | run 目录内 `jorek{n}.h5`（无 `manifest.json`） | `u_before=jorek{k}`、`S_used=projections{k}`、`u_after=jorek{k+1}`、`shard_start/end=spi_*_arr(jorek{k}/jorek{k+1})`、mesh/boundary 取 `jorek{k}` |

原生布局的关键实现点：

* **区间元数据由 t_now 推导**：`interval_start=T_k`（`jorek{k}.t_now`）、
  `interval_end=T_{k+1}`、`source_interval=[T_{k-1}, T_k]`；`macroscopic_vars.dat`
  的 `@times` 作为时间表，并对每个使用到的 `jorek` 帧做 `t_now == T_n` 交叉校验
  （容差 `TIME_TOL_JOREK=1e-4`）。provenance 写入 `manifest["provenance"]` 与
  `report.interval_source`。
* **静态 mesh 按 hash 缓存**：`x/size/vertex/contain_node/neighbours/boundary`
  的 sha256 指纹，首帧 + 末帧 + 每 `mesh_verify_stride`（默认 128）帧抽验；
  本 run 实测 66 次校验全部一致（`report.mesh_verifications=66`），
  边界码在同一批帧上校验（注册表 + `--extra`）。
* **边界帧登记不静默**：`projections{2700}` 为前缘（无 `jorek{2700}`）、
  `jorek{11104}` 为末帧（无 successor），均登记 issue 并进入
  `report.boundary_frames_{leading,trailing}`；配对范围 2701..11103。
* **`require_single_step=True`（Phase B/C 硬门禁）**：native 判据登记为
  `single_step_method="native_adjacent_frame"`——(i) 记录后继是下一记录帧
  （`following == index+1`，或缺口已登记且被删帧都是零步长重复帧）；
  (ii) `dt_step > 0`、`dt_source > 0`；(iii) 场区间与下一记录的源窗口是同一物理区间
  `dt_step(k) == T_{k+1} - T_k == dt_source(k+1)`，容差 `1e-6` JOREK 单位（可覆盖）。
  **登记说明**：任务书写的是字面 `dt_step == dt_source`（export_v3 布局仍按此判定，未改）；
  本 run 的 tstep 在 33 处变化（0.05–5.0），字面相等会在 22 个**合法**单步对上误报，
  故 native 用"相邻帧 + 窗口链接"判据，方法与容差均写入报告，供复核。
* 未登记索引缺口直接 fail-fast（"register them explicitly -- do not skip silently"）；
  登记缺口只允许覆盖零步长重复帧（去重场景）。

## 3. restart 与零步长（任务 B）

### 3.1 冻结事实（本次全量复核，与任务输入一致）

| 项 | 实测 |
|---|---|
| `jorek{2701..11104}` | 8404 帧（编号连续） |
| `projections{2700..11104}` | 8405 帧 |
| `part{2710..11100}` | 840 帧（每 10 步）+ `part_restart.h5` |
| 零步长帧（`T_n == T_{n-1}`，容差 1e-8） | `2704, 5681, 6441, 7221, 8611, 9291` |
| 受影响配对（`dt_step = 0`） | `2703, 5680, 6440, 7220, 8610, 9290` |
| 有效单步记录（`dt_step > 0`） | **8397**（8403 − 6） |
| `dt_step` | min 0.05 / median 0.5 / max 5.0 |
| `part{n}.time/t_norm == T_{n-1}` | 840/840 通过（容差 1e-4） |
| restart 事件上的 projections | `projections{n-1}` 偏移恰为 `-dt_source`（2703:-5.0、5680/6440/7220/8610:-1.0、9290:-0.2）；`projections{n}` 与 `jorek{n}` 对齐 |

### 3.2 登记集与 restart 报告

* `available_restart_files.txt`（715 条，0..7140 每 10 步）被解析为 **checkpoint 注册表**；
  6 个实际 restart 事件（由时间表重复检测）与注册表交叉登记：事件 2704 的
  `nearest_checkpoint_le_event=2700` 等。注册表内落入 run 窗口的 444 条逐条检查
  jorek/projection/part 时间与配对，输出在 CLI 的 `restart.checks`。
  **注意**：该文件是"可用 checkpoint 清单"，不是 restart 事件清单——本 run 的 6 个
  事件帧都不在清单内（编号非 10 的倍数），报告按事件 + 注册表双轨登记。
* `--restart-indices 2704,5681` 或 `--restart-file` 可显式覆盖注册表。

### 3.3 零步长策略（`--zero-dt-policy {error,list}`）

* `error`（默认）：fail-fast，输出精确索引、受影响配对与生成侧修复指引；
* `list`（仅旧数据诊断，禁止用于 Phase B/C）：列出 dt=0 帧与受影响配对，
  丢弃零长记录，登记 `zero_dt_record_dropped`；事件后一条记录的零长
  **source 窗口**（`[T_{n-1}, T_n]`，两端同刻）以 `zero_dt_source_window` 登记
  （不静默丢弃，故 8397 计数可复现）。

### 3.4 重启去重规范（交付生成侧）

1. 每个 code restart（判据 `T_n == T_{n-1}`）只保留**一个**帧：推荐保留前一个
   已写帧 `n-1`（其 `S_used=projections{n-1}` 窗口正常），**不写**重复帧 `n`，
   随后从 `n+1` 继续；
2. 去重后 `t_now` 严格递增：`T_{n-1} < T_{n+1}`，配对 `(n-1 → n+1)` 的时间差等于
   真实单步 `dt(n+1)`（本 run 为 5.0/0.2/0.1 等），`dt > 0` 全部成立；
3. `projections{k}` 与同号 `jorek{k}` 对齐：`projections{k}.t_now/t_norm == T_k`，
   **不得**保留提前一步写出的 offset 帧（当前事件的 `n-1` 帧就是这种偏移帧）；
4. `part` 保持每 10 步不变：`part{m}.time/t_norm == T_{m-1}`（restart 不影响）；
5. 索引连续，或有缺口就登记：直接删帧造成编号缺口时，在 run 目录写
   `phase_a_native_run.json`（`{"gap_indices": [2704, ...]}`）或用 CLI
   `--gap-indices`；loader 会核验缺口帧为零 dt 重复帧，否则 fail-fast；
   若重新编号成连续索引，则时间表须同步重写；
6. loader 硬断言（去重后必须全过）：`t_now` 严格递增、无 `dt<=0`、
   `projections` 同号对齐、`part{n}.time = T_{n-1}`；违反 → fail-fast +
   精确索引 + "生成侧需在 restart 处去重"提示，不做静默丢弃。

## 4. 预算与 condition（任务 C）

* **预算**：`read_series_tail()` 只用 `dataset[-1]` 取每个 `xtime*` 序列的最后一项
  （原生 `jorek{n}.h5` 存的是完整历史，长度=`index_now`，逐 step 读全量是 O(N²) I/O）；
  `native_step_budget(run_dir, n)` 用 `jorek{n}` 与 `jorek{n-1}` 的尾部差值给出
  逐 step 烧蚀原子、电子注入、电离/辐射能量，并登记 `abl_history.dat` 对应行；
  `summarize_budget()` 优先用 `abl_history.dat` 总量 + `macroscopic_vars.dat`
  全局闭合（能量/粒子/边界通量），新增 `native_layout` 登记段。
  杂质预算仍标记 Phase B1。
* **condition**：从生产 `jorek_namelist_spitest` 读值（缺省回退 `jorek_namelist`），
  经 `parameter_channel.build_condition_vector` 的取值链（H5 → namelist → JOREK 默认）
  生成 61 维向量，落组为冻结词表（`pellet_mass` 2、`background_plasma` 5、
  `boundary_flags` 8、`excluded_transport` 42、`excluded_numeric` 4；
  `film_eligible=1` 仅 15 条物理项）；`eta/visco/visco_par` 做 H5 ↔ namelist 交叉校验
  （H5 胜出并登记）；`ZKpar_T_dependent / ZK_par_max / Ti_min_ZKpar / Te_min_ZKpar`
  作为 ZK 温度依赖与上限的登记项写入 `condition_summary.to_dict()`。
  重复名 `bc_natural_open` 去重为 `bc_natural_open#2` 并登记。

## 5. 真实数据证据（字段/时间）

* `jorek002701.h5`：`values (9,4,1,4125)`、`x (2,4,1,4125)`、`size (4,4,3986)`、
  `vertex (4,3986)`、`contain_node (5,3986)`、`neighbours (4,3986)`、
  `boundary (4125,)`、`freeboundary (1,) |S1 b'F'`、`t_now=8372.0`、
  `t_norm=4.085775226e-7`、`tstep=5.0`、`index_now=2701`、`n_spi_tot=846`、
  16 个 `spi_*_arr`、12 个 `xtime_*`。
* `projections{n}`：`values (5,4,1,4125)`、`t_now`（物理秒）。
* 时间对齐（抽样）：`jorek{t}` 全部等于 `macroscopic_vars @times`；
  `projections2703/t_norm=8377.0=T_2702`（偏移 −5.0）；`projections2704/t_norm=8382.0=T_2704`；
  `part2710/time/t_norm=8407.0=T_2709`、`part11100 → 12367.8=T_11099`（偏差 <6e-11）。
* 全量扫描：8404 帧的 `t_now` 与时间表一致；除 6 个事件外
  `projections{k}.t_now/t_norm == T_k`；`part` 840/840 满足 `T_{n-1}`。

## 6. 验收 CLI（用法与输出）

```bash
# 原生 run（默认 error：发现零步长即 fail-fast，exit 1）
.torch-cuda-env/bin/python tools/check_export_v3.py \
  --run-dir data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner

# 旧数据诊断（list）：枚举 dt=0 帧/配对，其余全段照常输出
.torch-cuda-env/bin/python tools/check_export_v3.py \
  --run-dir data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner \
  --zero-dt-policy list --out .scratch/phaseA-prime-native/native_check_list.json

# 去重后的新数据（应直接通过；缺口登记）
.torch-cuda-env/bin/python tools/check_export_v3.py \
  --run-dir <regenerated-run> --gap-indices 2704,5681,6441,7221,8611,9291
```

`--run-dir` 同时接受 export_v3 目录（原有行为）。exit 0/1 语义与现有一致：
合同/配对/单调性/linkage 全部通过 → 0（warning 不影响）；任一 fail-fast → 1 +
`{"ok": false, "error": ...}`。JSON 段：`manifest/provenance/contract/alignment/
restart/shards/boundary/condition/files/linkage/budget`。

**默认模式（当前未去重数据，exit 1）**：

```
FAIL: .../phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner: 6 zero-dt frame(s)
from code restarts: [2704, 5681, 6441, 7221, 8611, 9291]. index 2704: T_2704=8382.0 ~
T_2703=8382.0 (dt=0.0); ... Affected pairs (dt_step=0): [2703, 5680, 6440, 7220, 8610,
9290]. Generation-side fix (plan 1: dedup at every code restart): keep exactly one frame
with strictly increasing t_now, re-align projections{k} with jorek{k}, keep the part
cadence, and register the removed duplicate indices as gaps. For legacy-data diagnostics
only, pass --zero-dt-policy list (never for Phase B/C).
```

**list 模式（旧数据诊断，exit 0；约 109 s）**：

```
run            : phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner [layout=native]
steps          : 8397 [2701..11103] windows=8397
dt_step        : 0.05 / 0.5 / 5 | dt_source: 0 / 0.5 / 5
single-step    : require=True tol=1e-06 method=native_adjacent_frame
zero-dt        : policy=list indices=[2704, 5681, 6441, 7221, 8611, 9291]
                 pairs=[2703, 5680, 6440, 7220, 8610, 9290]
restart        : registry=715 events=[2704, 5681, 6441, 7221, 8611, 9291] checked=444
restart dropped: [] | gaps []
issues         : ['boundary_frame_leading', 'boundary_frame_trailing',
                  'freeboundary_global_flag', 'part_time_checked', 'restart_registry',
                  'restart_registry_aligned', 'zero_dt_frames_listed',
                  'zero_dt_record_dropped', 'zero_dt_source_window']
shards         : start 846/846/846 | end 846/846/846
boundary       : available=True codes=[0, 1, 2, 5, 9, 11, 15, 19, 20, 21] missing_steps=0
condition      : available=True n_entries=[61] groups=['background_plasma',
                  'boundary_flags', 'excluded_numeric', 'excluded_transport', 'pellet_mass']
files          : n=16801 total=96184.69 MB median=6.952 MB
linkage        : checked=200
budget         : atoms=1.22461e+20 | layout=history | energy residual=8.44806e+13
```

完整 JSON：`.scratch/phaseA-prime-native/native_check_list.json`（摘要 `.txt`、
默认模式 JSON `.scratch/phaseA-prime-native/native_check_error.json`）。

**去重形态演示（合成 fixture：删除重复帧 2704 + 登记缺口，默认严格模式 exit 0）**：

```
run            : dedup_run [layout=native]
steps          : 5 [2701..2706] windows=5
dt_step        : 0.5 / 0.5 / 0.5 | dt_source: 0.5 / 0.5 / 0.5
single-step    : require=True tol=1e-06 method=native_adjacent_frame
zero-dt        : policy=error indices=[] pairs=[]
restart        : registry=0 events=[] checked=0
restart dropped: [] | gaps [2704]
issues         : ['boundary_frame_leading', 'boundary_frame_trailing',
                  'freeboundary_global_flag', 'part_time_checked']
```

（命令行：`--run-dir /tmp/.../dedup_run --gap-indices 2704`；产物
`.scratch/phaseA-prime-native/dedup_demo.json` / `.txt`。真实 run 重生成后应呈现同一形态。）

## 7. 测试与结果

```bash
.torch-cuda-env/bin/python -m pytest tests/test_step_dataset.py tests/test_budget.py tests/test_check_export_v3.py -q
# 60 passed, 1 skipped（新用例 23 个 + 原有 38 个）

.torch-cuda-env/bin/python -m pytest -q
# 246 passed, 1 skipped, 2 warnings（基线 223 passed / 1 skipped；+23，无回退）
```

覆盖：原生配对/区间推导、condition 命名与分组、零步长 error/list、去重缺口
（登记通过 / 未登记 fail-fast / 非零 dt 缺口拒绝）、未登记 projections 偏移、
登记 restart drop 与逐条检查报告、part 时间失配、边界帧登记、mesh 缓存校验、
CLI 原生模式与退出码、预算 tail-only 读取与逐 step 差值、真实数据
skip-if-absent（时间表 6 事件 + 2701..2706 窗口样本）。

## 8. DoD 对照

| DoD | 状态 | 证据 |
|---|---|---|
| 原生逐步 run 可直接跑验收 CLI | ✅ | §6；`--run-dir` 自动检测；exit 0/1 |
| 8397 个有效配对可复现 | ✅ | list 模式 `n_steps=8397`、`n_windows=8397`（§3.1） |
| restart 登记报告可复现 | ✅ | 6 事件 + 715 checkpoint 注册表 + 444 逐条检查（`restart.events`/`restart.checks`） |
| 去重后严格通过 / 未重生成时演示 fail-fast + list | ✅ | 默认 error 退出 1 并给出精确索引与修复指引；list 输出并登记；合成 fixture 覆盖去重缺口正例 |
| 预算/condition 读取可用 | ✅ | `native_step_budget`/`summarize_budget` native 段；61 维 condition + 交叉校验 + ZK 登记 |
| 测试全绿无回退 | ✅ | 246 passed / 1 skipped |
| 报告与进度条目 | ✅ | 本报告 + `implementation-progress.md` |

## 9. 开放问题（按风险排序）

1. **（高）去重后的 run 尚未生成**：当前 run 只能以 `--zero-dt-policy list` 诊断；
   重生成后应用默认模式重跑验收，`report.zero_dt.indices` 应为空、
   `zero_dt_source_windows` 应为空，且缺口（若有）必须登记。
2. **（高）restart 事件清单来源**：`available_restart_files.txt` 是 checkpoint 可用清单
   （0..7140/步长 10），不含 6 个事件帧；建议生成侧在 `phase_a_native_run.json`
   或 manifest 中显式登记 restart 事件索引（loader 已支持 `restart_indices`）。
3. **（中）native 的 `S_used` 语义**：原生布局只能提供 `projections{k}`（marker 导出通道，
   `source_object="projection"`）；ADR-0016/0018 要求推理输入是内部 `inj_source`，
   原生 run 因此只能作监督/交叉校验数据。Phase B1 需要内部源字段（export_v3 或
   H5 新增块），否则须显式降级。
4. **（中）零长 source 窗口**：事件帧 `n`（如 2704）的源窗口 `[T_{n-1}, T_n]` 两端同刻，
   list 模式保留并登记（8397 计数因此可复现）；去重后消失。
5. **（中）`freeboundary`/`axis_node` 形状**：原生 H5 是全局 `'F'` 标志与逐节点 flag，
   A′ 规范要逐节点 int8 与索引表；loader 仅登记 warning，转换留给生成侧。
6. **（中）单步判据登记差异**：native 用"相邻帧 + 窗口链接"（`1e-6`）而非字面
   `dt_step == dt_source`；已在 §2/报告字段登记，需用户/评审确认可接受。
7. **（低）预算口径**：烧蚀率积分 vs 累计比值 ≈20（前一轮已登记），本任务未改口径。
8. **（低，流程）**：`gh` CLI 缺失，issue 认领/关闭未执行。
