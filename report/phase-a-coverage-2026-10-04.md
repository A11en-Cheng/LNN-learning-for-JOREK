# Phase A 覆盖报告：契约与细粒度数据（2026-10-04）

**范围**：`roadmap_v2.md` Phase A（A0–A3）。依据只有 ADR-0018/0019 + roadmap_v2 +
schema §3.1a；ADR-0001–0017 为 v1 历史，未作为 gate/前置/对照。

**结论**：A0–A2 全部落地，A3 收尾完成；全量 pytest 全绿。细粒度（每流体步）导出
规范已可独立交付；旧 EHL-2 数据已按 ADR-0019 重建配对并通过真实数据断言。

> **2026-10-04 后续修订（A′）**：用户确认导出档位改为 **A′**——每流体步必需
> `u_before/after + S_used + shard_start/end + 元数据`；完整 marker 降为 1 个 run 的
> 低分辨率诊断流（M1/M2/M3）。本文 §2/§4 中出现的 `p_start/p_end`（marker）语义
> 以 `report/data-export-spec-v3.md` §2.3/§2.3b 的 `shard_start/shard_end` 为准；
> 开放问题 #1 已按 A′ 关闭。

---

## 1. 产出清单（DoD 对照）

| DoD | 产出 | 证据路径 | 状态 |
|---|---|---|---|
| 1a 契约 v3 冻结 | 数据契约 v3（S 对象 + 时间语义 + 粒度 + restart 规则） | `report/data-contract-v3.md` | ✅ |
| 1b 入索引 | report 索引新增 current 条目 | `report/README.md` | ✅ |
| 1c 导出规范可交付 | 字段级导出规范 + 方案 A/B（体积冲突） | `report/data-export-spec-v3.md` | ✅ |
| 2a 时序断言 | 输出契约模块（严格递增/proj 对齐/part=场帧−dt/restart 登记剔除） | `src/data/output_contract.py` | ✅ |
| 2b 配对加载器 | `(u_k, part{k}, projections{k}) → u_{k+1}` fail-fast 加载器 | `src/data/paired_dataset.py` | ✅ |
| 2c 测试 | 合成 fixture + 真实数据 skip-if-absent 断言 | `tests/test_output_contract.py`、`tests/test_paired_dataset.py` | ✅ |
| 3 重建演示 | 1 个 run 的 ADR-0019 配对索引 + restart 处理记录 | `tools/rebuild_ehl2_pairs_v3.py`、`.scratch/n/pairs_v3_demo/` | ✅ |
| 4 进度与覆盖 | 本报告 + implementation-progress 新条目 | `report/implementation-progress.md` | ✅ |
| 5 开放问题 | 按风险排序的清单（§5） | 本报告 §5 | ✅ |

---

## 2. 契约 v3 关键内容（已冻结）

1. **S 对象**：被复刻对象 = model502 内部 `inj_source`（Gauss 点沉积、进 RHS、
   对场显式）；`projections*.h5` 是 marker 投影，只作监督/交叉校验，不得作推理输入；
   每份导出必须登记 `source_object` provenance（`inj_source` | `projection`）。
2. **时间语义**：`jorek{n}` = `u(T_n)`（JOREK 单位）；`projections{n}` = 窗口
   `[T_{n−1}, T_n]`（一个流体步）的源，内部时间 `T_n`（物理秒，`÷t_norm` 归一）；
   `part{n}` = `T_{n−1}` 的 marker 状态（物理秒）。唯一配对
   `(u_k, p_{k−1}=part{k}, S_k=projections{k}) → u_{k+1}`。
3. **粒度**：目标为每流体步七元组 `(interval_start, interval_end, u_before, S_used,
   u_after, p_start, p_end)`；按 ADR-0019 错步语义，源采样窗口 `[T_{n−1}, T_n]` 与场
   推进区间 `[T_n, T_{n+1}]` 的端点分别写时间戳（导出规范 §1/§2）。旧 EHL-2 为帧级
   dump（10/50 流体步间隔），只做区间两端监督；禁止帧间插值伪真值。
4. **restart 边界**：3550/4150/4950/5330/6600/7350（run1）与 5050（v400）登记；
   默认剔除，未登记偏移 fail-fast。
5. **review 字段**：逐节点 `boundary` + `freeboundary`（与场帧同源同索引）；
   condition 分组标签与组语义（浅层 FiLM 用，`excluded_*` 禁止入 FiLM）。

### 2.1 2026-10-04 全量核实（契约冻结依据）

run `scan_1d22_ang12_massloss_neon`，620 个配对帧，逐帧读取
`jorek*/projections*/part*` 的 `t_now`/`time` 并与 `macroscopic_vars.dat` 的
`@times` 对照（容差 `1e-4` JOREK 单位）：

| 关系 | 结果 |
|---|---|
| `part{k}.time ÷ t_norm == T_{k−1}` | 620/620 ✅ |
| `projections{k}.t_now ÷ t_norm == T_k` | 614/620；其余 6 帧恰为登记 restart 边界 |
| restart 帧 `projections` 偏移 | 恰为 `−dt_source(k)`（一个流体步） |
| v400 编号 5050 抽样 | `proj`/`part` 同为 `T−1.0`，与登记一致 ✅ |

> 细化说明：ADR-0019 核实结果第 4 条的措辞"同号的 `part`+`projections` 比 `jorek`
> 早一个步长"在数据上表现为：`part` 始终满足常规规则 `T_{k−1}`（620/620 无例外），
> **仅 `projections` 在 restart 帧提前一个流体步**。这不改变剔除/重对齐政策。
> **已于 2026-10-04 修正 ADR-0019 措辞为 projection-only 偏移。**

---

## 3. 代码与测试证据

### 3.1 `src/data/output_contract.py`

* `collect_run_timing` / `read_frame_timing`：读取三方文件时间并归一化到 JOREK 单位；
* `parse_macroscopic_times` / `resolve_step_times`：逐步时间表（显式表 > `macroscopic_vars.dat`）；
* `check_run_contract`：严格递增、`dt<=0`、投影/粒子时间断言、restart 登记剔除
  （`policy="drop"|"error"`）、覆盖边缘单边帧登记；返回 `ContractReport`；
* `RESTART_BOUNDARY_INDICES`：登记 6+1 个 restart 边界。

### 3.2 `src/data/paired_dataset.py`

* `PairedRunDataset`：按契约构造配对索引；`PairSample` 暴露
  `u_before/source/u_after/p_start/dt_step/dt_source/t_before/t_after/t_norm`；
* `ParticleSnapshot`：`part*.h5` 的懒加载视图（`groups/<gid>/<dataset>`）；
* fail-fast：非单调、`dt<=0`、配对缺失/半配对、节点数不一致、文件在索引后消失。

### 3.3 测试（合成 fixture 必测 + 真实数据 skip-if-absent）

| 文件 | 覆盖 |
|---|---|
| `tests/test_output_contract.py` | 合成契约通过；未登记偏移 fatal；登记边界 drop/error；登记边界偏移量必须恰为 `−dt_source`；`part` 必须等 `T_k−dt_source`（非 dump 间隔）；半配对 fatal；无标记帧允许；缺逐步表时降级为显式 issue；非单调 fatal；`macroscopic_vars` 解析；真实 run 窗口 + 真实 restart 边界（skip-if-absent） |
| `tests/test_paired_dataset.py` | 配对语义/形状/时间；fail-fast（非单调、缺投影）；restart 剔除；index window；真实数据 pair（7360/7370，4125 节点，`dt_step=0.5`、`dt_source=0.05`）；登记表断言 |
| `tests/test_parser_contract.py` | 删除 v1 `JorekSequenceDataset` 对齐用例（由配对加载器测试取代），保留 parser 契约 |

命令与结果：

```bash
.torch-cuda-env/bin/python -m pytest tests/test_output_contract.py tests/test_paired_dataset.py -q
# 23 passed
.torch-cuda-env/bin/python -m pytest -q
# 128 passed, 2 warnings（基线 106 − 1 个 v1 用例 + 23 个新用例）
```

---

## 4. 旧 EHL-2 数据覆盖报告（重建演示）

演示命令：

```bash
.torch-cuda-env/bin/python tools/rebuild_ehl2_pairs_v3.py
```

产物目录：`.scratch/n/pairs_v3_demo/`（新目录，未覆盖任何旧产物）

| 产物 | 内容 |
|---|---|
| `pairs_index.jsonl` | 613 条合法配对（`index`、`index_next`、`t_state/t_next`、`dt_step`、`dt_source`、三方文件名） |
| `restart_boundaries.json` | 登记表 + 6 个实测偏移（含 `deviation_jorek` 与 `-dt_source` 对照）+ 剔除结果 + 规则说明 |
| `coverage.json` | 契约报告（帧数、配对、issues、逐步表来源）、run 时间跨度、dt 统计、节点数、`t_norm` |

覆盖统计（run `scan_1d22_ang12_massloss_neon`）：

| 项 | 值 |
|---|---|
| `jorek` 场帧 | 675（索引 0–8900；早段 50 步 dump，2700 起 10 步 dump） |
| marker 配对帧（part+projections 齐备） | 620（2710–8900） |
| 合法配对 | 613（右端点 2710–8890；末帧 8900 无下一场帧） |
| 覆盖边缘单边帧 | 2700（只有 projections，无 part；登记 `unpaired_boundary_frame`，跳过） |
| restart 边界剔除 | 3550、4150、4950、5330、6600、7350（6 个） |
| `dt_source`（源窗口） | min 0.05 / median 0.2 / max 5.0（JOREK 单位） |
| `dt_step`（场推进区间） | min 0.5 / median 2.0 / max 50.0（= dump 间隔） |
| 节点数 / `t_norm` | 4125 / 4.085775226e-7 s per JOREK unit |

单位与来源：场时间 JOREK 单位（`jorek.t_now`）；marker 时间物理秒（`projections.t_now`、
`part.time`）；空间几何米；逐步时间表来自 run 内 `macroscopic_vars.dat`（`@times`）。

### 4.1 restart 边界记录（实测偏移）

| 索引 | `T_k` | `projections` 时间 | 偏移（JOREK 单位） | `−dt_source` |
|---:|---:|---:|---:|---:|
| 3550 | 8660.000 | 8659.000 | −1.000 | −1.0 |
| 4150 | 8809.500 | 8809.000 | −0.500 | −0.5 |
| 4950 | 9159.000 | 9158.000 | −1.000 | −1.0 |
| 5330 | 9204.800 | 9204.600 | −0.200 | −0.2 |
| 6600 | 10324.800 | 10319.800 | −5.000 | −5.0 |
| 7350 | 10474.300 | 10473.800 | −0.500 | −0.5 |

处理：默认剔除（`policy="drop"`）；`policy="error"` 下同一批帧直接报错，供"必须
重新导出"的验收使用。

---

## 5. 开放问题（按风险排序）

1. **粒子导出体积（已决定：A′）**：实测 `part*.h5` 172 MB/帧（1e6 slots，18.2 万存活），
   每步全量不可行。用户确认 A′：主路径每步只导出**场 + 源 + shard + 元数据**
   （dof-0 约 7–9 GB/run）；完整 marker 仅对 1 个 run 做低分辨率诊断流
   （M1 每帧 / M2 每 5–10 步 / M3 欧拉矩）。导出规范已按此更新。
2. **restart 帧的生成侧协议（中）**：新数据必须自带 `interval_start/end`，消除隐式
   一步偏移；若生成侧无法保证，需沿用 v3 §4.3 的登记/剔除流程并在 manifest 标注。
3. **帧级旧数据的适用边界（中）**：`S_k` 只覆盖一个流体步而 dump 间隔是 10/50 步；
   旧数据只能做区间两端监督，不能支撑"区间内部单步推进"验收；B/C 阶段的单步验证
   必须等细粒度数据。
4. **ADR-0019 措辞细化（已处理）**：restart 帧只有 `projections` 偏移（恰为
   `−dt_source`），`part` 始终 = `T_k − dt_source`（常规规则）；ADR-0019 已按此收紧。
5. **budget 口径（中，B1 前置）**：导出规范要求生成侧给出与 JOREK 一致的 FEM 核算
   脚本/公式；未落地前预算闭合只能做离线近似。
6. **Issue tracker（流程）**：本环境 `gh` 缺失（`command -v gh` 无输出），无法执行
   `gh issue list --label ready-for-agent` / 认领 / close；已在报告中记录，待环境恢复后补做。

## 6. 下一步建议（一句话）

把导出规范交付 JOREK 侧并先跑 1 个 run 的方案 A；同时以本契约模块为门禁，任何
新数据进入 B 阶段前必须通过 `check_run_contract` 与配对加载器断言。
