# v2 文档隔离与脏数据审计（2026-10-04）

目的：新架构工作（roadmap_v2 Phase A–E：自生成源 + stencil-conv 九场 + 错步闭环）
可能读到的 v1 错误信息入口清单、已落地的隔离措施、以及"哪些可安全复用"。
本文是隔离规则的操作清单；决策依据见 ADR-0018 / ADR-0019。

## 1. 隔离规则（已落地）

- `docs/adr/README.md`：区分**现行**（ADR-0018/0019）与**历史**（ADR-0001–0017），
  历史 ADR 全部加了顶部 banner：引用须标注"（历史）"，冲突以 0018/0019 为准。
- `AGENTS.md` + `docs/agents/domain.md`：规定新架构工作的阅读顺序与引用规则。
- `README.md` / `CONTEXT.md`：顶部加"2026-10-04 架构修订"banner；CONTEXT 的
  `## 训练语义` 与 `## 验证与治理` 标为 v1 历史。
- `report/README.md`：报告表加 `scope`（current / v1 / historical），引用规则明确。
- `config.yaml`：顶部标 RETIRED（v1 九场/teacher-source 管线）。

## 2. 脏数据入口清单

### 2.1 文档

| 入口 | 脏点 | 现状 / 措施 |
|---|---|---|
| `docs/adr/0001–0017` | v1 训练语义（physics loss@插值点、隐式 CfC、LOO、teacher forcing）、旧量化 gate | 全部加历史 banner + 索引；**不得作为新工作依据** |
| `new_version.md` | v1 路线图；"目标帧"配对错误 | 顶部勘误 + "仅历史" |
| `CONTEXT.md` | 仍含 v1 词汇（Eulerian 矩、flux-coordinate attention、source mapping sub-network 等） | 顶部隔离说明 + 两章节标 v1；逐条 [v1] 标记待补 |
| `README.md` | 首段仍是 v1 目标（非均匀插值 / Physics Loss） | 顶部 banner 指明现行路线；正文保留为 v1 背景 |
| `report/pitfalls.md` | #46 等旧条目称 `source[:, k+1]` 为"目标帧"（错误） | 新增 #64 更正；**引用旧条目时以 #64 为准** |
| `report/roadmap-review-2026-09-29.md` | 基于 v1 讨论；其中"能量/辐射通道需 history"仍是有效证据 | scope=v1；引用标"（历史证据）" |
| `report/stage-summary-2026-10-03.md` / `ab-condition-2026-10-03.md` / `roadmap-v2-readiness-2026-10-03.md` | v1 指标与"基线"定位 | 顶部勘误 / scope=v1；指标不作基线 |
| `report/physics-residual-correction.md` | v1 物理损失语境 | scope=v1（待补顶部 banner） |
| `report/spi-analytic-baseline.md` | 解析骨干证据（**B1 仍需要**） | scope=v1 evidence；只作解析骨干正确性证据 |
| `roadmap_v2.md`（§1.2 #10、§3 接口表、§4 Phase A/B1/C、§5.3/§5.7、§6 A1/A2、§7） | 与 JOREK 源码/实测冲突的源对象（projections 只是代理）、B1 结构（无 marker 链）、Phase C 循环（p=shard）、restart 策略（dt<=0 一律报错） | **文件按用户指令不改**；勘误层 = `docs/adr/0020-marker-source-contract.md` + `report/conformance-roadmap-v2-2026-10-07.md`；引用这几处时以 ADR-0020 为准 |
| `report/phase-b1-source-module-2026-10-07.md`（2026-10-07 前的正文） | "proxy（登记代理）"框架与 "proxy vs inj_source" 标题 | 顶部勘误 + §2/§5 改写；记录投影 = marker run 真实场源（ADR-0020 决定 1） |
| `config.yaml` | 退役九场管线默认值 | RETIRED banner |

### 2.2 代码

| 入口 | 脏点 | 现状 / 措施 |
|---|---|---|
| `tests/train_cfc_interpolation.py` | `source[:, end]` = 一步未来源（ADR-0019 泄漏）；teacher-source 训练 | 顶部 LEGACY banner；**禁用**，按 ADR-0019 重写配对；对应 v1 单测已归档 `tests/legacy/` |
| `src/data/interp_dataset.py` | 同号加载 state/source；无 ADR-0019 断言 | LEGACY banner；其 v1 测试 `test_interp_dataset.py` 已归档 `tests/legacy/`；复用前先改配对与断言 |
| `src/models/cfc_interp.py` | teacher-source 控制注入；`advance_with_source_integral` 为 v1 语义 | LEGACY banner；`test_cfc_interp.py` 已归档；组件可参考，语义不可直接复用 |
| `src/models/source_head.py`、`tests/train_source_head.py` | 同号 `(u_n, shard_n) → S_n` 配对错误 | LEGACY banner；`test_source_head.py` 已归档；改为 `(u_{n−1},u_n,p_{n−1}=part{n})` 后重训 |
| `src/training/trainer.py` | v1 GNN trainer | LEGACY banner |
| `src/evaluation/run_benchmark.py` | v1 GNN benchmark | LEGACY banner |
| `src/models/gnn_encoder.py`、`gnn_decoder.py`、`src/data/dataset.py`、`src/models/lnn_processor.py`、`src/evaluation/benchmark.py` | v1 GNN 栈 | **待补 banner**（backlog）；新架构禁用 GNN |
| `src/models/ltc_cell.py` | docstring 的 physics-residual 动机属 v1 | 加 NOTE：组件可复用，语义以 ADR-0018/0019 为准 |
| `src/physics/*` | 物理损失/残差模块属 v1（已暂停） | 只作离线诊断；启用条件见 ADR-0017/0018；不进训练 |
| `src/data/window_manifest.py`、`data_manifest.py`、`fragment_state.py`、`parameter_channel.py`、`nonlocal_table.py` | 代码本身可用，但其 manifest/统计语义是 v1 | 复用前按 Phase A 契约 v3 重建/校验 |

### 2.3 数据与产物（`.scratch/`、`output/`；均未提交）

| 入口 | 脏点 | 规则 |
|---|---|---|
| `.scratch/n/manifest*.json`（977 窗等） | v1 窗口语义、teacher 源统计 | **不得复用**；Phase A 重建 |
| `.scratch/n/eval_*.json`、`.scratch/n/fig/*` | v1 teacher-source 指标 + 错误配对 | 仅历史；不得作基线/Gate |
| `output/cfc_interp/**`（checkpoints/metrics） | v1 训练产物（含 `source[:, k+1]` 泄漏） | 仅历史；不得作基线 |
| `.scratch/n/frames/*.pt` | 原始帧缓存（状态/源/shard 按帧） | 理论可复用，但配对语义 v1；**最稳妥是重建**，至少加 ADR-0019 断言 |
| `output/source_head/**` | v1 source head（同号配对） | 仅历史；解析骨干部分可作证据 |
| `output/fragment_dynamics/v402_analytic/**` | 解析碎片骨干验收数据 | B1 可用作**解析骨干证据**（ADR-0015 注册值） |
| LOO 折残件 | 残缺样本 | 忽略 |

## 3. 新工作安全清单（Phase A/B 开工前必读）

**只引这些**：`roadmap_v2.md`、`docs/adr/README.md`、ADR-0018/0019、
`CONTEXT.md`（现行章节）、`report/model502-schema-and-source-protocol.md` §3.1a、
`report/marker-projection-causality.md`（修正版）、本文。

**可安全复用（加校验）**：原始 `jorek/part/projections` 文件（按 ADR-0019 重建配对）、
`src/data/model502_schema.py`、解析骨干相关模块（`spi_model`、`nonlocal_table`、
`fragment_dynamics`）、`src/models/ltc_cell.py`（组件）。

**禁止直接复用**：旧 manifest / checkpoint / 评测 JSON；`train_cfc_interpolation.py`
的训练语义；`interp_dataset.py` 的配对；`source_head` 的同号配对；GNN 栈；
`config.yaml` 的默认值；旧 ADR 的量化 gate。

**测试目录状态（2026-10-04）**：`tests/legacy/` 归档 20 个 v1 测试文件（GNN 栈、
teacher-source 配对、v1 physics loss、被取代的 sampler），pytest 不再收集；
现行套件 106 passed / 0 skipped（见 `tests/README.md`）。历史条目中的
`tests/test_*.py` 路径现为 `tests/legacy/test_*.py`。

**开工前两件代码事**：
1. 时间断言测试（jorek 单调；projection 配对时间差 = 0；part = 配对场帧 − dt；
   restart 登记事件：跨界面配对/滞后投影跳过、未登记 dt<=0 fail-fast，ADR-0020 决定 5）。
2. 按 ADR-0019 重建 EHL-2 配对（`(u_k, p_{k−1}=part{k}, S_k) → u_{k+1}`）。

## 4. 待办（quarantine backlog）

- 给 `gnn_encoder.py`、`gnn_decoder.py`、`dataset.py`、`lnn_processor.py`、
  `benchmark.py` 补 LEGACY banner。
- `CONTEXT.md` 逐条补 `[v1]` 标记（Eulerian 矩、flux-coordinate attention bias、
  source mapping sub-network、hybrid fragment dynamics 等）。
- `report/physics-residual-correction.md` 顶部补 scope banner。
- 考虑把纯 v1 报告移入 `report/legacy/`（需同步更新链接）；暂缓。
- 可加一条 CI/脚本检查：新文档引用 ADR-0001–0017 时必须带"（历史）"字样。
