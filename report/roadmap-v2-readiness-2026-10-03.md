# Roadmap v2 就绪评估与仓库清点（2026-10-03）

范围：评估 [roadmap_v2.md](../roadmap_v2.md) 的可执行性与内部一致性，清点代码/数据/产物/文档/环境，
核对 §1.1 资产表与 E0–E10 的真实状态，给出进入 Phase 0 / 3.1b（E10）前的准备清单。
证据来源：仓库全量测试、`manifest_h5.json`、`output/cfc_interp/` 的 checkpoint 与 `metrics.json`、
`.scratch/n/eval_*.json`、训练脚本与 `.scratch/n/run_*.sh`、ADR-0012..0017、`report/pitfalls.md` #60–#63。

> **2026-10-04 勘误（后补）**：本报告的"最可信基线"框架已被 ADR-0018 取代——旧
> teacher-source 结果只作历史证据，不作为基线或 Gate。§1.2 中涉及时序的部分以 ADR-0019
> 为准（`projections{n}` 驱动 `u_n→u_{n+1}`；`part{n}` 在 `T_{n−1}`；旧管线一步未来泄漏）。
> 本报告的问题清单（roadmap 内部矛盾、选优协议、单 seed、细粒度数据）仍然有效。

## 0. 结论（TL;DR）

1. **仓库处于"可立即开跑"的状态**：162 passed / 5 skipped（83 s）、8 个 run 的帧缓存齐全（1.2 GB）、
   `manifest_h5` 硬门槛 `n_nonmonotonic_windows == 0`、基线 checkpoint 与全窗评测 JSON 都在；
   GPU（RTX 5070 Ti）在提权后可用。
2. **当前最可信基线 = `output/cfc_interp/allwin_nocond_h1_1000/checkpoint_best.pt`（ep437，无 condition）**：
   v400 全 61 窗 R² **0.9744** / relL2 **0.0983** / closer **0.4177**；
   距 Phase 3 一步 Gate（R² ≥ 0.98、`derr|u|` ≤ 0.09）只差 **−0.0056 / +0.0083**，
   closer 与 Phi/omega 两项已达标（0.4177；Phi 0.4445 / omega 0.4344）。
   ⇒ roadmap 把"单步表达力拯救（E10）"放在 3.2/Phase 4 之前，判断正确。
3. **roadmap_v2 有三处内部矛盾必须先修**：① Phase 0 的"8 折 LOO 全量评测"与 §5/E1 的"暂缓"冲突；
   ② E0 参照的 `ltc_roll_h1_1000` 是旧协议（rollout / 20 步每 epoch / 548 训练窗 / 13 窗评测），
   与现基线不可比；③ §5 配方缺实际使用的 `--horizon-mode jump --sub-steps 2 --burn-in-substeps 1`。
4. **两个方法论缺口会直接污染 E0/E2/E10 的结论**：
   (a) `--save-best` 仍只在 held-out run 前 12 个窗上选优（pitfall #62），而 A/B 要分辨 ~0.3–0.6% 的 R² 差异；
   (b) 选优集与最终评测集同为 v400 ⇒ 报告的 best 数字带选择偏差；当前没有任何 >1 seed 的基线（E2 未做）。
5. **E10/E3/E7 的代码全部未实现**：可学习时间尺度 τ、Heun 二阶积分、FiLM/AdaLN、stencil-conv
   在 `src/` 中均无实现；Phase 2 的 source head 只有 5 epoch 的旧探索（train list 还含已删除的 v200 run）。

## 1. roadmap_v2.md 评估

### 1.1 优点（可直接执行的部分）

- **Gate-gated 结构**（ADR-0002）：每个 Phase 有产出、Gate、回退，且明确"未过 Gate 不展开下游"。
- **§3 的 JOREK 交错格式骨架 + 接口边界表**是本版本最有价值的增量：把"源沉积确定性 / 源强解析骨干+
  有界修正 / 九场可学 / 碎片确定性"分区，避免了 1.5.1 反组装 gate 的旧错误（ADR-0016）。
- **§1.2 结论性认知（1–14）与 §1.3 "不要继承"**逐条有 pitfall/报告编号支撑，等价于一份防重蹈覆辙清单。
- **§5 的硬约束**（同窗口/同预算/同 seed 组、跨 h 只比 R²/relL2/derr、报告必须写窗口集合与 dt 桶）
  已经是经过 #58/#62/#63 教训固化的规范。

### 1.2 需要修正的内部矛盾与过时引用

| # | 位置 | 问题 | 建议改法 |
|---|---|---|---|
| 1 | §4 Phase 0 ① vs §5/E1 | Phase 0 仍写"8 折 LOO 全量评测"，但 2026-10-03 用户已决定 LOO 暂缓（E1 也标"暂缓"） | 改为"LOO 暂缓（2026-10-03 用户决定）；保留 `.scratch/n/run_loo.sh` 与 manifest 折，启用条件 = Phase 5 泛化评估或用户指定" |
| 2 | §6 E0 判据 | 参照 `ltc_roll_h1_1000` 是旧协议：`--horizon-mode rollout`、20 步/epoch、548 训练窗、13 窗 `--inner val` 评测，与现基线（916/61/jump/30 步）不可比 | E0 改为以 `allwin_nocond_h1_1000/checkpoint_best.pt`（ep437）为冻结基线：先重评核对 61 窗数字，再决定是否重训复现 |
| 3 | §5 配方 | 缺 `--horizon-mode jump --sub-steps 2 --burn-in-substeps 1`；且 `report/stage-summary` 的 20 步/epoch 与现状 30 步/epoch 并存 | 把配方替换为 `.scratch/n/run_allwin_nocond.sh` 的实际命令并标注版本日期 |
| 4 | §4 Phase 0 任务 | 未包含 pitfall #62 的选优协议缺陷（`_evaluate max_windows=12`；cond 组 best 停在 ep34） | Phase 0 增加"选优协议"任务：定义并登记 ①选优窗口集合（覆盖 dt 桶）②选优与最终评测是否共用 v400 |
| 5 | §4 Phase 3 Gate 表述 | "Phi/omega 的 closer 从 >0.5 降到 ≤0.5"沿用旧口径；当前 61 窗下已是 0.4445/0.4344 | 把 Gate 拆成"当前值 vs 目标值"两列，避免误读为未达成；剩余缺口 = R² 与 `derr|u|` |
| 6 | §1.1 资产表"评测 ✅" vs Phase 0 ④ | `tests/eval_phase3_structure.py --inner` 默认仍是 `val`，Phase 0 要求改为 `all` 默认（未实施） | 作为 Phase 0 的代码小改项（或由外层脚本强制传 `--inner all`） |
| 7 | §1.2 #11 行文 | "去掉后基本追平旧基线"实际已**超过**旧基线（0.9744/0.0983 vs 0.9778/0.0933 是同批 61 窗的两组；须写清各自 epoch 与选优协议） | 在 A/B 报告与本 roadmap 中都注明"best 数字带选优偏差" |

### 1.3 Phase 3 一步 Gate 与现状（`--inner all`，v400 全 61 窗）

| Gate 项 | 目标 | `allwin_nocond` best（ep437） | 旧 `step_s1` final（同 61 窗重评） | 判定 |
|---|---:|---:|---:|---|
| mean R² | ≥ 0.98 | 0.9744 | 0.9778 | **未过**（差 0.56% / 0.22%） |
| `derr\|u\|` | ≤ 0.09 | 0.0983 | 0.0933 | **未过**（差 9.2% / 3.7%） |
| mean closer | ≤ 0.50 | 0.4177 | 0.4453 | 过 |
| Phi / omega closer | ≤ 0.50 | 0.4445 / 0.4344 | 0.4705 / 0.4535 | 过 |

`derr|u|` 的 dt 分桶（n = 1 / 28 / 27 / 5）：dt<3 **0.539**、3–15 0.121、15–60 0.052、≥60 0.177。
dt<3 只有 v400 注入后斜坡的 1 个窗口（#63 已证在 condition-on 下发散），其余桶已接近或优于 0.09。
**判读**：E10 的空间主要在细 dt 与整体拟合，而不是"重新设计架构"；同时注意这 0.22–0.56% 的差距
是否只是单 seed 噪声，必须先做 E2。

## 2. 仓库清点

### 2.1 代码

| 区域 | 规模 | 状态 |
|---|---|---|
| `src/data`（13 文件 / 3836 行） | schema、manifest、interp dataset、fragment state、parameter channel、nonlocal table、marker projection、ablation series | 当前主路径；`model502-state-v2` 有契约测试 |
| `src/models`（10 / 2655） | LTC cell、lnn_processor、cfc_interp、source_head、fragment_*、gnn_* | gnn_encoder/decoder 属旧 GNN 路线（E7 最后手段备选）；其余为当前 |
| `src/physics`（12 / 2692） | 残差、弱形式、FEM 算子、SPI 解析模型、ADAS 闭包 | 仅离线诊断（ADR-0017 暂停进训练） |
| `src/evaluation`（3 / 1079）、`src/training/trainer.py`、`src/utils`、`src/diagnostics` | 旧 GNN evaluator/trainer + 可靠性校验 | 与当前插值路线并存；当前训练脚本在 `tests/train_cfc_interpolation.py` |
| `tests`（87 文件 / 13349 行） | 单测/探针/训练与评测脚本/绘图 | 162 passed / 5 skipped；真数据测试依赖 `data/EHL-2`（symlink，存在） |

### 2.2 测试与探针（本次实跑）

- `.torch-cuda-env/bin/python -m pytest -q`：**162 passed, 5 skipped, 82 s**；2 条 warning 仅为 `torch.jit.script` 弃用。
- 跳过项：3 × `test_model502_full_loss.py`（缺 cached real frame）、2 × `test_segmented_sampler.py`
  （旧九场流程的 frame cache 已退役，superseded by `window_manifest`）。
- LTC 结构性质（恒等/半群）由 `tests/test_ltc_cell.py`、`tests/probe_ltc_semigroup.py` 守护，全绿。

### 2.3 数据与缓存

| 项 | 状态 |
|---|---|
| `data/EHL-2` | symlink → `/mnt/d/EHL-2/`（735 GB，11 个目录；8 个纯 2D eligible + 3 个 ntor11） |
| `manifest_h5.json` | 977 窗 / 8 run / 30 segment / `n_nonmonotonic_windows = 0`；折-5 训练 916 窗、v400 验证 61 窗 |
| 帧缓存 `.scratch/n/frames` | 8 个 run 全部命中（1.2 GB），E0/E2/E10 可直接开跑，无需重建 |
| 时间倒退事故 | `scan_1d21_ang12_massloss_neon` frame 119→120 已切段；`n_nonmonotonic_windows = 0` 硬门槛通过 |

### 2.4 产物与报告

- `output/cfc_interp/`：96 个子目录 / 121 MB，含 `ltc_roll_h1_1000`、`allwin_{cond,nocond}_h1_1000`、
  `step_s{1,2,5}_b10` 等；`allwin_nocond` 训练实测 12123 s（1000 epoch × 30 步，≈3 h 22 min/次）。
- `.scratch/n/`：评测 JSON/日志 + 33 张图（condition A/B、曲线、dt 桶、剖面）。
- LOO：只有 fold-0 的 `checkpoint_best.pt`（ep40 残缺样本），不作为结论（用户已决定暂缓）。
- 报告：`stage-summary-2026-10-03`、`ab-condition-2026-10-03`、`pitfalls` #1–#63、`roadmap-review-2026-09-29`、
  `spi-analytic-baseline`、`marker-projection-*` 等；`report/README.md` 索引本次补齐。

### 2.5 环境

- 解释器必须是 `.torch-cuda-env/bin/python`（torch 2.11.0+cu130）；系统 `python3` 无 torch。
- GPU：RTX 5070 Ti；**沙箱内 `torch.cuda.is_available() = False`，提权后为 True** ⇒ GPU 训练/评测需逐次授权。
- `gh` CLI 在本环境不可用（issue tracker 流程暂不可执行）；git worktree 干净，无未提交改动。

## 3. 过往工作与成果评估（对照 §1.1 资产表）

| 资产 | roadmap 声明 | 核实结果 |
|---|---|---|
| 数据契约（九场 + symlog） | ✅ 有回归测试 | ✅ `test_normalization_contract`、`test_model502_schema` 全绿 |
| 数据管线（manifest/缓存/shard/投影读取） | ✅ | ✅ 977 窗、时间倒退已修、8 run 缓存齐全 |
| 参数通道（61 维取值链） | ✅ 查表链；concat 已证伪 | ✅ 代码在；A/B 确认 concat 伤害 OOD（#63）；分组+FiLM **未做** |
| 模型（LTC 传播子/残差解码/dt 特征/增广积分） | ✅ ~17.2 万参数 | ✅ 单测/探针全绿；`teacher_source=True`，增广积分已实现但未启用 |
| 物理（离线残差/弱形式/非局域表/解析骨干） | ✅ 仅诊断 | ✅ ADR-0017：ψ 独立预测 dR2≈0.97；扩散方程刚性 ⇒ 必须隐式 |
| 评测（R²/MSE/relL2/closer/derr/H1/budget/skill + dt 桶 + JSON） | ✅ | ✅ `eval_phase3_structure.py` 可用；但 `--inner` 默认仍 `val` |
| 诊断图 | ✅ | ✅ `.scratch/n/fig` 33 个 |
| 探针（半群/恒等/时间一致性/cell 对照） | ✅ | ✅ 全绿 |
| 训练器（cell/dt/horizon/lr-schedule/sub-steps/save-best） | ✅ | ✅ 全部存在；τ/Heun/FiLM/stencil 不存在 |

### 3.1 E0–E10 实际状态

| 编号 | 状态 | 证据/缺口 |
|---|---|---|
| E0 复现基线 | **未做（参照过时）** | 现基线 checkpoint 在，但 E0 文本指向旧 run；建议先重评再决定重训 |
| E1 8 折 LOO | 暂缓（用户决定） | 仅 fold-0 残件；`.scratch/n/run_loo.sh` 保留 |
| E2 多 seed | **未做** | 所有产物 `seed = 0`；噪声带未登记 |
| E3 通道分组 + FiLM | 部分 | concat A/B 已完成（负结果）；分组与 FiLM **无代码** |
| E4 shard→投影 head | 有代码、未过 gate | `source_head` 仅 5 epoch 旧探索；hybrid 只改善 3 个电荷/密度通道，dE_ion 变差；train list 含已删除 run |
| E5 去教师源单步 A/B | 未开始 | 自生成源未接入训练器 |
| E6 闭环 rollout | 未开始 | 课程 teacher forcing 未实现 |
| E7 空间耦合（stencil-conv 优先） | 未开始 | 无 stencil/谱代码；GNN 旧路径仍在且测试绿 |
| E8 子帧验证 | 未开始（缺数据） | 回退规则已定（禁止伪真值） |
| E9 JOREK 单步对照 | 未开始 | 无重启帧对照脚本/成本测量 |
| E10 LTC 表达力拯救 | **未开始** | ①lr/schedule 复核未做；②τ_θ、③Heun 均无代码 |

### 3.2 已固化的关键结果（可直接作为对照）

- LTC 四项（R²/relL2/closer/skill）超过旧 ncps cell；S=1/2/5 无系统性差异（半群性验收）。
- 训练预算与 lr 是主因：3e-3 + cosine 才能收敛；恒定 lr ~ep620 后失稳。
- 61 维全集 concat 是 OOD 捷径：同 manifest/预算/seed，R² 0.9681→0.9744（condition off）。
- 帧间插值必须显式入网（dt 非均匀，训练 36 种 / 评测中位 50）。
- 评测窗口集合决定结论（R² 0.994 → 0.974）；硬 argmax 磁轴失效，改真轴 psi 值差。

## 4. 推进 roadmap_v2 的准备清单（按顺序）

### P0a — 文档与协议收口（不需要训练）

1. 修 §1.2 表中的 7 处 roadmap 矛盾（重点是 LOO 状态、E0 参照、§5 配方、选优协议）。
2. 新建 `report/gates/` 模板：窗口集合、dt 桶、R²/MSE/relL2/closer/`derr|u|`/skill、seed 组、provenance。
3. `eval_phase3_structure.py --inner` 默认改为 `all`（或包装脚本强制传参）。
4. **冻结选优协议**（E0/E2/E10 的共同前提），二选一并登记：
   (a) 固定预算、不早停（报告 final + 每 N epoch 的全窗曲线）；或
   (b) 选优用训练 run 的 inner-val / 独立 run，v400 只作最终评测；
   无论哪种，都必须覆盖 dt 桶，不能只看前 12 窗（#62）。

### P0b — E0/E2（第一次 GPU 批）

5. E0a：用现有 `allwin_nocond_h1_1000/checkpoint_best.pt` 重跑
   `eval_phase3_structure.py --inner all`，核对 0.9744/0.0983/0.4177 三个数与 JSON provenance（分钟级）。
6. E0b（建议）：seed 0 重训一次（≈3.4 h），验收 `best_epoch`/曲线形状与现 checkpoint 一致，冻结为"基线 v2"。
7. E2：seed 1/2 同配方（≈6.8 h），登记 R²/relL2/closer/derr 的 seed 极差；极差 ≥ Gate 余量（0.22–0.56%）
   时，E10 的 A/B 必须按 2 seed 中位数判定。

### P1 — E10（3.1b，进入 Phase 3.2 的硬前置）

8. ① lr/schedule 复核：warmup、EMA、不同 cosine 终点（同窗口/同预算 A/B）。
9. ② 可学习时间尺度 τ_θ：`rate = softplus(...)/τ`，τ 初始化 1（按层或按场共享）；改动集中在 `src/models/ltc_cell.py`，
   重跑恒等/半群探针后可复用现有训练器。
10. ③ Heun 二阶解析积分：子区间内对 rate/target 线性外推后积分；ts=0 恒等、半群以二阶精度近似，
    必须重跑 `tests/probe_ltc_semigroup.py` 验收。
11. 每项通过后才动下一项；单步 Gate（R²≥0.98、derr≤0.09、closer≤0.5）不过 ⇒ 停在 3.1b。

### P2 — 依赖新数据或新代码

- E3：先做 E3 的 61 维通道分组（物理场景 vs run 指纹），再实现 FiLM/AdaLN（当前无代码）。
- E7：stencil-conv 原型（复用 `window_manifest` 的 FEM size/vertex 与 `field_sampling`）不依赖新数据，可在 E10 并行做小样验证。
- E4/E5/E6/E9：source head 重训（注意剔除已删除 run、加 history 输入）→ 自生成源 → 闭环 → JOREK 单步对照。

### 预算估计（以实测 12123 s/次为准）

| 批次 | 内容 | GPU 时 |
|---|---|---:|
| P0b | E0b + 2 seed | ≈ 10 h |
| P1 | E10 三项 ×（1–2 seed） | ≈ 14–20 h |
| P2（先做） | E3 分组 + FiLM、E7 stencil 原型 | 视实现而定，首批 ≈ 10 h |

### 阻塞与风险

- **GPU 需提权**：沙箱内 CUDA 不可见；每批训练/评测前需申请授权（本次已验证提权后可用）。
- **选优/评测同 run 偏差**：现报告的 best 数字（含 A/B）都带这一偏差，跨实验可比但绝对值乐观；P0a-4 必须先定协议。
- **单 seed**：任何 <0.6% 的 R² 差异在 E2 之前不可判定；E10 的 A/B 设计与 Gate 判读都受此约束。
- **数据 P0 未到**：子帧/逐流体步投影缺失 ⇒ E8/E5 的区间内验证与预算闭合只能按回退规则执行。
