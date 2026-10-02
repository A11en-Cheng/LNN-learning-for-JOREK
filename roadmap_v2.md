# Roadmap v2（2026-10-03）：复刻 JOREK 交错格式的神经网络代理

## 0. 本文档的定位

- **取代 `new_version.md` 成为工作路线图**。`new_version.md` 保留为 v1 记录：
  其中的数据契约、Phase 1.5/2 的历史结论、ADR 索引仍然有效，但**目标与训练语义**以本文为准。
- 依据：`report/stage-summary-2026-10-03.md`（阶段总结）、`report/pitfalls.md`（60 条踩坑）、
  `CONTEXT.md`（术语 + JOREK 耦合时序）、`docs/adr/0012..0017`。
- 读者：本人 + 后续 agent。**每个 Phase 都有产出与 Gate；未过 Gate 不展开下游。**

---

## 1. 上一个版本留下了什么（框架盘点）

### 1.1 可复用资产（不要重造）

| 层 | 资产 | 位置 | 状态 |
|---|---|---|---|
| 数据契约 | 九场 identity 通道 + symlog（`model502-state-v2`） | `src/data/model502_schema.py` | ✅ 有回归测试 |
| 数据管线 | manifest（8 run / 8 折 / `t_ns=8380` 切分 / blocked purge 划分）、帧缓存、shard 读取（16 字段）、投影读取（dof-0 节点值） | `src/data/{window_manifest,interp_dataset,fragment_state}.py` | ✅ |
| 参数通道 | 61 维零维参数（H5 → namelist → JOREK 默认 的取值链） | `src/data/parameter_channel.py` | ✅ 但**未接入网络** |
| 模型 | LTC 闭式传播子（恒等 + 半群到机器精度）、残差解码、显式 dt 特征、`multi_substep`、`advance_with_source_integral`（增广积分 `A(t)`）、`model_kwargs_from_meta` | `src/models/{ltc_cell,lnn_processor,cfc_interp}.py` | ✅ 约 17.2 万参数 |
| 物理（离线） | 九残差、弱形式算子（P1 拉普拉斯/各向异性/对流）、非局域烧蚀表、Parks/NGS 解析骨干 | `src/physics/*`、`report/spi-analytic-baseline.md` | ✅ 仅诊断，**不进训练** |
| 评测 | R²/MSE/relL2/**closer**/`derr|u|`/H1/budget/skill + **dt 分桶** + `--inner all` + JSON | `tests/eval_phase3_structure.py`、`tests/report_run_table.py` | ✅ |
| 诊断图 | loss 曲线 / 指标 vs dt / 逐场 R–Z 剖面（等比例） | `tests/plot_*.py`、`tests/make_diagnostics.sh` | ✅ |
| 探针 | 半群/恒等验收、时间一致性、cell 对照 | `tests/probe_*.py` | ✅ |
| 训练器 | `--cell/--input-dt/--horizon/--horizon-mode/--lr-schedule/--sub-steps/--save-best-metric` | `tests/train_cfc_interpolation.py` | ✅ |

### 1.2 结论性认知（已用实验固定，不要再试）

1. ncps CfC cell **不是时间传播子**（`ts=0` 非恒等、LSTM 不吃 ts）；LTC 闭式解才是（pitfall #43/#45/#46）。
2. **步数 S=1/2/5 对结果没有系统性影响**（LTC 的半群性）⇒ 步数不再是模型自由度（#59）。
3. **训练预算与 lr 是主因**：3e-3 + 1000 epoch + cosine 才能收敛；恒定 lr 在 ~ep620 后失稳（#47/#54）。
4. **dt 不是常数**（原始 55 种、监督 36 种，训练中位 5 / 评测中位 50）⇒ dt 必须显式入网（#48）。
5. **mean dR2 不能当主轴**（细 dt 失真）；用 `closer` + `derr|u|`，且 closer 跨 h 不可比（#49/#51）。
6. **评测窗口集合会改变结论**（R² 0.994 → 0.974）；Gate 必须写明窗口集合与 dt 桶（#58）。
7. **硬 argmax 磁轴失效**（relL2≲1% 时纯抖动）⇒ 用真轴节点 psi 值差（#54）。
8. **更长 warm-up 无收益**（10/15/20 对比，#59）。
9. **JOREK 的耦合是"场隐式 + 碎片显式 + 源显式 RHS + 投影单独求解"**，不是全隐式（`CONTEXT.md` 第 91 行起）。
10. **背景三源不存在**（namelist 恒 0）；`projections*.h5` 与 model502 内部驱动场的解析沉积**不是同一离散对象**（ADR-0016）。

### 1.3 明确"不要继承"

物理损失进训练回路、逐场分裂演化器（必发散）、jump-horizon 语义、20 epoch 扫描的绝对值、
单 seed 能力声明、把 `projections*.h5` 当作模型内部源的精确复刻。

---

## 2. 本版本 roadmap 的目的

**让神经网络成为 JOREK 交错格式里可替换、可自洽推进的模块，而不是一个"吃未来帧源"的插值器。**

三个可度量目的：

- **A（自洽）**：给定 `(u_k, p_k)`，在不使用任何**未来帧记录量**的前提下推进 N 帧，
  全程无 NaN、无冻结坍缩，且场级指标不低于同窗口的 data-only 插值基线。
- **B（时序对齐）**：训练与推理都按 JOREK 的错步格式（场一步隐式、碎片一步显式、源显式）。
- **C（接口清晰）**：明确"确定性复用 / 小网络学修正 / 端到端学"的边界，并为"能否在 JOREK 里加速"给出可复现的对照。

**非目标**：跨装置泛化、`ntor11` 三维模耦合、全粒子 GNN、物理损失（除非 Phase 4 重启条件满足）。

---

## 3. 要复刻的骨架：JOREK 交错格式（源码核实）

```
t_k:  (u_k, p_k)
  │
  ├─(1) S_k = Source(u_k, p_k)                源：解析源形状 + FEM 沉积
  │        model502: total_imp_source → inj_source（Gauss 点逐点，显式 RHS）
  │        marker 侧: sample_rhs → 投影矩阵解 M x = b → projections*.h5（帧平均，诊断/导出）
  │
  ├─(2) u_{k+1} = SolveImplicit(u_k, S_k)     九场同时隐式（construct_matrix + solve_sparse_system）
  │        源项只进 RHS、不进 LHS ⇒ 对场是显式
  │
  └─(3) p_{k+1} = FragEvolve(p_k, u_{k+1})    碎片：显式前向 Euler，步长 = 整个流体步 tstep
           update_spi: R/V/radius 显式推进；abl 用刚解出的局域场代入解析定标律
           （含 max(local, nonlocal)，无子循环、无隐式）
```

**接口边界（本版本的核心决策）**：

| 模块 | 选型 | 理由 |
|---|---|---|
| (1) 源的空间沉积（FEM 组装） | **确定性**（复用 JOREK 基函数/形函数） | 让网络学装配会把 FEM 截断误差当物理（1.5.1 gate 的教训，ADR-0016） |
| (1) 源强（烧蚀率、能量分配） | **解析骨干 + 有界学习修正** | 解析-only 已达逐 shard 中位比 1.00–1.09、log-corr ≥0.98（ADR-0015）；端到端学习反而更差 |
| (2) 九场一步推进 | **可学（本版本主战场）** | JOREK 每步解一次大稀疏系统，最贵、最有加速价值；但必须在**无未来源**的闭环下评估 |
| (3) 碎片推进 | 优先**确定性复用**（必要时小修正） | 显式、便宜、已有解析骨干 |

---

## 4. 详细线路图

### Phase 0 — 口径与基建收口（不需要新数据，最先做）

- **任务**：① 8 折 LOO 全量评测（manifest 已就绪）；② 关键配置 2–3 seed 复核；
  ③ Gate 表模板固定（窗口集合 + dt 桶 + R²/MSE/relL2/closer/derr/skill）；④ 把 `--inner all` 设为默认评测口径。
- **产出**：`output/cfc_interp/loo/<fold>/`、`report/gates/*.md`。
- **Gate**：任一结论在 2 个 seed 间的极差 < 已登记噪声带（R² 5% / closer 20%）。
- **回退**：若 folded 划分窗口数不足，改用 5 折分组（每折 2 run，ADR-0012 允许）。

### Phase 1 — 数据补强（依赖用户的新 2D 数据）

- **任务**：按优先级要 ① **子帧场 dump**（每 5–10 流体步一帧，1–2 run 即可）；
  ② **帧区间累计/逐流体步投影**；③ 更多 SPI 场景与可登记的 dump cadence；
  ④ 逐 shard 逐流体步烧蚀历史。
- **产出**：新 manifest（含子帧索引）+ 数据覆盖报告。
- **Gate**：子帧数据能与帧数据在同一 run 内对齐（时间戳连续、无缺口）。
- **回退**：若拿不到子帧数据，用现有帧间线性/二阶插值当"伪真值"，但所有结论必须标注"插值未验证"。

### Phase 2 — 自生成源（shard → 投影场）

- **目的**：把"未来帧教师源"从推理路径里拿掉。
- **结构**：shard 特征（16 字段 + 邻域场，FEM 形函数插值）→ 小网络输出**乘性修正**
  `M = exp(MLP(...))`（对烧蚀率与能量/辐射分配）→ 解析骨干 → **确定性 FEM 装配**到 4125 节点。
- **训练**：监督 = 逐帧记录的 5 通道投影（用其 dof-0 节点值）；损失用**逐场相对 L2 + 预算（总原子量/总辐射）**，
  并强制 `M` 有界（clamping，按先验物理极值）。
- **Gate**：held-out run 上逐场相对 L2 优于解析-only 基线；总烧蚀/辐射预算误差 ≤10%；
  对碎片位置/半径/速度的响应方向正确（单调性测试）。
- **回退**：若学习修正再次变差（ADR-0015 的先例），退回解析-only 并记录为负结果。

### Phase 3 — 九场一步推进器（本版本主战场）

- **3.1 基线（已有）**：data-only LTC 一步预测（吃目标帧教师源），Gate 见 §5。
- **3.2 去教师源**：把 (1) 换成 Phase 2 的自生成源，做"同窗口、同预算"的 A/B。
  **这是本版本最关键的对照实验**：教师源 vs 自生成源，指标差多少。
- **3.3 空间耦合**：Phi/omega 是唯一 closer>0.5 的场、误差呈环状（非局域耦合）。
  先做最便宜的：把**局部输运项**（图算子算出的 `{Φ,ψ}`、`{u,ψ}`、`advection`）作为显式输入特征；
  若不够，再上 stencil-conv / 谱分支；GNN 作为最后手段。
- **Gate**：一步（h=1）在 held-out 全窗口上：R² ≥ 0.98、`derr|u|` ≤ 0.09、closer ≤ 0.50，
  且 Phi/omega 的 closer 从 >0.5 降到 ≤0.5。

### Phase 4 — 闭环 rollout（把"插值器"变成"预报器"）

- **训练课程**：① 强教师（当前）→ ② Dagger（偏离阈值就注入真值纠偏）→ ③ 纯自回归长窗口。
- **硬防线**：所有学习修正乘子按先验极值 clamping；导数特征按训练集分位数 clamp（pitfall #24 的教训）。
- **Gate**：在登记 horizon 内（先做 H1：1–2 帧间隔）无 NaN、无冻结坍缩；
  场级 `derr|u|` 相对教师源基线劣化 < 30%；预算（原子/能量/辐射）闭合 ±20%。
- **回退**：闭环发散 → 缩短 horizon、加纠偏频率，或回到 Phase 3 的自生成源单步。

### Phase 5 — 与 JOREK 的接口与加速 benchmark

- **接口定义**：网络进 `(u_k, p_k)`、出 `(u_{k+1}, p_{k+1})`，不依赖任何未来帧；单位/JOREK 时间语义写清。
- **单步对照（必须做）**：取若干重启帧，用 JOREK 走一步 vs 网络走一步，
  逐场比较（R²/`derr|u|`/预算），并记录两者的**单步成本**。
- **Gate（能否承担加速）**：单步成本 ≤ JOREK 的 X%（待定，先测基线），
  且单步精度满足 §5 的 Gate；长窗口（H1）不发散。
- **产出**：接口文档 + 对照报告 + 可复现脚本。

---

## 5. 训练与实验规范（硬约束）

**当前基线配方（任何新实验的对照）**

```
cell=ltc  --input-dt  --input-deriv2  --input-condition  --residual  --loss-scale dynamics
lr 3e-3 (cosine → 3e-4)   batch 32 × 30 步/epoch   1000 epoch
λ_H1=0.01  λ_axis=0.3  λ_phys=0  --save-best --save-best-metric closer
```

**数据协议（2026-10-03 起）**：单一训练 = **7 个纯 2D run 的全部窗口**训练
（`--train-inner all`，含 inner split 标为 val/dropped 的窗口）+
**留出 run（v400）的全部窗口**验证（`--val-source held-out`，供 `--save-best`）。
零维参数（61 维）通过 `--input-condition` 在每个窗口接入（取传播区间起始帧，
用训练 run 统计量标准化）。开训前硬门槛：manifest 的
`summary.n_nonmonotonic_windows == 0`（每个窗口逐帧 `t_now` 严格递增，见
pitfalls #60/#61）。8 折 LOO 暂缓，后续作为泛化评估再启用。

**报告口径（缺一不可）**：loss / val_mse / R² / MSE / relL2 / `derr|u|` / closer / skill_pers；
**按 dt 分桶**；**写明评测窗口集合**（`--inner val|all`）；磁轴用"真轴节点 psi 值差"。

**实验规范**：同窗口、同预算、同 seed 组；每个结论至少 2 seed；跨 h 只比 R²/relL2/`derr|u|`。

---

## 6. 实验清单（可执行）

| 编号 | 目的 | 数据/命令 | 判据 |
|---|---|---|---|
| E0 | 复现当前基线 | 现有 manifest + 基线配方 | 与 `ltc_roll_h1_1000` 指标一致 |
| E1 | 8 折 LOO（暂缓） | 现有 manifest 的 8 个 fold | 逐 run 指标分布；2026-10-03 用户决定先做单一全窗口训练 |
| E2 | 多 seed | E0 ± seed | 噪声带登记 |
| E3 | 参数通道接入（已并入基线） | `--input-condition`（+61 维） | R²/`derr|u|` 是否改善 |
| E4 | shard→投影 head | Phase 2 结构 | 相对 L2 优于解析-only、预算闭合 |
| E5 | 去教师源单步 | 教师源 vs 自生成源 A/B | 劣化 < 30% |
| E6 | 闭环 rollout | 课程 1→2→3 | H1 不发散、指标可接受 |
| E7 | 空间耦合 | 输运项特征 → 谱/stencil → GNN | Phi/omega closer ≤0.5 |
| E8 | 子帧插值验证 | **需新数据**（Phase 1） | 插值点有真值可比 |
| E9 | JOREK 单步对照 | 重启帧 + 成本测量 | 精度 Gate + 成本比 |

---

## 7. 风险与回退

| 风险 | 触发 | 回退 |
|---|---|---|
| 学习修正又变差 | E4 指标劣于解析-only | 退回解析-only，记录负结果（ADR-0015 先例） |
| 闭环发散 | E6 NaN/爆量 | 缩短 horizon、提高纠偏频率、clamping 收紧 |
| 空间耦合成本过高 | E7 训练时间/显存不可接受 | 先用局部输运项特征，谱分支低秩近似 |
| 新数据拿不到子帧 | Phase 1 未完成 | 用帧间插值当伪真值，结论标注"未验证" |
| 单 seed 误判 | E2 显示噪声带大 | 所有 Gate 改为 2–3 seed 的中位/极差 |

---

## 8. 与 v1 roadmap 的差异（变更清单）

1. **目标**：v1 = "纯数据场插值器"；v2 = "JOREK 交错格式内可替换、可自洽推进的模块"。
2. **训练语义**：v1 吃目标帧教师源；v2 必须自生成源（Phase 2）并在闭环下评估（Phase 4）。
3. **物理损失**：v1 曾计划进回路（Phase 4），v2 明确**暂停**，重启前置 = 整体隐式矩阵或弱形式演化。
4. **度量**：v2 固定为 closer/`derr|u|` + dt 桶 + 窗口集合；弃用 mean dR2 与硬 argmax 磁轴。
5. **数据需求**：删除背景三源；新增子帧 dump 与区间累计投影为 P0。
