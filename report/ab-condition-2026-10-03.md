# Condition on/off A/B（2026-10-03）

**问题**：2026-10-03 的单一全窗口训练（7 个 run 全部 916 窗训练 + v400 全部
61 窗验证，`--input-condition` 接入 61 维零维参数）在真 OOD 评测上明显劣于旧基线。
用户判断这是 condition 通道造成的，并决定做唯一变量 A/B 后收口本阶段实验。

## 实验设计

两组训练**完全相同**，唯一差别是 `--input-condition`：

| 项目 | 值 |
|---|---|
| manifest | `.scratch/n/manifest_h5.json`（977 窗；`n_nonmonotonic_windows=0`） |
| 训练窗口 | 7 个 run 的全部窗口 = **916**（`--train-inner all`） |
| 验证/选优 | v400 全部 61 窗（`--val-source held-out`） |
| 配方 | LTC + dt + 一/二阶导数 + residual + dynamics loss；1000 epoch × 30 步；batch 32；cosine 3e-3→3e-4；λ_H1=0.01、λ_axis=0.3、λ_phys=0；seed 相同 |
| 唯一变量 | condition-on：61 维 `build_condition_vector`（含输运系数/run 指纹通道）；condition-off：不接 |

训练/评测产物：`output/cfc_interp/allwin_cond_h1_1000/`、
`output/cfc_interp/allwin_nocond_h1_1000/`、`.scratch/n/eval_{allwin_cond,allwin_nocond}_*.json`。

## 主结果（v400 全部 61 窗，同一 manifest_h5）

| run | mean R² | mean relL2 | mean closer | 说明 |
|---|---:|---:|---:|---|
| cond_best（ep34） | 0.9681 | 0.1248 | 0.5721 | 61 窗 |
| cond_final（ep1000） | 0.9321 | 0.1892 | 0.6329 | 61 窗 |
| **nocond_best（ep437）** | **0.9744** | **0.0983** | **0.4177** | 61 窗 |
| nocond_final（ep1000） | 0.9744 | 0.0983 | 0.4374 | 61 窗 |
| old step_s1 final（参考：无 condition、旧 inner-train 配方） | 0.9778 | 0.0933 | 0.4453 | 同 61 窗重评 |

**逐 dt 桶步进误差 `derr/|u|`**（同一批窗口）：

| bucket | n | cond_best | cond_final | nocond_best | nocond_final | old step_s1 |
|---|---:|---:|---:|---:|---:|---:|
| dt<3 | 1 | 3.118 | 1.190 | 0.539 | **0.379** | 0.335 |
| 3–15 | 28 | 0.133 | 0.170 | 0.121 | 0.122 | **0.110** |
| 15–60 | 27 | 0.089 | 0.153 | 0.052 | **0.049** | 0.053 |
| ≥60 | 5 | 0.252 | 0.426 | 0.177 | **0.169** | 0.173 |

逐场 R²（best-vs-best）：Phi 0.858→0.874、omega 0.884→0.906、
Z_j 0.994→0.999、v_par 0.991→0.995、rho_imp 0.996→0.999；
nocond 已基本追平旧 step_s1（R² 0.9744 vs 0.9778），且条件组的后期退化
（final Z_j 0.899、omega 0.699）在对照组中不存在。

## 逐窗诊断（condition 的伤害集中在哪）

condition-on best 的 per-window mean derr 按窗口起点：0–4 = 3.398、5–9 = 4.133、
10–14 = 1.669、15–20 = 0.197、≥21 = 0.109——误差几乎全部来自 v400 **注入后早期
斜坡**（`rho_imp` 从 0.006 涨到 0.013、`Phi≈0`、`omega≈2e-4` 的窗口）。
condition-off 把该段拉回：dt<3 窗 3.118→0.539、3–15 桶 0.133→0.121；
发展期 15–60 桶 0.089→0.052、≥60 桶 0.252→0.177。

场剖面单窗数值（`checkpoint_best`）：

| window | cond Phi/omega/rho_imp relL2 | nocond Phi/omega/rho_imp relL2 |
|---|---|---|
| w0（注入初段） | 3.47 / 4.83 / 19.68 | 0.60 / 1.56 / 2.67 |
| w30（发展期） | 0.114 / 0.088 / 0.066 | 0.045 / 0.063 / 0.029 |

## 图

- A/B 主图（逐场 R²/relL2 + 分 dt 桶 derr）：`.scratch/n/fig/allwin_condition_ab.png`
- 五 checkpoint 全表：`.scratch/n/fig/allwin_condition_ab_full.{png,md}`
- 训练曲线对比：`.scratch/n/fig/allwin_ab_curves.png`
- 单组：`allwin_{cond,nocond}_curves.png`、`allwin_{cond,nocond}_metrics_vs_dt.{png,csv}`、
  `allwin_{cond,nocond}_slices_w{0,30}.png`

## 结论

1. **确认是 condition 通道造成的**：同 manifest、同预算、同 seed，只去掉
   `--input-condition`，R² 0.9681→0.9744、relL2 0.1248→0.0983、closer 0.5721→0.4177；
   所有 dt 桶和全部 9 个场都改善，注入初段改善最大（dt<3 误差 3.12→0.54）。
2. 旧 `step_s1`（无 condition、旧 inner-train 配方）在同 61 窗上是 0.9778/0.0933/0.4453，
   即 **condition-off 全窗口训练已基本追平旧基线**；说明"全窗口训练 + 真 OOD 评测"本身
   不是问题，问题在当前 61 维 condition 的构造。
3. 但这不等于"不要 condition"：当前向量混合了物理场景量（弹丸质量/速度/角度、`n_spi`）
   与 run 指纹类量（`eta`、输运系数、`central_density` 等）。OOD 场景下后者提供的是
   不可迁移的捷径。下一轮把 condition 拆成"物理场景子集"与"数值/输运子集"分别做
   A/B，只保留前者（或对后者做低秩/正则），才是可用的 condition 方案。

**本阶段在此收口**：主实验结论 = 全窗口 + v400 全窗 OOD 评测 + 无 condition（或仅物理
条件子集）是当前可信基线；61 维全集 condition 暂不作为默认配置。
