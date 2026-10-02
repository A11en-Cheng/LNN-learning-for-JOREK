# 阶段总结（2026-09-29 → 2026-10-03）：纯数据 LTC 场插值器

范围：EHL-2 的 8 个纯 2D run、九场状态、五通道 marker 投影作为教师驱动，
**全部走 data（物理计算 / 物理损失暂停）**。姊妹文档：`new_version.md`（路线图）、
`report/pitfalls.md`（坑与证据）、`report/implementation-progress.md`（逐日流水）。

## 0. 一句话结论

把 CfC 换成 **LTC 精确传播子**之后，"时间一致性"不再需要拿精度换：在同一批窗口上，
held-out 的 **R² / relL2 / closer / skill_pers 四项全部超过旧的 ncps cell**；
**积分步数 S=1/2/5 对结果没有系统性影响**（重写验收通过）；
**更长的 warm-up（15/20）没有收益**；物理损失与闭环因"需要把九个场整体隐式求解"仍暂停。

## 1. 本阶段做了什么（压缩时间线）

| 时间 | 内容 | 关键产出/结论 |
|---|---|---|
| 09-29/30 | 数据契约与物理模块 | ψ/Z_j 通道修复、symlog 契约、`t_ns` 切分、v200 删除；九残差模块与弱形式原型；ADR-0015（解析骨干交付）/0016（反组装 gate 取消）/0017（物理暂停） |
| 10-01 | 纯数据基线 + 评测 bug | 基线逐位复现；修 `eval` 漏传 `with_lap`（pitfall #42） |
| 10-01 | 网络安全体检 → **LTC 重写** | 证明 ncps cell 不是传播子（`ts=0` 非恒等、LSTM 不吃 ts）；实现闭式 LTC（softplus rate / tanh target），删 LSTM，backbone 256 |
| 10-01 | 子步消融（否证） | S=2 最好，S=8/25 单调变差（#43） |
| 10-01 | 训练预算 | 220→1200 步使平均 dR2 0.194→0.416（#44） |
| 10-01/02 | 学习率扫描 | lr 3e-3 最优（dR2 0.147→0.323 vs 3e-4）；1e-2 变差（#47） |
| 10-02 | dt 清点 + dt 特征 | 原始 55 种 dt、监督 36 种；`--input-dt`（37 维输入）不破坏半群性（#48） |
| 10-02 | **度量口径重建** | 弃用 mean dR2（细 dt 失真），改用 closer / `derr|u|` / R² / MSE，并按 dt 分桶（#49–#51） |
| 10-02 | 长训 + 调度 | 4000 步反超 ncps（#53）；恒定 lr 1000 epoch 在 ~ep620 后失稳 → cosine 3e-3→3e-4（#54） |
| 10-02 | 诊断三件套 | loss 曲线、指标 vs dt、逐场 R–Z 剖面；修 `to_device` 丢网格 bug（#52） |
| 10-02 | 训练集重规划 + 两组扫描 | 每窗口只监督"warm-up 末帧→下一帧"；S=1/2/5 与 warm-up 10/15/20（#58/#59） |

## 2. 数据与训练集（当前状态）

- **语料**：8 个纯 2D run（`ntor=1`，n=0 分量）；窗口在 `t_ns`（SPI 注入）处切开、
  参数实质变化处切分；统计量只由该折训练窗口算。
- **dt 不是常数**：原始帧间隔 **55 种**（0.5/1/2/2.5/5/10/20/25/50/100/250 + 0.9× 旁支 + 杂值），
  训练监督区间用到 **36 种**（中位 5、均值 17.5、最大 250）；评测窗口偏粗（中位 50）。
- **manifest**：`.scratch/n/manifest.json`（dense 16 帧窗口）、`manifest_h5`（horizon=5）、
  重规划后的 `manifest_b10/b15/b20`（每窗口**恰好 1 个监督区间**：
  base = warm-up 末帧 → target = 紧接的那一帧）。
- **三件套**：`jorek*.h5`（九场）、`part*.h5`（shard 状态，16 字段含 `abl` 烧蚀率）、
  `projections*.h5`（五通道，取 dof-0 的节点值）。**背景三源不存在**（namelist 恒 0）。

## 3. 模型（当前基线）

- **LTC 闭式传播子**：网络输出 `rate=softplus(head)`、`target=tanh(head)`，
  状态用 `h(ts)=h₀e^{−rate·ts}+target(1−e^{−rate·ts})` 推进（`src/models/ltc_cell.py`）；
  恒等性（ts=0）与半群性到机器精度（单测 + `tests/probe_ltc_semigroup.py`）。
- 逐节点 encoder（**37 维** = 9 场值 + 18 一阶导 + 9 二阶导 + 1 个 dt 特征）→ latent 64
  → LTC（hidden 128，backbone 256×2）→ 残差解码（输出增量）→ 9 场；**约 17.2 万参数**，无 LSTM。
- 控制项：**目标帧**的 5 通道投影（`latent += control_encoder(S)`）⇒ 当前是**插值器**，
  不是预报器；源项头（`predict_source` + 增广积分 `A(t)`）已实现但未启用。

## 4. 训练配方（当前基线）

```
cell=ltc  --input-dt  --input-deriv2  --residual  --loss-scale dynamics
lr 3e-3 (cosine → 3e-4)   batch 32 × 20 步/epoch   1000 epoch
λ_H1=0.01   λ_axis=0.3   λ_phys=0（纯 data）
--save-best --save-best-metric closer
```

loss = 逐场按 `1/E[Δ²]` 配平的 MSE（+ H1 结构项 + soft-argmax 磁轴项）。

## 5. 结果（**必须连同评测窗口集合一起读**）

| run | 训练 | 评测集 | R² | relL2 | closer | skill_pers(中位) |
|---|---|---:|---:|---:|---:|---:|
| ncps cell（旧） | 1200 步 | v400 val，15 窗 | 0.9909 | 0.0797 | 0.598 | +0.424 |
| LTC 初版 | 1200 步 | 同上（15 窗） | 0.9864 | 0.0997 | 0.786 | +0.242 |
| **LTC + dt，长训** | 4000 步 | v400 val，15 窗 | 0.9922 | 0.0729 | 0.521 | +0.632 |
| **LTC + dt，rollout h=1** | 1000 ep（cosine） | v400 val，13 窗 | **0.9937** | **0.0663** | **0.461** | +0.631 |
| LTC + dt，每窗口 1 区间（S=1） | 1000 ep（cosine） | v400 **全部**窗口，65 窗 | 0.9756 | 0.1042 | 0.481 | +0.384 |

**（重要）**最后一行 R² 低于前两行**不是模型变差**：换成"全部窗口"后评测集包含
28 个细 dt（3–15）与 9 个粗 dt（≥60）窗口；在**同一个 15–60 桶**里，新模型的
`derr/|u|` = 0.0529 反而**优于**旧模型的 0.0612 ⇒ "R² 0.994" 是窗口选得容易（pitfall #58）。

**分场（LTC + dt，v400，65 窗）**：relL2 —— psi 0.006 / T_i 0.017 / T_e 0.025 /
Z_j 0.060 / rho 与 rho_imp 0.062 / v_par 0.073 / **omega 0.289 / Phi 0.344**；
closer —— T_i 0.41 / omega 0.46 / rho_imp 0.46 / psi 0.47 / Phi 0.47 / Z_j 0.58 / v_par 0.58。
**Phi/omega 仍是唯一明显落后的两个场**（差值空间形态呈环状/非局域）。

**两个附加扫描（同批窗口、1000 epoch、cosine）**：

- **步数 S=1/2/5**（目标帧固定）：R² 0.9756 / 0.9741 / 0.9735，relL2 0.104 / 0.110 / 0.108
  ⇒ **无系统性差异**，步数不再是模型自由度（对照：旧 ncps cell 在 S=8/25 时明显变差）。
- **warm-up 10/15/20**（S=2）：R² 0.9741 / 0.9721 / 0.9714 ⇒ **更长 warm-up 无收益**
  （三个 manifest 目标帧不同，属"没有正向证据"，不是"越长越差"）。

## 6. 被推翻 / 更正的历史结论

1. "时间一致性要付 2.8× 精度（0.416 vs 0.147）" → **错**：那是 lr 太低 + 训练不足 +
   encoder 看不到 dt；修正后四项反超（pitfall #53）。
2. **mean dR2 不能当主轴**：细 dt 上分母进入噪声底（两模型都 −40…−60）；改用 `closer`
   （与前后帧的归一化相似度）与 `derr/|u|`，且 closer 跨 h 不可比（#49–#51）。
3. "R² 0.994" 部分来自**评测窗口集合**（val 只含最易 dt 桶）→ 必须写清窗口集合（#58）。
4. **硬 argmax 磁轴**在 relL2 ≲1% 时失效（顶部平坦、argmax 抖动 2–4 cm）→ 改用真轴 psi 值差（#54）。
5. "dt 只有 5 和 2" → 实际 36 种（训练）/55 种（原始），训练与评测分布差 ~10×（#48）。
6. "20 epoch 扫描的绝对值"是早期瞬态，不能当模型能力（#44）。

## 7. 工具与产物

- 评测：`tests/eval_phase3_structure.py`（R²/MSE/relL2/closer/derr/H1/budget/dR2 + dt 分桶 +
  `--json` + `--inner all`）、`tests/report_run_table.py`。
- 诊断：`tests/plot_training_curves.py`、`tests/plot_metrics_vs_dt.py`、
  `tests/plot_field_slices.py`、`tests/make_diagnostics.sh`；图在 `.scratch/n/fig/`。
- 探针：`tests/probe_ltc_semigroup.py`（恒等/半群验收）、`tests/probe_time_consistency.py`、
  `tests/probe_cell_semigroup.py`。
- 训练：`tests/train_cfc_interpolation.py`（`--cell/--input-dt/--horizon/--horizon-mode/
  --lr-schedule/--sub-steps/--save-best-metric`）；脚本在 `.scratch/n/*.sh`。
- 主要 checkpoint：`output/cfc_interp/{long60x20, ltc_dt_lr3e3_long200, ltc_roll_h{1,5}_1000,
  step_s{1,2,5}_b10, warm_b{15,20}_s2}`。

## 8. 未完成 / 下一步（不需要新数据的排在前面）

1. **8 折 LOO 全量评测**（manifest 已就绪，只用过 v400 一折）；
2. **多 seed 复核**（路线图硬要求；单 seed 噪声 R² ~5%、closer 可到 20%）；
3. **参数通道（61 维）接入网络**（数据每帧都有，目前完全未用）；
4. **shard 状态 `p(t)` 接入网络**（数据集支持，训练全用 `with_shards=False`）；
5. **自回归 / 无教师输入版 rollout**（把"插值器"变成"预报器"的前置）；
6. **空间耦合**（Phi/omega 的环状误差指向非局域耦合：GNN 或谱分支）；
7. **物理损失 / 闭环**（暂停；重启前置 = 整体隐式矩阵或弱形式演化）。

## 9. 若补新 2D 数据（建议）

| 优先级 | 数据 | 解锁 |
|---|---|---|
| P0 | **子帧场 dump**（每 5–10 流体步一帧，哪怕 1–2 个 run） | 唯一能直接验证"非均匀时间插值"的数据（现在插值点没有真值） |
| P0 | 帧区间累计 / 逐流体步**投影**导出（路线图 P0a） | 源项语义与预算闭合 |
| P1 | 更多 SPI 场景（速度/角度/量级）+ 可登记的 dump cadence | LOO 场景泛化（现训练集一个 run 占 405/628 窗口）、跨 run 可比性 |
| P2 | 逐 shard **逐流体步**烧蚀历史（P1 级导出） | 碎片动力学 gate |
| — | ~~背景三源~~ | **不需要**：namelist 里恒为 0 |
