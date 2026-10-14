# Phase B1 自生成源模块（shard/p → S）：组件、单步训练/验证与物理验收（2026-10-07）

**状态**：current。**依据**：ADR-0018（决定 1/3/8/11/12）、ADR-0019（配对/时间语义）、
`roadmap_v2.md` §4 Phase B1 / §5 第 3、7 条 / §6 B1 行 / §7 风险、`report/data-contract-v3.md`、
`report/data-export-spec-v3.md`。ADR-0015 的重启条件以"（历史）"引用。
**非目标**：闭环 rollout（Phase C）、stencil/场推进器改动（B2）、边界拓扑（用户侧）、
GNN/谱分支、legacy/roadmap/ADR 改动。

> ## 勘误（2026-10-07，ADR-0020 决定 1/2）
>
> 本报告标题与正文原先把 `projections*.h5` 称为"登记代理（proxy）"、把
> `inj_source` 语义当作唯一目标，并按 roadmap_v2 原文写成 "proxy vs inj_source"。
> 经 JOREK 源码复核（`mod_elt_matrix.f90:944,972,989-1003,1079,1129,1179,1219,1542,1717,1728`）
> 与生产 namelist（`use_marker=.t.`）确认：**marker run 里记录投影就是能量/辐射/
> 电荷通道的真实场源**；解析 `total_imp_source` 只负责杂质密度源。因此：
>
> * §2/§5/§6 中的"proxy（登记代理）"一律读作"**marker feedback 真源**"；
> * 本报告 §5 的"proxy vs inj_source 差异"改写为"**解析 shard 沉积 vs marker
>   feedback 的对象差异（支撑/标度/云分布）**"，且该差异是 ADR-0020 决定 2
>   要求 marker 链 + 有界残差去闭合的部分，不是"不应存在的误差"；
> * "投影 relL2 ≈1.0" 的读法改为：解析 shard 沉积与 marker 云对象不同
>   （48–92% 投影质量落在解析沉积为 0 的节点），**不能**用解析-only 直接对齐
>   marker feedback；B1.1 的解析骨干仍按逐 shard 烧蚀率（1.0021）与原子预算
>   （1.0002）验收，B1.2 的 marker 链（`--marker-chain`）才是与投影同对象的监督臂；
> * 本报告未改动的数字（1.0021 / 3.8e-7 / 22× / 14.8×）仍然有效，
>   它们是对象级证据，不是性能 gate。

**范围**：`src/models/source_module.py`（自生成源：确定性碎片演化 + FEM 局域采样 +
解析骨干 + 有界学习修正 + split heads + 确定性 FEM 组装）、
`src/models/fem_sampling.py`（双线性形函数采样/组装）、
`tools/train_source_module_b1.py`（单步训练/验证 + 预算/单调性验收）、
`src/data/budget.py` 的烧蚀口径对账、`tests/test_source_module.py`。

---

## 1. 结论摘要

1. **源模块已按 ADR-0019 契约建成**：`(u_{k-1}, u_k, p_{k-1}) → S_k`，推理路径只使用窗口两端
   状态与窗口起点碎片状态，没有任何未来帧记录量；碎片演化 `p_k = FragEvolve(p_{k-1}, u_k)`
   与 JOREK `update_spi` 的字段/顺序逐项对齐（`models/pellet_module.f90:280-330, 760-800`）。
2. **解析骨干（默认交付，学习修正关闭）在真实帧上复现 JOREK 碎片响应**：
   523 个验证窗口上窗口末端逐 shard 烧蚀率总和比 **中位 1.0021**（P10 0.976 / P90 1.023），
   半径相对误差中位 **3.8e-7**，逐窗口原子预算比 **1.0002**；分相位（early/mid/late）
   的解析律对照为 1.006 / 1.004 / 1.004（§6.3 fidelity probe），
   与 ADR-0015（历史）登记的"中位比 1.00–1.09、within-2x 85–96%"量级一致。
   **非局域 `max(local, nonlocal)` 分支是必需的**：只开局域分支持 late 相位比 0.355。
3. **`projections*.h5` 不能监督这个对象**：把解析组装（`inj_source` 语义）与记录的 marker
   投影逐通道对比，最小二乘标度差 11–5.5e5、中位积分比 179–2.1e6、空间相关 0.06–0.29、
   **48–92%** 的投影目标质量落在解析场为 0 的节点上（§5）。这是 ADR-0016/0018 所说的
   "两个离散对象"的量化证据，因此本报告只把投影作为**登记代理**，不设任何性能 gate；
   把标度强行校准到投影会把原子预算放大 22×。
4. **有界学习修正（消融臂）的增益集中在辐射通道，且记忆头在单步窗口无可测收益**：
   同窗口/同预算/同 seed 下，原子预算已无空间（1.20% → 1.24–1.35%，
   解析骨干已在数据精度内），辐射预算 0.764 → **0.5485**（3 seed 一致）；
   投影代理的相对 L2 在任何臂上都是 ≈1.0（对象错配，§6.2）。
   记忆 vs 无记忆在单步窗口无差异（登记为负结果；判定场景是 Phase C / 多帧累计目标）。
5. **"烧蚀率口径 ≈20×" 已定位为重启计数口径，不是单位错误**：`xtime_spi_ablation` 在每次
   code restart 归零（本 run 5 次），其末值只覆盖最后一段；rate 积分的物理总量
   = 1.8117e21 atoms，与"逐段正增量之和"比 **1.0003**，与末值比 **14.8**（§4）。

---

## 2. 对象、时序与配对（不变量）

* **被复刻对象（2026-10-07 勘误，ADR-0020 决定 1）**：双源对象——
  解析 shard 密度源 `total_imp_source → inj_source`（Gauss 点沉积进 RHS、对场显式）
  与 **marker feedback 五通道**（`projections*.h5`；marker run 的能量/辐射/电荷
  真实场源）。本报告的解析臂按 `inj_source` 语义实现、按逐 shard 烧蚀率验收；
  `--marker-chain` 的 B1.2 臂才是与记录投影同对象的监督路径。差异量化见 §5。
* **时序契约**（ADR-0019）：`S_k` 的采样窗口是 `[T_{k-1}, T_k]`，被 `u_k → u_{k+1}` 的求解消耗。
  训练配对固定为 `(u_{k-1}, u_k, p_{k-1}) → S_k`，其中 `p_{k-1}` 是窗口起点（T_{k-1}）碎片状态。
* **原生 loader 的 shard 语义（本次新增证据）**：`spi_abl_arr(jorek{k})` 的逐 shard 求和与
  `xtime_spi_ablation_rate[k]` 在正常帧上比 0.996–0.999（k=3000/5000/9000），
  说明 `jorek{k}` 里的碎片数组是**窗口 [T_{k-1}, T_k] 末端**（T_k）的状态、
  其 `spi_abl` 用 `u_{k-1}` 算出（与 `update_spi` 在解出 delta 之前被调用一致）。
  因此：
  * 训练侧把 `p_{k-1}` 取为 **上一条记录的 `shard_start`**（= `spi(jorek{k-1})`），
    `p_k` 取为当前记录的 `shard_start`（= `spi(jorek{k})`）；
  * `src/data/native_run.py` 的 `shard_start/end` 时间标注（`T_{k-1}`/`T_k`）比
    物理时间**提前一帧**——这是 loader 语义与导出规范的登记差异，见 §8 开放问题 #1。
* **推理路径无未来量**：`SourceModule.forward` 的输入只有 `(u_start, u_end, p_start)` 与
  标量/表常量；测试 `test_module_output_does_not_depend_on_future_frame`
  用"污染 jorek{k+1}"的两份 run 验证同一窗口输出逐位相同。

---

## 3. 模块结构

```
src/models/source_module.py
  ├─ NeonTables            ADAS scd96/acd96/plt96_ne + ion_ne（电价/电离势/辐射系数，缓存）
  ├─ _neon_closure         逐 shard 日冕平衡电价 Z̄、Z̄²、E_ion[J/atom]、n_e·L_z[W/atom]
  └─ SourceModule
       ├─ 确定性碎片演化  4 阶子步：位置(当前速度) → 半径侵蚀(当步烧蚀) → 火箭加速度
       ├─ 局域场采样      FemMeshSampler（双线性形函数；element search 用节点邻接 + 质心候选）
       ├─ 解析 5 通道     逐子步积分 → 逐 shard 窗口积分 → 形函数沉积 → 集总质量归一
       ├─ 单位换算        N/Z/Zeff ÷ n_norm；dE_ion/rad_sink × t_norm/E_norm
       ├─ split heads     instant: Z_eff/Z_imp/N_imp；memory: dE_ion/rad_sink（GRUCell 潜状态）
       └─ 有界乘性修正    log-space，clamp ±log(2)（默认关闭；零初始化 = 解析骨干精确复现）
src/models/fem_sampling.py
  ├─ locate              节点邻接候选 + Newton 逆映射（真实网格 846 点：全内点、22 ms）
  ├─ sample/deposit      形函数插值 / 散射组装（∫x m = Σq 精确守恒）
  └─ node_measure        集总节点体积 ∫N_i 2πR dR dZ（Gauss 积分，缓存）
```

**关键实现口径（可复核）**：

| 项 | 取值 | 依据 |
|---|---|---|
| 解析骨干 | Parks/NGS neon（`spi_abl_model=3`）+ **非局域 `max` 分支**（`NonlocalAblationTable`，按帧重建） | `parks_ablation_neon`；ADR-0015（历史） |
| 电价/辐射 | 日冕平衡（SCD/ACD）+ `ion_ne.dat` 电离势 + ADF01 PLT 冷却系数 | `models/pellet_module.f90:520-560`、`marker_spi_source.f90:590-600` |
| 单位 | `n_norm = central_density×1e20`；`E_norm = 1/(μ0(γ-1))`，`γ=5/3` | `marker_spi_source.f90:257-261` |
| 组装 | 双线性形函数 + 集总质量（预算精确守恒）；JOREK 用一致质量 Hermite 投影 | `src/physics/fem_weak.py`（B0/B2 同口径） |
| 碎片数组布局 | **显式** `shard_layout="a_prime"`（A′ loader 序）或 `"fragment_state"`（v1 序） | 两套顺序不同；混用实测产生 1e27 烧蚀爆炸 |

---

## 4. 数据门禁与烧蚀口径对账

### 4.1 严格模式（默认）

```
$ .torch-cuda-env/bin/python tools/check_export_v3.py --run-dir <phaseA run>
FAIL: ... 6 zero-dt frame(s) from code restarts: [2704, 5681, 6441, 7221, 8611, 9291] ...
      Affected pairs (dt_step=0): [2703, 5680, 6440, 7220, 8610, 9290]
exit 1
```

**结论**：去重后的 run **尚未生成**（生成侧方案 1 未落地），因此本任务的全部训练/验证
都在 `--zero-dt-policy list` 的**非最终数据、含 6 个重启零步长帧**上完成；
去重 run 到位后必须重跑默认模式并登记 `zero_dt=[]`（§8 开放问题 #0）。
（`list` 模式输出，本次复跑，产物 `.scratch/phase-b1-source/gate/list.json`：
`steps=8397 [2701..11103] windows=8397`、`dt_step 0.05/0.5/5`、`shards 846/846/846`、
`boundary codes [0,1,2,5,9,11,15,19,20,21]`、`condition n_entries=61`、
`linkage checked=200`、`restart registry=715 events=6 checked=444`；
`budget atoms=1.22461e20`（注意：这是**未做 §4.2 口径修正**的末值）。）

### 4.2 "≈20×" 差异的调查结论（`ablation_series_reconciliation`）

`models/pellet_module.f90:319-321,767-768` 核实：

| 序列 | 语义 |
|---|---|
| `xtime_spi_ablation` | 累计烧蚀原子：每步 `+= t_norm·tstep·Σ_shards(abl·species)` |
| `xtime_spi_ablation_rate` | 每步 `Σ_shards(abl·species)` [atom/s] |

本 run 实测（`jorek011104.h5` 全历史）：

| 量 | 值 |
|---|---|
| 逐 step 增量 / (rate·dt·t_norm) 中位比 | **0.99948**（8396 个有效步） |
| 累计序列重置点 | `[5680, 6440, 7220, 8610, 9290]`（5 次，恰为 restart 事件） |
| rate 积分（全历史物理总量） | **1.81172e21 atoms** |
| 逐段正增量之和 | 1.81122e21 atoms（比值 **1.00028**） |
| 累计末值 | 1.22461e20 atoms（只覆盖最后一段；比值 **14.79**） |
| `abl_history.dat` | 只到 index 9290（最后一次 restart 的快照），总 7.436e19 |

**结论**：差异**不是单位/换算错误**，而是**重启段计数口径**——
`xtime_spi_ablation`（及 `abl_history.dat`）在 code restart 处归零/截断，
末值只覆盖最后一段。物理总量应取 rate 积分或逐段正增量之和；
预算闭合验收一律用这个总量（本模块的逐窗口原子预算与 rate 序列同口径，见 §6）。
（前轮登记的 ≈20× 与本轮的 14.8× 属同一口径问题在不同帧/段上的取值。）

---

## 5. 解析 shard 沉积 vs marker feedback 的对象差异（登记，必读；2026-10-07 勘误）

### 5.1 证据

逐通道对比"解析组装（`inj_source` 语义）"与"记录的 `projections*.h5`"（523 个验证窗口，
`.scratch/phase-b1-source/metrics.json` → `arms.analytic_raw.proxy_truncation`）：

| 通道 | 最小二乘标度（train） | 中位积分比 `⟨T⟩_m/⟨A⟩_m`（train） | 空间相关 | 目标质量落在解析=0 节点上的比例 |
|---|---|---|---|---|
| dE_ion | 5.54e5 | 2.11e6 | 0.29 | （两侧量纲不同，比值无物理意义） |
| rad_sink | 2.68e3 | 1.18e4 | 0.22 | 0.48 |
| Z_eff | 11.4 | 179 | 0.15 | 0.80 |
| Z_imp | 14.5 | 195 | 0.12 | 0.85 |
| N_imp | 22.1 | 256 | **0.057** | **0.92** |

### 5.2 读法（为什么这不是"模型差"）

* `projections*.h5` 是 **marker 出通道**：`N_imp = Σ weight·timesteps` 等按**全体 marker**
  沉积后解一致质量投影系统得到的**节点场**（`marker_spi_source.f90:535-537, 626-634`），
  包含 `timesteps` 计数因子与 marker 云的支撑；模块产出的是**碎片点沉积源**
  （`inj_source` 语义：Gauss 点、进 RHS）。
* 两者的**支撑**与**标度**都不同：本模块在 846 个碎片位置做形函数沉积，投影是 marker 云的
  L2 投影 ⇒ 标度差 2–6 个数量级、空间相关 0.06–0.29、
  **48–92%** 的投影目标质量落在解析场为 0 的节点上（§5.1 表）。
* 因此：**不得**把投影当精确复刻，也**不得**用它的相对 L2 作为性能 gate；
  投影只用于登记截断误差与交叉校验（本报告 §6 的 relL2 即该登记值）。
* 若把标度强行校准到投影（`analytic_proxy_calibrated` 臂），投影 relL2 只从 0.9997 降到
  0.9734，但原子预算被同步放大 **22×**（§6 表）——说明这两个对象不能同时满足，
  这正是"proxy ≠ inj_source"的判据。

### 5.3 生成侧需求（内部源导出）

按 `report/data-export-spec-v3.md` 的 `source_object="inj_source"` 导出：

* 每流体步写出**实际进矩阵 RHS 的源贡献**（`total_imp_source → inj_source` 的逐 Gauss 点值，
  或至少按场切片的 RHS 块），字段/形状/单位/`namelist_sha256` 齐全；
* 同时给出与该源对应的**碎片状态**（`shard_start`/`shard_end` 的显式时间戳）与
  `interval_start/end`（ADR-0019 双区间）；
* 至少 1 个 run；若可给出 3 帧 fixture，可直接接入本模块的监督路径。

在内部源到位前，本模块的交付形态是"`inj_source` 语义 + 记录代理交叉校验"，
任何性能数字（Phase D）都必须用内部源重做。

---

## 6. 单步训练/验证

### 6.1 设置

| 项 | 取值 |
|---|---|
| 数据 | 原生 A′ run（`--zero-dt-policy list`，非最终）：`2705..11103`，stride 8 |
| 关键参数 | `source_object=projection`（登记代理）、`cadence=fluid_step`、`require_single_step=True` |
| 排除 | 6 个零步长 restart 索引 ±2 帧 |
| 划分 | 4 段连续 block，最后 2 段为验证集（防泄漏；`val_index_range` 见 `metrics.json`） |
| seed | 0/1/2；同窗口、同预算、同 seed 对照 |
| 损失 | 逐通道相对 L2（代理，权重 0.1，离群截断 ≤10）+ 原子预算（记录 `xtime_spi_ablation` 增量）+ 辐射预算（记录 `xtime_rad_power·tstep`，train 注册换算因子）；两个预算项的比值用批内中位数下限（`robust_relative_error`）避免近零窗口支配 |
| 优化 | Adam lr 3e-4，30 epoch，chunk 12（截断 BPTT），grad clip 1.0，修正 clamp ±log2 |

### 6.2 对照结果（验证集）

（数值见 `.scratch/phase-b1-source/metrics.json`，下表为其中关键字段。）

| 臂 | 代理 relL2（中位） | 原子预算相对误差（中位） | 原子比（中位） | 辐射预算相对误差（中位） |
|---|---|---|---|---|
| analytic-only（交付默认，无拟合） | 0.99967 | 0.0120 | 1.0002 | 0.764 |
| analytic + 投影标度校准（登记） | 0.97337 | 21.07 | 22.07 | 632.9 |
| analytic + 有界修正（记忆头，3 seed 中位） | 0.99946 | 0.0135 | 1.0050 | 0.5485 |
| analytic + 有界修正（无记忆消融，3 seed 中位） | 0.99953 | 0.0124 | 1.0006 | 0.5485 |

逐 seed（记忆臂）：loss 4.237→3.949 / 4.237→3.923 / 4.239→3.940；修正 log 尺度均值
0.51–0.55（clamp 边界 ±0.693），全部 seed 收敛且未饱和；原子比 1.004–1.007，
辐射误差 0.5485（3 seed 一致）。

（数据：523 个验证窗口 = 全 run 的后半段 `6904..11098`；完整 JSON：
`.scratch/phase-b1-source/metrics.json`。）

**读法**（不预设 performance gate；全部为登记数字）：

* 解析-only 的**原子预算**已经在数据精度内（比 1.0002、相对误差 1.2%）——与 §3 的逐 shard
  验收一致；有界修正对它几乎没有空间（0.0120 → 0.0124–0.0135），即"不需要学"；
* **辐射预算**是唯一有可测增益的通道：0.764 → **0.5485**（3 seed 一致），
  修正 log 尺度均值 0.5（仍在 ±0.693 内、未饱和），梯度有效；
  该指标的残差主要是记录序列 `xtime_rad_power·tstep` 的单位/时间口径（§4/§8 #3），
  不是本模块的数值问题；
* **记忆 vs 无记忆在单步窗口上没有可测差异**（登记为负结果）：ADR-0018 决定 8 的历史证据是
  **帧/事件级**的（R² 0.15–0.22 → 0.81–0.96），单步窗口装不下这个效应；
  记忆头的判定场景是 Phase C 闭环与多帧累计目标；
* 投影代理的 relL2 在任何臂上都是 ≈1.0（§5 的对象错配），
  而"把标度校准到投影"会把原子预算放大 22× ⇒ 该代理不能作为 gate 或性能基线；
* 训练稳定性修复（登记）：逐窗口预算序列存在近零值，原始比值会让 loss 被离群窗口支配
  （实测未修复时 loss ≈ 50 且不下降）；最终 loss 用"离群截断的相对 L2 + 中位数下限"
  （`outlier_clamped_relative_l2` / `robust_relative_error`），报告指标仍用未截断的中位相对误差。

### 6.3 物理性验收

| 验收 | 结果 | 证据 |
|---|---|---|
| 逐 shard 烧蚀率对照 `spi_abl_arr` | 比 **1.0021**（P10 0.976 / P90 1.023，523 窗口） | `shard_acceptance.abl_sum_ratio_*` |
| 碎片半径响应 | 相对误差中位 **3.8e-7** | `shard_acceptance.radius_relative_error_median` |
| 位置响应 | 窗口内位置 MAE ~1e-9 m 量级 | `test_real_frame_smoke_and_acceptance`（真实帧） |
| 总烧蚀预算闭合 | 模块原子/记录原子 中位 **1.0002** | `shard_acceptance.module_atoms_over_recorded_step_median` |
| 辐射预算 | 经注册换算因子后 0.764 → 0.549（修正臂） | §6.2、§8 #3 |
| 解析律分相位对照 | early/mid/late = 1.006 / 1.004 / 1.004（同一帧）；模块演化臂 1.005/1.000/1.002 | `.scratch/phase-b1-source/probe.json` |
| 非局域分支必要性 | 局域-only：early 0.995 → mid 0.500 → late 0.355 | 同上（`--nonlocal off` 对照） |
| 源强有界 | 逐通道非负；修正 clamp ±log2；`source ≥ 0` 与上界均有测试 | `test_correction_is_clamped_to_the_prior_bound` |
| 单调性 | 半径↑→烧蚀↑；T_e↑→烧蚀↑且末端半径↓；v_R↑→R↑（真实帧 + 合成） | `monotonicity`（真实帧）、`test_radius_and_velocity_responses_are_monotone` |
| 确定性 | 同输入同输出（逐位） | `test_numerical_determinism`、`test_zero_initialised_correction_is_identity` |
| 梯度 | 修正头与碎片输入均有限非零梯度 | `test_gradients_flow_to_heads_and_shards` |
| 时序断言 | `projections{k}` 同号对齐、`t_source_end == t_before`、`require_single_step=True` | `test_window_pairing_uses_only_past_and_current_frames` |

---

## 7. 测试与复现

```bash
# 数据门禁（当前未去重：默认模式 fail-fast，exit 1；诊断用 list）
.torch-cuda-env/bin/python tools/check_export_v3.py \
    --run-dir data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner

# 组件 + 契约测试（合成 fixture 必跑，真实帧 skip-if-absent）
.torch-cuda-env/bin/python -m pytest tests/test_source_module.py -q

# 单步训练/验证（缓存 → 校准 → 4 臂对照 → 预算/单调性）
.torch-cuda-env/bin/python tools/train_source_module_b1.py \
    --run-dir data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner \
    --out .scratch/phase-b1-source --stride 8 --epochs 30 --seeds 0,1,2 --nonlocal frame

# 解析骨干分相位 fidelity probe（同帧 vs 模块演化；局域-only 对照）
.torch-cuda-env/bin/python tools/train_source_module_b1.py --probe-only \
    --probe-stride 120 --probe-limit 60 --out .scratch/phase-b1-source --nonlocal frame

# 全量回归
.torch-cuda-env/bin/python -m pytest -q
```

### 7.1 测试结果

```
$ .torch-cuda-env/bin/python -m pytest tests/test_source_module.py -q
20 passed
$ .torch-cuda-env/bin/python -m pytest -q
266 passed, 1 skipped        # 基线 246 passed / 1 skipped（+20，无回退）
```

### 7.2 产物（未提交，路径登记）

| 路径 | 内容 |
|---|---|
| `.scratch/phase-b1-source/metrics.json` | 4 臂对照、校准、单位因子、shard 验收、单调性 |
| `.scratch/phase-b1-source/probe.json` | 解析律分相位 fidelity probe |
| `.scratch/phase-b1-source/gate/strict.json` / `list.json` | 数据门禁严格/诊断输出 |
| `.scratch/phase-b1-source/cache/` | 1046 个窗口的缓存（analytic/target/features/预算） |

复跑说明：`--out` 相同且 `cache/index.json` 存在时直接复用缓存（跳过 ~6 min 的重建）；
`--rebuild-cache` 强制重建。全流程 CPU 单进程，无需 GPU（GPU 命令未使用，故未提权）。

---

## 8. 开放问题（按风险排序）

0. **（高）去重 run 未生成**：默认模式数据门禁仍是 fail-fast；本报告全部数字都建立
   `--zero-dt-policy list`（非最终数据，含 6 个重启零步长帧）。去重 run 到位后必须
   重跑默认模式（`zero_dt=[]`）并复跑本报告的解析-only 基线。
1. **（高）native loader 的 shard 时间标注提前一帧**：`SourceModule`/训练器按物理时间使用
   `spi(jorek{k-1})` 作为窗口起点、`spi(jorek{k})` 作为窗口终点（本节 §2 有 rate 序列证据），
   而 `native_run` 把 `jorek{k}` 的数组标为 `T_{k-1}`。建议 loader 修正标注或显式登记，
   否则下游任何"按时间戳取 shard"的代码都会错一帧。
2. **（高）内部 `inj_source` 未导出**：本模块的监督只能用注册代理，proxy 差异达 2–6 个数量级
   （§5）。生成侧按 §5.3 导出后，应把 `S_used` 换成 `inj_source` 重训有界修正，
   并重做 Phase D 的性能登记。
3. **（中）辐射/电离能量序列的 SI 换算**：`xtime_E_ion_power/xtime_rad_power` 的
   JOREK→SI 因子只部分文档化（本报告用 train-only 注册因子），建议生成侧给出
   与 `E_norm/t_norm` 对应的显式换算公式，或直接导出 SI 量。
4. **（中）proxies 的 marker 支撑**：若后续要"marker 回退路线"（ADR-0016 候选 c），
   需要低分辨率 marker 诊断流（导出规范 §2.3b），当前数据只有 1e6 快照的 part 文件。
5. **（中）非局域表的 provenance**：本报告按帧重建（`src/data/nonlocal_table.py`，已向量化到
   18 ms/帧）作为登记替身；但 JOREK 侧的 `int3d_new` 实际只在 `nl_avg_Te(1)==0` 时重建一次
   （快照语义），与"按帧重建"存在系统性差异——需要生成侧确认实际实现，或直接导出
   `nl_Psi/nl_avg_Te/nl_avg_ne`（ADR-0015（历史）的 Update 已提出同一需求）。
6. **（低）`abl_history.dat` 语义**：本 run 的该文件只覆盖最后一段（index ≤ 9290），
   与 §4.2 的重启口径一致；建议生成侧在 restart 后重写完整历史或加显式 provenance。
7. **（低，流程）`gh` CLI 缺失**：issue 认领/关闭未执行（与 Phase A′ 报告一致）。

---

## 9. 改动文件清单

| 文件 | 类型 | 内容 |
|---|---|---|
| `src/models/source_module.py` | 新增 | 自生成源模块：确定性碎片演化 + 解析 5 通道骨干 + split heads + 有界修正 + FEM 组装 + 预算输出 |
| `src/models/fem_sampling.py` | 新增 | 双线性形函数定位/采样/沉积 + 集总节点测度（预算精确守恒） |
| `tools/train_source_module_b1.py` | 新增 | 单步训练/验证：缓存、blocked 划分、代理校准、4 臂对照、预算/单调性指标 |
| `src/data/budget.py` | 扩展 | `ablation_series_reconciliation()`（≈20× 口径对账），导出符号更新 |
| `src/data/nonlocal_table.py` | 微调 | `node_volume_weights()` 向量化（43× 加速；与逐元素循环版差异 ≤8e-14）——按帧重建非局域表成为可行路径 |
| `tests/test_source_module.py` | 新增 | 20 个用例：FEM 采样/组装、解析骨干、配对/时序、记忆头、clamp、确定性/梯度、预算对账、真实帧验收（skip-if-absent） |
| `report/phase-b1-source-module-2026-10-07.md` | 新增 | 本报告 |
| `report/README.md`、`report/implementation-progress.md` | 更新 | 索引 + 进展条目 |

未改：`tests/legacy/**`、`roadmap_v2.md`、`docs/adr/**`、`src/data/native_run.py`、
`src/data/step_dataset.py`、`src/models/{field_propagator,implicit_stencil,source_head}.py`、
旧 manifest/checkpoint/评测 JSON；`git` 未使用（`.git` 只读）。


---

## 10. DoD 对照

| DoD | 状态 | 证据 |
|---|---|---|
| 数据门禁通过（去重后 `zero_dt=[]`；未到位则明确标注非最终） | ⚠️ 非最终 | 严格模式 exit 1（6 个零步长帧，§4.1）；`list` 模式 `steps=8397` 全量跑通，报告与产物均标注"非最终数据" |
| 源模块 + 解析骨干 + 记忆头合入 | ✅ | `src/models/source_module.py`、`src/models/fem_sampling.py`（§3） |
| 单步训练/验证可复现 | ✅ | `tools/train_source_module_b1.py` + `metrics.json`（§6/§7） |
| 预算闭合 | ✅ 登记 | 原子比 1.0002、半径 3.8e-7、辐射 0.764→0.549（§6.2/§6.3） |
| 单调性验收 | ✅ | 真实帧扰动 + 合成测试（§6.3） |
| 解析-only 对照完成 | ✅ | 4 臂同窗口/同预算/同 seed（§6.2） |
| proxy vs `inj_source` 差异登记 | ✅ | §5（标度/相关/支撑 + 生成侧需求 §5.3） |
| 烧蚀口径 ≈20× 结论 | ✅ | §4.2（重启段计数，非单位错误） |
| 全量 pytest ≥246 无回退 | ✅ | 266 passed / 1 skipped（§7.1） |
| 报告 + 进度条目 | ✅ | 本报告；`report/implementation-progress.md` 2026-10-07 条目 |
| 非目标未触碰 | ✅ | 无闭环、无场推进器/边界改动、无 GNN/谱分支；legacy/roadmap/ADR 未改 |
