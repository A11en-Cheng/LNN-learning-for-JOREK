# 数据契约 v3（2026-10-04 冻结）

**状态**：current / frozen（2026-10-07 按 ADR-0020 修订源对象、restart 与预算分段）。
依据只有 [ADR-0018](../docs/adr/0018-architecture-first-and-physics-structured-build.md)、
[ADR-0019](../docs/adr/0019-recorded-output-timing-contract.md)、
[ADR-0020](../docs/adr/0020-marker-source-contract.md)、`roadmap_v2.md`
Phase A 与 [model502-schema-and-source-protocol.md](model502-schema-and-source-protocol.md) §3.1a。
ADR-0001–0017 是 v1 历史：引用须标注"（历史）"，不得作为 gate、前置或对照。

本契约冻结三件事：**S 对象语义**、**时间语义与粒度**、**ADR-0019 配对与 restart 边界规则**。
可直接交给数据生成方的导出字段级规范见 [data-export-spec-v3.md](data-export-spec-v3.md)；
代码实现见 `src/data/output_contract.py` 与 `src/data/paired_dataset.py`。

---

## 1. S 对象语义

### 1.1 双源对象（ADR-0020 决定 1）

marker run（生产 namelist `use_marker = .t.`）里实际驱动场的源拆成两条：

1. **解析 shard 密度源** `total_imp_source → inj_source`：Gauss 点逐点沉积到 FEM
   网格（每粒子按其所在单元的基函数加权）；只进入求解 RHS，**对场是显式**
   （LHS 不含源项）。在 model502 里它只负责杂质密度源
   （`mod_elt_matrix.f90:1219`，`rhs_ij_5 :1542`，`rhs_ij_8 :1717`）。
2. **marker feedback 五通道** `(dE_ion, rad_sink, Z_eff, Z_imp, N_imp)`：由
   `marker_spi_source.f90` 逐粒子沉积、按 `rhs_gather_time` 平均写出
   `projections*.h5`，被 `mod_elt_matrix.f90:989-1003,1079,1129,1728`
   直接消费 —— 这是能量/辐射/电荷通道的**真实场源**（不是代理）。

两条不得混用、不得静默互换；每条导出/记录必须登记 `source_object` provenance。
对 marker run，B1 的监督目标就是**记录投影本身**（ADR-0020 决定 1/2）。

### 1.2 `projections*.h5` 的定位

`projections*.h5` 是 **marker 投影/导出通道**（`[dE_ion, rad_sink, Z_eff, Z_imp, N_imp]`
五通道，逐节点值）：

* 在 marker run 里它就是能量/辐射/电荷通道的真实场源（见 §1.1），
  B1/B2/C 的目标/输入对象是同一套五通道（共享 S 接口，ADR-0020 决定 4）；
* 与解析 shard 沉积（`inj_source` 语义）**不是同一离散对象**：支撑、标度、
  marker 云分布不同（B1 报告 §5：48–92% 投影质量落在解析沉积为 0 的节点上），
  交叉校验时必须登记该对象差异与截断误差；
* 推理路径仍不得直接读取**未来帧**的记录量：闭环里 S 由 B1 自生成
  （marker 链 + 有界残差），记录投影只作训练目标与交叉校验。

每个导出文件/记录必须自描述 `source_object` provenance，取值至少区分
`inj_source`（内部源）与 `projection`（marker 投影）。两者不得混用、不得静默互换。

### 1.3 增广状态 `(u, p_shard, h_marker)`（ADR-0020 决定 3）

主架构里的确定性碎片状态 `p_shard` = **碎片（`type_SPI`）状态**（位置、速度、半径、
烧蚀率、species 等，即 `jorek*.h5` 的 `spi_*_arr`，≤~846 个碎片）；每步导出
`shard_start`/`shard_end`。**完整 marker 群体（约 1e6 宏粒子，含电荷态分布 `P_imp`）
不是主路径输入**：全粒子回放不可行（172 MB/帧、每 10 步一帧、单 run ~144 GB；
ADR-0020 背景 §2），其信息由**降阶 marker 链**与跨 step 记忆 `h_marker`
（GRU 潜状态 + 云摘要）携带。B1 的确定性主项 = shard 烧蚀/诞生 → ADAS 电荷态/
电离能/辐射 → 场向平流 + 垂直扩散 → FEM 沉积 + gather-time 平均；
NN 只学有界残差。`part*.h5`（每 10 步快照：`i_life`/`P_imp`/`weight`/`x`/`v`）
是降阶输运的标定源，不是训练输入。

---

## 2. 时间语义（ADR-0019）

### 2.1 定义

设 `T_n` = 第 n 个 JOREK 流体步的时间（`macroscopic_vars.dat` 第 n 条 `@times` 记录），
每一步为**一个流体步**（可含 JOREK 内部子步，但记录粒度以流体步为单位）：

| 记号 | 定义 | 用途 |
|---|---|---|
| `dt_source(k)` | `T_k − T_{k−1}` | `S_k` 采样窗口 `[T_{k−1}, T_k]` 的长度 |
| `dt_step(k)` | `T_{k+1} − T_k` | 场推进 `u_k → u_{k+1}` 的目标区间长度 |

### 2.2 文件级契约

| 文件 | 物理时间 | 单位 | 角色 |
|---|---|---|---|
| `jorek{n}.h5` | `T_n` | JOREK 时间单位（`t_now`） | 九场状态 `u_n` |
| `projections{n}.h5` | 窗口 `[T_{n−1}, T_n]` 的时间平均，内部时间 `T_n` | 物理秒（`t_now`，`÷ t_norm` 得 JOREK 单位） | 源 `S_n`，被第 `n+1` 步求解消耗 ⇒ 驱动 `u_n → u_{n+1}` |
| `part{n}.h5` | `T_{n−1}` | 物理秒（`/time`，`÷ t_norm` 得 JOREK 单位） | `S_n` 采样窗口的**起点**粒子状态 `p_{n−1}` |

`t_norm` 从同帧/配对场帧读取（单位：秒 / JOREK 时间单位），换算
`t_seconds = t_jorek × t_norm`。

### 2.3 唯一配对规则

```
(u_k, p_{k−1} = part{k}, S_k = projections{k}) → u_{k+1}
```

同一窗口内碎片演化与源沉积同时进行：`p_n = FragEvolve(p_{n−1}, u_n)`，
随后 `u_{n+1} = Solve(u_n, S_n)`。

**禁止** `source[:, k+1]` 式配对（一步未来的源；v1 管线的泄漏错误，ADR-0019 决定 2）。

### 2.4 断言（fail-fast，代码在 `src/data/output_contract.py`）

1. `jorek` 的 `t_now` 严格递增；`dt <= 0` 只允许出现在**登记过的 restart 事件**
   （见 §4.3），未登记一律 fail-fast；
2. 非 restart 帧：`projections{k}.t_now ÷ t_norm == T_k`（数值容差 `1e-4` JOREK 单位）；
3. `part{k}.time ÷ t_norm == T_k − dt_source(k)`；
4. 配对缺失（同一窗口对象不齐）直接报错；覆盖窗口边界的单边帧登记后跳过；
5. restart 边界帧按 §4.3 登记处理（跳过跨界配对与滞后投影，事件帧保留），
   偏移必须显式且等于 `-dt_source`。

### 2.5 冻结依据（2026-10-04 全量核实）

`scan_1d22_ang12_massloss_neon` 620/620 个配对帧：

* `jorek` 的 `t_now` 与 `macroscopic_vars.dat` 第 n 步一致；
* `part{k}.time ÷ t_norm == T_{k−1}`（620/620，容差 1e-4）；
* `projections{k}.t_now ÷ t_norm == T_k`（614/620）；其余 6 帧恰为登记
  restart 边界（§4），偏移量恰为一个流体步 `dt_source`。

---

## 3. 单位、索引与命名

### 3.1 单位

| 量 | 单位 | 说明 |
|---|---|---|
| 空间坐标 / 几何 | 几何米 | `x` 为 `[2, 4, harmonics, nodes]` 的 Bezier 控制点；节点索引全局一致 |
| 状态时间 | JOREK 时间单位 | `jorek.t_now`、`macroscopic_vars @times` |
| 标记/时间戳输出 | 物理秒 | `projections.t_now`、`part./time`；换算用 `t_norm` |
| 状态场 | 见 schema §1 | 9 通道顺序固定：`psi, Phi, Z_j, omega, rho, T_i, v_par, rho_imp, T_e` |
| 源（投影）通道 | 见 schema §3.1 | 5 通道顺序固定：`dE_ion, rad_sink, Z_eff, Z_imp, N_imp` |

### 3.2 索引与命名

* 文件编号 `n` = JOREK 流体步号（与 `macroscopic_vars.dat` 行号一致）；
* 现存记录命名：`jorek{n:06d}.h5`、`projections{n:06d}.h5`、`part{n:05d}.h5`；
* 新导出命名与字段见 [data-export-spec-v3.md](data-export-spec-v3.md) §2；
* 索引与时间戳不得用文件名隐式约定；每个导出记录必须自带
  `interval_start`/`interval_end`/`step_index`（ADR-0019 决定 3）。

### 3.3 时间戳规则

1. 同一数据集内只允许一种时间单位，并在 metadata 中写明；
2. 所有时间戳必须可由 `t_norm` 换算到 JOREK 单位与物理秒；
3. 记录的时间是**区间端点**，不允许用区间平均值代替端点（ADR-0019 决定 3）。

---

## 4. 粒度与 restart 边界

### 4.1 目标粒度：每流体步

每个流体步自描述导出：

```
(interval_start, interval_end, u_before, S_used, u_after, shard_start, shard_end)
```

按 ADR-0019 的错步语义，记录 `n` 同时携带两个显式区间（**禁止只给一个含糊 interval**）：

* **源采样窗口** `[T_{n−1}, T_n]`：`S_used = S_n` 的时间平均窗口，`shard_start`/`shard_end`
  分别是 `T_{n−1}`/`T_n` 的碎片状态（完整 marker 为可选诊断流，不参与主路径）；
* **场推进区间** `[T_n, T_{n+1}]`：`u_before = u_n`、`u_after = u_{n+1}`，
  由 `S_used` 驱动（`interval_start/end` 指这一段）。

字段语义、形状、单位见导出规范 §2。细粒度不可得时退化为**登记子步组**
（组内步数、组两端时间显式登记），禁止帧间插值伪造真值（roadmap_v2 Phase A 回退）。

### 4.2 旧 EHL-2 的记录粒度

现存 EHL-2 的 dump 间隔为 10（细段）或 50（粗段）流体步，而 `S_k` 只覆盖
`[T_{k−1}, T_k]` 这**一个流体步**。因此帧级旧数据只在区间两端有真值：
只监督/评估区间两端，不做区间内部插值评估（roadmap_v2 §1.2 #14）。

### 4.3 restart 边界帧（登记事件，跳过跨界与滞后投影；ADR-0019 补充/ADR-0020 决定 5）

生产 run 的代码重启产生 `dt=0` 重复帧；生成侧决定**不重导**（2026-10-07），
loader 默认 `zero_dt_policy="registered"`。登记表见 `data/restart_events.json`
（或 `<run>/restart_events.json`、sidecar `restart_events`、CLI `--restart-events`）。

| run | 登记事件 n（T_n == T_{n−1}） | 跨界配对/滞后投影 n−1 |
|---|---|---|
| `phaseA_dataset_1d22_ang15_massloss_neon_v400` | 2702, 4111, 5871, 6871, 8371, 9421, 10421, 11421, 11711 | 2701, 4110, 5870, 6870, 8370, 9420, 10420, 11420, 11710 |
| `phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner` | 2704, 5681, 6441, 7221, 8611, 9291 | 2703, 5680, 6440, 7220, 8610, 9290 |
| `scan_1d22_ang12_massloss_neon`（v1 记录 run） | 3550, 4150, 4950, 5330, 6600, 7350（历史登记） | — |
| `scan_1d22_ang15_5_massloss_neon_v400`（v1 记录 run） | 5050（历史登记） | — |

规则：

1. 事件 n 的 `projections{n−1}` 被提前一个流体步写出：
   `t_projection = T_{n−2}`（偏移恰为 `−dt_source`）；`projections{n}` 与
   `jorek{n}` 对齐；`part{k}` 始终满足 `T_{k−1}`（2026-10-04/07 全量核实）；
2. **跳过**跨界配对 `(n−1 → n)`（零步长）与滞后投影 `projections{n−1}`
   （记录 n−1 整体丢弃）；
3. **保留**事件帧 n：`S_n = projections{n}` 对齐，场步 `n → n+1` 合法；
   其源窗口 `[T_{n−1}, T_n]` 是**退化窗口**（长度 0），登记
   `restart_degenerate_source_window`，**不得**伪造/插值；
4. **未登记**的 `dt<=0`、或登记事件上偏移不等于 `−dt_source`，都是硬错误
   （fail-fast），不得静默修剪；`zero_dt_policy="error"` 保留给新生成数据的
   严格去重模式（每 restart 只写一帧、`projections` 与 `jorek` 对齐）；
5. 新导出必须让 restart 帧同样满足 `interval_start`/`interval_end` 元数据，
   不允许再出现隐式偏移。

### 4.4 原生布局的推导 interval、freeboundary/axis 归一化（ADR-0020 决定 7）

原生逐步 run（`jorek*/projections*/part*`）没有自描述 interval 文件，
`interval_start/end`、`source_interval_start/end` 由同帧 `t_now` 与
`macroscopic_vars.dat` 时间表**推导并交叉校验**（容差 `1e-4` JOREK 单位）；
loader 在 manifest/provenance 中登记 `interval_source`。规则：

1. `interval_start = T_k`（`jorek{k}.t_now`）、`interval_end = T_{k+1}`、
   `source_interval = [T_{k−1}, T_k]`，`S_used = projections{k}`；
   禁止用文件名编号隐式约定时序（文件名必须与 `t_now` 一致）；
2. **shard 时间语义（原生）**：`jorek{k}` 的 `spi_*_arr` 是 **T_k** 的状态
   （实测 `spi_abl_arr` 逐 shard 和 == `xtime_spi_ablation_rate[k]`）；
   loader 的 `shard_start/end` 标注真实帧时间 `T_k`/`T_{k+1}`；
   源窗口起点 `p_{k−1}` 是**上一条记录**的 `shard_start`（`spi(jorek{k−1})`，T_{k−1}）。
   代码以断言强制该配对（`tests/test_step_dataset.py`、B1 训练器）；
3. `freeboundary` 原生是**全局 `'F'/'T'` 标志**，不是逐节点 int8；
   生成侧若要节点级数组，按同一全局标志展开为逐节点 `int8`（0/1），
   并在 manifest 登记 `freeboundary_normalization="global_flag_expanded"`；
   原生 loader 读取该标志并在 provenance 中登记（不伪造节点级语义）；
4. `axis_node` 在当前 2D run 不存在（`axis_node=F`/`treat_axis=F`，B0 §6.7）；
   契约允许缺省，缺省时不得推断磁轴节点（B0 的轴处理只由合成 fan mesh 验证）；
5. 每步 `jorek*.h5` 携带长度 = `index_now` 的 `xtime_*` O(N²) 历史；
   逐 step 读取必须用 tail（`dataset[-1]`），见 `src/data/budget.py`。

---

## 5. 验收检查（契约级）

* 时间戳严格递增（登记 restart 事件的 `dt=0` 例外，见 §4.3）、无缺口（gap 需显式登记）；
* 同一区间的 `u_before/S_used/u_after/shard_start/shard_end` 来自同一 run、同一网格、同一索引；
* `boundary` 代码 + `freeboundary` 与场帧同源、同索引（供 stencil 边界执行，ADR-0018 决定 11）；
* condition 条目带分组标签与组语义（浅层 FiLM 用，ADR-0018 决定 12）；
* 预算（原子/能量/辐射）可用导出的同一 FEM 基函数核算，且**按 restart 分段**
  （`xtime_spi_ablation` 在每次 code restart 归零；见 ADR-0020 决定 7 与
  `src/data/budget.py`）；
* S 对象满足共享五通道接口（顺序/布局/单位/gather-time 平均，
  `src/models/source_contract.py`，ADR-0020 决定 4）；
* 契约测试全绿（`tests/test_output_contract.py`、`tests/test_paired_dataset.py`、
  `tests/test_step_dataset.py`、`tests/test_budget.py`）。

## 6. 变更控制

契约 v3 已冻结。字段/语义变更必须走新 ADR 或 roadmap_v2 修订，并在本文与导出规范
同步更新版本号；禁止用旧 manifest/checkpoint/评测 JSON 的语义回填本契约
（隔离规则见 [docs/adr/README.md](../docs/adr/README.md) 与
[v2-doc-isolation-and-dirty-data-audit-2026-10-04.md](v2-doc-isolation-and-dirty-data-audit-2026-10-04.md)）。
