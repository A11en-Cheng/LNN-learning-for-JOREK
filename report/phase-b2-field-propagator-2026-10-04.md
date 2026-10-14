# Phase B2 交付报告：九场推进器（隐式 stencil + LTC + 浅层 FiLM）（2026-10-04）

> **2026-10-04 B2.1 后续**：本报告的开放问题 #2（准稳态）、#3（系数链）、#4（逐项权重）、
> #5（通道映射）与 #6 的主要部分已在
> `report/phase-b2.1-quasi-steady-2026-10-04.md` 收口；后续以 B2.1 报告为准。

依据：ADR-0018 决定 5/6/7/11/12/13、ADR-0019（配对/时间语义）、`roadmap_v2.md`
§4 Phase B2、§5 第 4/5/10/11 条、§6 B2 行、§7"显式 stencil 稳定 cap / FiLM 深层注入 /
边界 BC 缺失"风险行、`docs/jorek-normalization.md`（混合单位制与逐项稳定界）、
`report/data-export-spec-v3.md`（condition 分组词表）、B0 报告
`report/phase-b0-stencil-boundary-2026-10-04.md`。

**未做**：训练、长窗口/闭环 rollout、A′ 数据 loader、GNN/谱分支、全集 concat、
"无 condition"对照臂、legacy 测试改动、roadmap/ADR 改动、数据删除。
`gh` 在本环境缺失（B0 已记录），未走 issue 流程。

## 0. 一句话结论

九场同一契约的 B2 推进器组件已落地：**stencil 只作空间算子**，物理 `dt=0.05`
（JOREK 时间）下用"逐通道谱主元的对称 Richardson 阻尼迭代"求解
`(I + dt·Σ_t α_t A_t/m) x = rhs`，前向 **无条件收缩**（谱半径 < 1，无瞬态放大）、
无 NaN、逐层 BC 复位、可微；逐项弱形式算子已精确装配并给出 **λ_max 与
`dt=2/(αλ)` 表**（η≈2.76e3 宽松，D_perp≈7.96e-2 临界，visco/visco_par/D_par/ZK∥
在 0.05 下显式不稳定）；FiLM 只在最浅编码层、恒等初始化，深层 stencil 对 condition
的 Jacobian **结构性为 0**，分组置换后局部响应形状不变（合成 cosine 1.0，真实帧
1.0000000000000002）。全量 pytest **177 passed**（基线 144，无回退）。

## 1. 改动文件

| 文件 | 内容 |
|---|---|
| `src/models/implicit_stencil.py` | 新增：逐项弱形式算子装配（P1 刚度 + P1 平行刚度）、精确 `λ_max`（稀疏广义特征值，替代 B0 的 A/m 代理）、`ImplicitStencilLayer`（对称主元阻尼迭代，逐通道/逐项可学习 α 与逐通道阻尼）、`ExplicitStencilLayer`（仅作安全测试对照） |
| `src/models/field_propagator.py` | 新增：九场 B2 栈（浅层编码 + 浅层 FiLM/AdaLN + LTC 闭式时间积分 + n 层隐式 stencil + 解码 + 每步 BC 复位）、boundary 码 fail-fast/登记、namelist `use_simple_bnd_types`/`no_mach1_bc` 取值链、参数量/条件深度报告、`build_propagator_from_frame` |
| `src/models/stencil_conv.py` | **缺陷修复**：element matmul 的 `[4, E]` 连接展平顺序（详见 §7）；`FemStencilGeometry` 增加 `pos` 字段（B2 逐项算子装配所需，向后兼容） |
| `tests/stencil_fixtures.py` | 新增合成 fixture：细网格（λ~1e6）、已确认边界码集合、Dirichlet-only 网格、均匀 b̂ |
| `tests/test_implicit_stencil.py` | 新增 16 个用例：逐项算子/λ_max/稳定界、显式 vs 隐式、收缩性/精确解对照、多层 BC/NaN、确定性与梯度、真实帧 λ/rollout（skip-if-absent） |
| `tests/test_field_propagator.py` | 新增 17 个用例：FiLM 恒等初始化/作用深度/分组置换、深层 Jacobian=0、物理 dt 多层、确定性/梯度/参数量、LTC 组件、boundary 码策略与 namelist、离散关系、真实帧 smoke |
| `tests/README.md` | 测试分组表新增 B2 两行 |
| `report/implementation-progress.md` | 本条目摘要 |

## 2. 模块结构

```
u  ──► ShallowEncoder(9→hidden, 恒等初始化)
      ──► FiLMHead(全局标量 γ(c)·h + β(c)，恒等初始化，仅此一处消费 condition)
      ──► LTCFieldStep(每通道 rate=softplus>0；闭式 h←decay·h+(1-decay)·target(features))
      ──► [ImplicitStencilLayer] × n_layers
             每层：边界输入 sanitize → rhs = x + dt·(显式项) + dt·forcing
                   → 求解 (I + dt·Σ_t α_t A_t/m) x_new = rhs（k 次阻尼迭代）
                   → 边界行不参与内部写入 → jb.apply_boundary_conditions 复位
      ──► ShallowDecoder(hidden→9, 恒等初始化) ──► 残差形式 out = u + decoder(S(h) − h)
```

要点：

- **九场同一契约**：隐藏宽度默认 = 9；九个通道同一堆栈，不做 7+2 拆分（决定 7）。
- **`dt` 显式**：`FieldPropagator.set_dt()` / `forward(dt=...)`，单位 JOREK 时间；
  LTC 的 `dt_reference` 与 stencil 层共同使用（默认 0.05，匹配 B2 验收口径）。
- **每层 BC 复位**：层内 `sanitize_boundary_inputs` → 边界行保持前一层输入 →
  `apply_boundary_conditions(reference=前一层输入, dt=dt)`；解码后再做一次输出级复位。
- **condition 接口**：`film_parameters(condition, groups)`、`encode(...)`；
  `condition_depth_report()` 给出"深层不含 condition 参数"的结构性证明。
- **源接口**：`forward(source=...)` 以显式 RHS 进入第 1 层（`source_proj` 零初始化，
  与 JOREK "源只进 RHS、对场显式"一致；默认对初始动力学无影响）。

## 3. 时间积分格式

**stencil 不是显式传播子**（决定 13）。设 `A_t` 为逐项弱形式算子（PSD），
`S_t = A_t/m`，`m` 为集总质量。B2 每层求解

```
(I + dt · Σ_t α_{t,c} · S_t) x_new = rhs                （α_{t,c}：逐通道/逐项系数）
rhs = x + dt·(η 显式项) + dt·forcing
```

迭代（`preconditioner="spectral"`，默认）：

```
D_c = 1 + dt · Σ_t α_{t,c} · λ_max(A_t)                 （逐通道标量主元）
x ← x + ω_c · (rhs − (I + dt·Σ_t α S_t) x) / D_c        （ω_c ∈ (0,1) 可学习）
```

在质量内积下 `M = I + dt·Σ_t α S_t` 与 `I` 对称，谱 ⊂ `[1, D_c]`，故迭代矩阵
`J = I − ωD⁻¹M` 对称、谱 ⊂ `[1−ω, 1−ω/D_c]`：**任意 dt、任意正 α、任意 ω ≤ 1 都收缩**，
不存在非正规瞬态放大。这一点是必需的：平行算子 `A_par` 有**正的非对角元**，
对角/Jacobi 预条件虽谱半径 < 1，但矩阵范数 > 1，会指数放大（实测 8 次迭代 ×1600）。
`preconditioner="jacobi"` 仅保留给各向同性场景与对照测试。

刚性项（visco/visco_par/D_par/ZK_i_par/ZK_e_par + D_perp）走隐式；只有 η 留在显式
（其显式界 2.76e3 ≫ 0.05）。显式 cap 仅作对照：B0 的 cap 是 α=1 几何界，带上物理 α
后一步放大 7.4e5 倍（组件测试钉死）；换成含 α 的 cap 后有效步长为
`dt_eff/dt = 1.5e-12`，物理时间不可表示。

**LTC 组件**：`h_new = decay·h + (1−decay)·target(features)`，
`decay = exp(−softplus(rate_raw)·dt) ∈ (0,1]`，系数在步内冻结、target 只依赖冻结的
features ⇒ `dt=0` 严格恒等、拆分 `dt1+dt2` 机器精度精确、任意 dt 不放大；
target 恒等初始化（训练后才提供驱动）。

## 4. 逐项 λ_max 与稳定界（替代 B0 的 A/m 代理）

**装配**：各向同性项复用 B0 的 P1 刚度核（已验证与 `fem_weak.build_p1_laplacian`
逐元素相等）；平行项按同源三角化装配
`∫(b̂·∇N_i)(b̂·∇N_j)`，`b̂` 由本帧 ψ 梯度 + F0 给出（`B=[ψ_Z,−ψ_R,F0]/R`）。
**精确谱**：`λ_max(diag(m)⁻¹A_t)` = 稀疏广义特征值（内部节点子集；小网格用稠密
广义特征值，测试互校到 1e-12）。

真实帧 `scan_1d22_ang15_5_massloss_neon_v400/jorek003000.h5`（4125 节点，198 边界节点，
内部 3927）；α 用 `docs/jorek-normalization.md` §3.2 表（与 H5 存值一致）：

| 项 | α（JOREK 单位） | λ_max 精确 | dt=2/(α·λ_max) | dt=0.05 显式稳定 | B0 行和代理 λ |
|---|---:|---:|---:|---|---:|
| η（磁扩散） | 1.443e-10 | 5.0230e6 | 2.759e3 | ✅ | 5.7527e6 |
| D_perp | 5.0e-6 | 5.0230e6 | 7.963e-2 | ✅（临界） | — |
| visco | 2.4e-3 | 5.0230e6 | 1.659e-4 | ❌ | — |
| visco_par | 5.0e-4 | 5.0230e6 | 7.963e-4 | ❌ | — |
| D_par | 5.0e-2 | 3.7604e6 | 1.064e-5 | ❌ | 4.3363e6 |
| ZK_i_par | 1.67e5 | 3.7604e6 | 3.185e-12 | ❌（刚性） | — |
| ZK_e_par | 4.10e6 | 3.7604e6 | 1.297e-13 | ❌（刚性） | 4.3363e6 |

结论与 `docs/jorek-normalization.md` §3.2 一致：**η 显式宽松；D_perp 临界；
visco/visco_par/D_par/ZK∥ 在物理 dt=0.05 下必须隐式**。精确 λ_max 比 B0 的行和代理小
13%（各向同性）/13%（平行），不改变任何结论，但把表从"代理"升级为"本离散算子的精确谱"。

**生成侧/取值链差异（登记）**：该 run 目录的 `jorek_namelist_2d` 里
`visco=1e-5, visco_par=5e-6, d_par=1e-2, zk_par_max=5e4`，而 H5 存
`visco=2.4e-3, visco_par=5e-4`（与文档 §3.2 一致）；`d_perp` 未在 namelist 出现。
参数取值链（H5 → namelist → JOREK 默认）优先 H5，`TransportCoefficients.from_h5_and_namelist`
已复用它并逐项登记 provenance。若按 namelist 文件值，visco 的界变为 3.98e-2（仍 <0.05）、
visco_par 变为 7.96e-2（临界区）；ZK∥/D_par 结论不变。**待生成侧确认哪套值是该 run 的有效值。**

> **2026-10-04 复核补充**：该 run 目录有 6 个 namelist 文件，H5 的
> `visco/visco_par/eta` 与 **`jorek_namelist_spitest`** 完全一致
> （2.4e-3 / 5e-4 / 1.443e-10）；`jorek_namelist_2d`（1e-5/5e-6/1e-2/5e4）等是
> 旧文件，不是该 run 的有效输入。因此 α 表以 **H5 + `jorek_namelist_spitest`**
> 为准（`D_par=5e-2`、`ZK_par_max=1e4`）；其余 namelist 不得用于稳定性计算。

## 5. 隐式稳定性与多层实测

| 检验 | 合成细网格 | 真实帧 jorek003000 |
|---|---|---|
| 单层收缩因子 max（逐通道） | < 1（实测 0.999999999998） | < 1（同量级） |
| 残差历史（对称迭代，80 步） | 2.911e11 → 9.677e8（单调下降） | — |
| 40 层/8 层 rollout | 有界（≤1.5×初值）、无 NaN | 有界（max 9.37，8 层 Δmax 0.035，rel 3.8e-3） |
| Dirichlet 行 40/8 层后 | `torch.equal` 初值 | `torch.equal` 初值 |
| BC 一致性报告 | `dirichlet_ok`、`sonic_ok` | `sonic_ok` |
| 边界 NaN → 3 层 | 全网格有限 | 全网格有限 |
| 内部 NaN → 边界 | 边界不被污染 | — |
| 确定性 | `torch.equal` | — |
| 梯度 | 逐参数有限 | — |
| 中度 α（1e-5）与稠密精确解 | 逐通道 1e-5 容差一致 | — |

注：ZK∥ 的隐式迭代在 dt=0.05 下收缩率 ≈ 1−1e-12（前向稳定但该极限基本冻结）；
见 §10 第 2 条。

## 6. 边界（复用 B0，不重写）

- 每层顺序：`sanitize_boundary_inputs` → 隐式求解（Dirichlet 行保持参考值） →
  边界行不参与内部写入（`torch.where(bc_rows, values, out)`）→
  `jb.apply_boundary_conditions`（Dirichlet hold、Neumann 单侧、Mach-1 鞘层、鞘层通量汇）。
- **fail-fast**：未登记码（不在 B0 表内）直接 `ValueError`；B0 §6 待确认码
  `{3,4,9,12,19,20,21}` 需显式 `acknowledge_pending_codes=True`，并把
  `pending_codes`/`code_counts` 登记进 `boundary_report`。真实帧实测登记：
  `{0:3927, 1:32, 2:94, 5:10, 9:6, 11:32, 15:10, 19:4, 20:4, 21:6}`，
  其中 pending = `[9,19,20,21]`（B2 已按 B0 映射执行并登记）。
- **namelist 取值链**：`read_namelist_bc_flags` 读 `use_simple_bnd_types`、`no_mach1_bc`、
  `bc_natural_open`、`freeboundary`、`mach_one_bnd_integral` 并登记 provenance；
  `use_simple_bnd_types=.true.` 时按 B0 §6.4 折叠（11→1、15→5、19→9、12→4、20/21→9）。
  真实帧 namelist：`use_simple_bnd_types` 未设（默认 F）、`no_mach1_bc` 未设（F）、
  `bc_natural_open=.true.`、`freeboundary=.f.`。
- 磁轴/度量奇异沿用 B0：共点节点合并、退化三角形丢弃、`1/R` clamp。

## 7. 发现的 B0 缺陷（已修 + 回归测试）

`src/models/stencil_conv.py` 的 element matmul 路径把 `[4, E]` 连接矩阵按行主序展平
（`vertex_cluster.reshape(-1)`），得到的是"连续 4 个单元的同一条边"，而不是
"同一单元的 4 个角点"，与 `message.reshape(E, 4, C)` 的 element-major 语义不匹配。
B0 的测试没发现它，因为守恒与常场保持在该置换下不变，且线性场检验用的是稀疏矩阵路径。
B2 的隐式层直接用 element matmul，稠密对照立即暴露（差异 ~100%）。
修复：`vertex_cluster.transpose(0,1).reshape(-1)`（B0 的 `StencilConv` 与 B2 的
implicit/explicit 层统一），并新增回归测试
`test_element_matmul_path_matches_the_assembled_weak_form_operator`（element 路径 vs
稀疏 `-Ax/m`，1e-8 容差）。B0 原 16 个测试修复后仍全绿。

## 8. FiLM 两层验收（结构性 + 行为性）

- **恒等初始化**：γ=1、β=0 精确；`h → γh+β` 在初始化时是恒等（测试断言）。
- **作用深度**：condition 只在 `FieldPropagator.encode` 内被消费（`FiLMHead`）；
  深层 stencil 层的 `forward(hidden, dt, forcing)` 不接受 condition；
  `condition_depth_report()` 证明 Film 参数与深层参数无共享。
- **结构性（深层 Jacobian≈0）**：对固定深路径探针，`∂(深层 stencil 输出)/∂(condition)`
  在 autograd 上**没有边**（`allow_unused=True` 全为 None），两次前向逐位相等；
  深层的局部 Jacobian（对输入）是条件无关的仿射、稀疏（非零率 < 20%）算子。
- **行为性**：同一局部探针（单节点 ε 扰动，取 FEM 1-ring 响应、去掉被扰动节点自身），
  打乱 `pellet_mass` 组后：合成网格 cosine = 1.0（增益比 0.33）；真实帧
  jorek003000 cosine = 1.0000000000000002（增益比 0.9908，γ 从 1.4698 变为 1.4563）。
  即只改变整体驱动/耗散强度，响应形状不变。
- **group 策略**：`pellet_*`/`background_plasma`/`boundary_flags` 允许 FiLM；
  `excluded_transport`/`excluded_numeric` 一旦被标为 eligible 直接报错；未知组报错。
  condition 输入经 symlog + 软压缩（有界）后进入 FiLM——原始量级跨 32 个数量级
  （n_spi 1e22 … eta 1e-10），不压缩会让 FiLM 饱和、分组置换不可辨。
- 默认是**全局标量** γ/β；`film_per_channel=True` 保留但未做形状验收（§10）。

## 9. 参数量、确定性与离散关系

- 参数量（真实帧、hidden=9、2 层、含 6 维 condition）：**795**；
  分模块 encoder 90 / decoder 90 / LTC 91 / FiLM 290 / stencil 2×72 / source 90；
  无 condition 的帧级栈为 505。参数全部显式登记（`parameter_report()`）。
- 确定性：同输入同输出（`torch.equal`），无随机初始化路径（stencil/LTC/FiLM 均确定性）；
  FiLM 权重扰动用固定 seed。
- 梯度：全模型反向一次，除零初始化的 `source_proj`（无梯度路径）外逐参数有限。
- 离散关系（复用 B0 口径）：常量被弱形式 Laplacian 精确零化（<1e-8）、线性场在内部行
  零化（<1e-8）、`delta_star/delta_plus` 与代码契约逐元素一致；真实帧的
  `Z_j≈Δ*ψ`、`ω≈Δ+Φ` 由 B0 测试继续覆盖（corr 0.992/0.997）。

## 10. 测试命令与结果

```bash
.torch-cuda-env/bin/python -m pytest tests/test_implicit_stencil.py tests/test_field_propagator.py -q
# 33 passed in ~8 s
.torch-cuda-env/bin/python -m pytest tests/test_stencil_operator.py tests/test_stencil_boundary.py -q
# 16 passed（B0 修复后仍全绿）
.torch-cuda-env/bin/python -m pytest -q
# 177 passed, 2 warnings in 85.5 s   （基线 144 → +33，无回退）
```

真实帧测试在 `data/EHL-2/.../jorek003000.h5` 存在时执行（本环境存在），
否则 skip-if-absent。帧级 rollout 只作 **smoke**：记录帧间距 ~10 个流体步，
既不是基线也不是 gate（ADR-0018 决定 4/10）。

## 11. 开放问题（按风险排序）

1. **边界码语义（已确认）/角点处理**：1/11/5/15=TARGET、9/19=CORNER free、
   2/12=TANGENT、20=CORNER fixed、21=CORNER inverted（用户 + JOREK 源码核对；
   见 B0 报告 §9）。model502 实施：2/3/12/20/21 → `apply_dirichlet_all`；
   1/3/4/5/9/11/15/19 → `apply_cs`；角点对在
   `construct_matrix_mod.f90:124-130` 被跳过。B2 的 BC 层需按此表复核（尤其
   corner-pair 的鞘层积分跳过）；Mach-1 线性化项仍是未复刻部分。
2. **刚性极限精度**（中高）：前向/反向稳定已保证，但 ZK∥ 在 dt=0.05 下的收缩率
   ≈1−1e-12，迭代基本冻结；真正的"沿磁力线准稳态投影"需要更强求解器
   （预条件 Krylov、multigrid 或精确核投影）或更多迭代。当前交付满足"隐式稳定"，
   不声称该极限的精度。
3. **系数取值链（已定位，待生成侧最终确认）**：H5 与 `jorek_namelist_spitest`
   一致（2.4e-3/5e-4/1e-10；D_par=5e-2、ZK_par_max=1e4）；`jorek_namelist_2d`
   等为旧文件。α 与稳定性表以 H5 + `jorek_namelist_spitest` 为准；生成侧确认后关闭。
4. **逐项算子口径（代码已核对，待实现）**：JOREK 的弱形式权重是逐项的——
   平行项 `BigR/BB2`（`mod_elt_matrix.f90:1546-1547,1595,1642`）、
   垂直/粘性项 `BigR`（`:1492,1565,1598,1641,1649`）、`visco_num` 不带 `BigR`
   （`:1496,2031`）。B2 当前 `metric_weight="none"`（或全局 `"jorek"`）不满足；
   B2.1 需改为**逐项权重**并重跑 λ_max/稳定性与分组置换测试。
5. **通道-项映射（代码已核对，待实现）**：`ZK_i_par`→T_i、`ZK_e_par`→T_e、
   `ZK_perp_num`→T_i/T_e（B2 当前未纳入）、`D_par_local*`→rho/rho_imp、
   `D_perp_num`→rho、`visco*`→u/omega、`visco_par*`→v_par
   （行号见 `docs/jorek-normalization.md` §3.1）。B2.1 按此表修订
   `DEFAULT_CHANNEL_TERMS`。
6. **Mach-1/鞘层线性化项**（中）：沿用 B0 的"声速 + 通量汇"近似；`density_reflection`、
   `gamma_sheath_*`、`vpar_smoothing`、`mach_one_bnd_integral`、`neutral_source` 与
   `rhs_ij_5..9` 的线性化未复刻（`mach_one_bnd_integral` 已读入 flags 但未使用）。
7. **FiLM per-channel 模式**（低中）：默认全局标量已过形状验收；`film_per_channel=True`
   未做验收，若启用需重跑分组置换与深层 Jacobian 测试。
8. **LTC target 训练后的有界性**（低中）：当前 target 恒等初始化 + clamp；训练后需
   保持 clamp/权重约束，否则 target 可能拉出分布。
9. **freeboundary=T**（低）：本数据为 F，未触发；真空耦合未实现（同 B0 §6.1）。
10. **帧级 smoke 的定位**（低）：ADR-0019 的配对 `(u_k, p_{k−1}, S_k) → u_{k+1}`
    尚未在 B2 组件里接线（属 Phase C）；本报告所有真实帧数字不是基线。
