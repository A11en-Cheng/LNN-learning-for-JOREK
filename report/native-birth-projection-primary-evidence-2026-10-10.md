# 原生 marker 出生与五通道投影：一手证据审计（2026-10-10）

审计对象为只读 `/home/ac_desktop/JOREK_code/jorek` 源码；本仓库 Python 仅作明确标注的对照。已先读 `docs/adr/README.md`；本文不以 ADR-0001..0017（历史）作为门槛。不修改实现、不执行训练、不运行或修改 JOREK。以下为静态源码证据与由公式推出的验证设计，**没有数值收敛或生产资格通过的声明**。所有行号为本次读取快照的闭区间；末尾 SHA256 锁定文件内容。

## 1. 两种概率对象与两类误差必须分开

JOREK 出生路径没有本仓库的空间概率表：它在圆柱体积均匀提议位置，定位网格，调用 `total_imp_source`，再以 `ran(4) < f_source_imp(...)` 决定接受；`f_source_imp` 返回 default-real/real*4 的非负阈值，阈值超过 1 时比较自然饱和。因此目标密度相对于体积测度为 `a(x)=min(max(real4(P(x)*dv/P_norm),0),1)`，再限制于可定位域并归一化。这里 `P` 为所有注入贡献之和，不能逐 shard clipping 后求和。证据：[初始化 498–545](/home/ac_desktop/JOREK_code/jorek/particles/mod_initialise_particles.f90:498)、[阈值 809–817](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:809)、[圆柱变换 74–84](/home/ac_desktop/JOREK_code/jorek/tools/mod_sampling.f90:74)。该密度表达式是对源码比较逻辑的数学推论。

本仓库 `NativeTabulatedBirthSampler` 是额外的显式数值实验：在 `R²/Z/phi` 中用顶点阈值构建正三线性函数，单元质量等于八顶点均值乘精确圆柱单元体积，归一化 CDF 后以四个 uniforms 和条件逆变换采样。采样对**该表**成立，不能据此证明连续原生目标正确；`production_qualified=False`、`continuous_source_qualified=False` 明确记录此限制。[本地表采样 27–63、配置 96–106、制表 160–220](../src/models/native_birth_positions.py)

另一个对象 `P_imp` 是每个 marker 的电荷态概率分布。原生程序默认 `CE_marker=.false.`；通过 ADAS SCD/ACD 插值作显式转移，对负分量逐项截零，再除以总和。这条更新没有 RNG draw。时间离散、ADAS 插值、截零和重新归一化造成的误差属于确定性算子误差；它们不等同于空间表误差或有限 marker 随机误差。[默认 98](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:98)、[转移 543–603](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:543)

**审计建议（公式推论，尚未执行）：**

1. 固定源、归一化、bounds 和 clipping，对连续 `a(x)` 作独立高精度体积积分，比较表积分与归一化后的分布/矩；依次细化 RZ 和角向。CDF 总和等于 1、单元体积正确、同一张表采样自洽均不能替代这一步。
2. n0 必须积分 `mean_phi(clip(total_source))`；不能换成 `clip(mean_phi(total_source))`。当前同 phi 中心分支使用解析 clipped mean 再在顶点舍入 real4，与原生逐点评估后 real4 舍入仍有固定差异；多中心分支使用角向 midpoint 积分。n0 均匀 dummy phi 只声称近似 RZ marginal，不声称原生 joint phi law。[本地语义 66–74、角向实现 162–192](../src/models/native_birth_positions.py)
3. 固定表分辨率和 marker 数做独立 seed ensemble，报告每通道均值、标准差、置信区间；再细化表并提高 marker 数。表细化检验固定偏差，seed ensemble 检验随机误差。单一 seed 与原生拒绝采样的逐 marker 不一致不构成偏差证据；ensemble 的不确定性也不能掩盖未消失的表偏差。

主任务补充的经验结果（主代理实测，本 sidecar 未重跑，见 [细化报告](native-birth-refinement-seed-2026-10-10.md)）：固定 `cpw=128`、96-bin 比较，独立解析/Gauss 参考得到 `probabilityL2≈1.1e-5 / 9.2e-6`。这支持该配置及所测分箱下的空间概率一致性，但不是五通道投影误差上界：分箱聚合不能控制箱内 FEM basis 加权误差；概率 L2 未包含出生权重、charge/radiation、轨迹、历史与投影逆矩阵的放大。即使每个分箱概率完全一致，箱内位置不同也可改变沉积 RHS。要给投影上界，需对应连续测度/测试函数误差以及同一 mesh/filter 下投影算子的稳定性界；完整窗口还需控制后续状态演化。此解释由 §2/§4 的原生沉积和求解公式推出，不否定上述概率实验，也不声明已完成额外验证。

## 2. 出生数、权重与五通道

标称 slot 权重 `physical_particles/n_particles` 只用于决定出生数；每 rank 使用总 ablation impurity rate 除以 CPU 数，出生数取 `int(rate_local*gather/nominal_weight)`，若局部注入量大于 `1e14` 则至少一个。实际 newborn 权重为 `real4(rate_local*gather/count)`，不是固定标称权重。出生位置边界使用 shard 周围五倍云宽，R/Z 各自与域求交；不重叠的坐标退回全域。[标称 94–96](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:94)、[bounds/rate/count 392–437](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:392)、[bounds 391–414、actual weight 537–541](/home/ac_desktop/JOREK_code/jorek/particles/mod_initialise_particles.f90:391)

令 `dt` 为粒子子步、`gather=n_steps*dt`、`w` 为 marker weight、`p_z` 为子步更新**前**的电荷分布；每个子步先计算下表贡献，再在旧位置乘 FEM basis、element size 和 toroidal basis 累积。最后统一除以 `gather`，然后解滤波投影系统。证据：[矩与辐射 531–548、能量差 598–607、投影 614–642、平均 669–679](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:531)。

| 索引 | 原生子步量 | RHS 归一化（不含空间 basis） |
|---|---|---|
| 1 | `dEion_dt = w * sum_z((p_new-p_old)*E_ion_tot(z))*EL_CHG` | `t_norm/E_norm` |
| 2 | `rad_sink = w*sum_z(p_old(z)*Lz(z))*dt` | `t_norm/E_norm` |
| 3 | `Z_eff = w*sum_z(p_old(z)*z²)*dt` | `1/n_norm` |
| 4 | `Z_imp = w*sum_z(p_old(z)*z)*dt` | `1/n_norm` |
| 5 | `N_imp = w*dt` | `1/n_norm` |

`dEion_dt` 是一次子步的能量变化，源码变量名不意味着需要再乘一次 `dt`。`E_ion_tot` 是累积电离能量，创建于 274–278。索引 3/4 是加权二阶/一阶电荷密度矩，不能直接称为平均电荷；程序计算平均电荷时读取通道 4/5 并相除。[累积能 274–278、旧投影反馈 496–526](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:274)

## 3. “零出生”与“中性 newborn 首个子步”

**本窗口没有新出生**只跳过 `initialise_particles_marker`，并不跳过已有 alive markers 的全部子步，因此五通道可继续非零。[出生条件 429–440、所有 markers 循环 477–486](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:429)

**刚出生的中性 marker**被设为 `P_imp(0)=1`、其他为 0，并初始化 `V_prev`。[初始化 537–541](/home/ac_desktop/JOREK_code/jorek/particles/mod_initialise_particles.f90:537) 其首个子步：

- 更新前 `Z_imp=Z_eff=0`，所以通道 3/4 的该 marker 首个贡献为零。
- `N_imp=w*dt>0`，所以通道 5 的该 marker RHS 贡献不为零。
- 当 `T_eV>1` 且中性 `Lz` 非零时，通道 2 可以立即非零；源码计算 iZ=0 的辐射，没有“中性必无辐射”的分支。
- 通道 1 使用更新后的电荷分布差；发生电离时可立即非零。

以上由 [531–603](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:531) 直接推得。首个贡献后 `P_imp` 更新，下一子步通道 3/4 即可非零。不能把“零出生”“中性初态”“零 startup projection”“零整个窗口五通道”相互替换。只有没有任何有效 marker、正 gather 等附加条件下，累计 marker RHS 才为零；源码对 `i_elm<=0` 退出该 marker 子步。[484](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:484)

## 4. 投影滤波、history 与完整固定场窗口

反馈 projection 使用 `filter_perp/hyper/par` 及对应 n0 系数构造系统。tokamak build 选择 `mod_uncoupled_projection`，并通过 `assemble_system` 建立滤波矩阵；MPI 汇总 RHS 后求解 `Ax=B`，写回辅助节点 values/deltas。[分支 28–32、装配 188–195、projection 调用与清理 242–277](/home/ac_desktop/JOREK_code/jorek/particles/diagnostics/mod_project_particles.f90:28)、[汇总/解 202–225、写回 274–294](/home/ac_desktop/JOREK_code/jorek/particles/diagnostics/mod_uncoupled_projection.f90:202)

因此给定固定 mesh、filter、boundary 和累计 RHS，投影求解本身是确定性线性操作（数值求解/并行舍入另计）；“投影随机误差”来自输入 cloud 的空间采样及其轨迹/权重，而不是 `Ax=B` 新抽 RNG。条件于 cloud 的比较应复用相同累计 RHS；比较出生算法应报告 projected channels 的 ensemble。若允许通道 4/5 的反馈进一步改变 drift、electron density 和后续 charge，则整个多窗口映射不再是纯线性误差传递。[反馈 496–526、推进 655–662](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:496)

原生不是天然固定场 oracle：场读取器为 `read_jorek_fields_interp_linear`；raw deltas 随时间插值，static 条件才跳过这部分。[reader 75–77](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:75)、[插值 158–185](/home/ac_desktop/JOREK_code/jorek/particles/mod_fields_linear.f90:158) 新born `V_prev` 在 `sim%time` 采样，而子步从 `sim%time-step_rest_time` 开始，不能把这两个时刻无条件合并。[初始化调用 433–437、窗口 350–355、子步 486](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:350)

**完整固定场窗口验证设计（未执行）**：固定连续场值/梯度及其时间接口，保留实际 gather、residual clock、所有已有与新生 markers、旧辅助投影 4/5、出生权重/初始 `P_imp`/`V_prev`，执行全部子步的 charge/radiation、五通道旧位置沉积、drift/AB 推进和定位，最后作时间平均及相同滤波投影。不能仅验证出生点直方图或单个 newborn 即声称五通道窗口正确。AB 的 `V_prev` 是历史状态；pusher 用归一化当前/前次速度并在末尾更新 `V_prev`，当前 `V_ext` 参与速度但出生初始化只取流体 `V`。零速度还可能导致归一化除零，固定场例子应选择有效速度或另行明确处置，不默默将原生算子替换成 Euler。[AB 53–78](/home/ac_desktop/JOREK_code/jorek/particles/pushers/mod_fieldline_euler.f90:53)

完整固定场窗口输出误差同时包含出生分布、charge 积分、轨迹/定位器、history、沉积与滤波差异，因此只能检验整条链的输出一致性，不能独自隔离空间表偏差。应先按 §1 固定源做连续密度与表的确定性积分/细化比较，再用相同窗口其余算子做 seed ensemble；逐层控制比较对象才可归因。

原生先定义 projection event，再定义 JOREK stepper；窗口末尾调用 events。场推进的 `update_values/update_deltas` 与新 time endpoints 在 JOREK event 内完成。[events 175–190、调用 749–754](/home/ac_desktop/JOREK_code/jorek/particles/examples/marker_spi_source.f90:175)、[场更新 424–425、endpoints 566–569](/home/ac_desktop/JOREK_code/jorek/core/mod_jorek_timestepping.f90:424)

## 5. RNG 与 debug 契约

原生出生每次初始化由 rank 0 生成 seed 并广播；每线程 PCG sequence 为 `rank*n_threads+thread+1`。手动 seed 可以固定 seed，但不同 rank/thread 拓扑、分配与重试仍改变 slot 与 stream 的对应关系；不能仅靠 seed 声称拓扑无关 replay。[seed/stream 416–418、452–460](/home/ac_desktop/JOREK_code/jorek/particles/mod_initialise_particles.f90:416)、[manual seed 10–24](/home/ac_desktop/JOREK_code/jorek/tools/mod_random_seed.f90:10)

默认 `random_seed()` 将 `use_xor_time_pid=.true.`，采用 time/PID 经 LCG 得到的 seed；只有显式传入 `.false.` 才尝试 `/dev/urandom`，失败仍回退 time/PID。故不能将无参调用描述为默认读取 urandom。[15–22、45–74](/home/ac_desktop/JOREK_code/jorek/tools/mod_random_seed.f90:15)

**四与七需精确归属：**原生 initializer 调 `initialize(7,...)`，但实际声明 `ran(4)` 并 `next(ran)`；PCG wrapper 明确不使用 n_dims，`next` 按输出数组交给底层。因此这份源码可证明原生此次请求四维输出，不能把 initialize 参数 7 当成每次 draw 七个数的证据。[声明 330、初始化 460、draw 498](/home/ac_desktop/JOREK_code/jorek/particles/mod_initialise_particles.f90:330)、[PCG wrapper 22–56](/home/ac_desktop/JOREK_code/jorek/tools/mod_pcg32_rng.f90:22)

本仓库默认 rejection 路径则确实 `np.random.random(7)`；opt-in table 每个位置四个 uniforms，额外 FEM rejection 会再抽位置。两者是 MT19937，不是生产 PCG32 位级 replay；必须用配置 fingerprint 区分。births 在失败时恢复完整 NumPy RNG state，成功时推进；外层 coupled reference 在失败时恢复 Python/NumPy/Torch RNG 及模型。[本地 birth 配置 27–45、draw/rollback 102–137](../src/models/native_source_birth.py)、[本地外层事务 1005–1029](../src/models/native_coupling.py)

底层 PCG 确认按 `size(out,1)` 循环；原生另有 `use_fixed_rng_value` 调试开关：先抽 PCG 值，再覆盖为 `fixed_rng_value`。它仍推进 RNG state，而且恒定提议可能始终被拒绝；不能把这个开关当成正常随机分布或 seed ensemble。[底层 PCG 39–48](/home/ac_desktop/JOREK_code/jorek/tools/mod_pcg32.f90:39)

**debug 要求（由上述事务语义推导，未证明所有 debug 入口实现）：**只读诊断应不额外消耗事件 RNG、不重新 sample、不重建隐式跨事件 cache；需要重跑时在独立 RNG 或完整 snapshot/restore 中执行。失败重试需比较完整 state，而不只 seed。不同 draw 协议的 mode 切换不应承诺 legacy trajectory replay。此次所读原生出生链未见 Python 式 debug snapshot/rollback 契约，不能把本地保障写成 JOREK 原生保障。

## 6. SHA256 源快照

前十一个路径相对于 `/home/ac_desktop/JOREK_code/jorek`；最后三个相对于本仓库。每项为完整文件哈希，行号引用只对该快照有效。

| 文件 | SHA256 |
|---|---|
| `particles/examples/marker_spi_source.f90` | `4f6e832b107569d7a8a0155ce5995d04d59cc50771d256cdf99c1e692318991c` |
| `particles/mod_initialise_particles.f90` | `269ce1cea5d5029c561f8ba93aacf4987f9c24163708f6df1a085cd6ba61f22d` |
| `tools/mod_sampling.f90` | `25219d60fba943e3547ddf394b9c19a228970a2855ad3f50bb6ad1693884c7b2` |
| `tools/mod_random_seed.f90` | `b3fb9035aa0350a16bc61bbe125cfdb1ae23ba4254b32a7afc9685d30904a28e` |
| `tools/mod_pcg32_rng.f90` | `a6009b65f7e95e60b284c273303ac5e5ff053aba22d02fca3f192c6062bb89b1` |
| `tools/mod_pcg32.f90` | `8d37a21cbb86ff3d3f567d98a90da955a054287cfb4cbb0f51c505e0c5bf64f8` |
| `particles/diagnostics/mod_project_particles.f90` | `f04d4b2a25ada16193ad893b7350d96c173ea1d3c02ef703f97ce3c44cca70c9` |
| `particles/diagnostics/mod_uncoupled_projection.f90` | `a79b3dc940d23de1c0cd10831d3bea58e84e2703e50d699cc8863f13e06ea25e` |
| `particles/pushers/mod_fieldline_euler.f90` | `95b0635c67e9499c563547fefa74a247d21dc6feec7fd06a6141dfe3d72467a9` |
| `particles/mod_fields_linear.f90` | `aad6cb4b82d878bbdbb0de5833af5d61c8c00cb29a7c645ce3ccc54ead2bc96b` |
| `core/mod_jorek_timestepping.f90` | `f45b62fff5165f2c0f728c7a24d14b919628ff455ed4d1497daafaecceecb5fe` |
| `src/models/native_birth_positions.py`（本地对照） | `444f7d882ed6fa9a6ed27cb3d51b94b546eab648cf23362a6b3cbef47f7b16b6` |
| `src/models/native_source_birth.py`（本地对照） | `cbd01cac5aae076536a84d43f668f9d75e5a70e9b3680bfc348a05b8eb9e60aa` |
| `src/models/native_coupling.py`（本地对照） | `f6dd533258fe83451c301888297a1154cf0b9d5fcdfb9c7ade23432d90bd551b` |
