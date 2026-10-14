# Report index

Canonical reports, status as of 2026-10-08.  Domain vocabulary lives in
[CONTEXT.md](../CONTEXT.md); physics conventions live in
[src/physics/definitions.md](../src/physics/definitions.md); code-verified normalization
lives in [docs/jorek-normalization.md](../docs/jorek-normalization.md).

**Scope rules**：`current` = 新架构（roadmap_v2 + ADR-0018/0019/0020/0021）可用；`v1` = 历史证据，
引用时必须标注"（历史）"，不得作为新工作的 gate/前置。隔离规则见
[docs/adr/README.md](../docs/adr/README.md)。

| document | scope | status | role |
|---|---|---|---|
| [native-compiled-atomic-optimization-2026-10-11.md](native-compiled-atomic-optimization-2026-10-11.md) | current | opt-in exact ADAS interpolation; no training/default change | 两事件反序配对、完整末态与RNG指纹一致；inner整步减少27.41%，v400无整步收益；保留最终拒绝守卫与实测版本区别 |
| [native-element-function-residual-bottlenecks-2026-10-11.md](native-element-function-residual-bottlenecks-2026-10-11.md) | current | cached-path full-event residual profiling; no training or algorithm changes | inner源推进占83%，ADAS/定位/沉积为主要项；v400转向shard源/诊断及重复接口准备；分开普通计时与剖析开销，完整末态/RNG一致 |
| [native-element-function-cache-2026-10-11.md](native-element-function-cache-2026-10-11.md) | current | opt-in bicubic/Horner function cache; no training/default change | 查询约5–32×；两组完整耦合耗时减少1.47%/8.85%，保留inner末段步长逐位差异及单槽相对补证 |
| [native-multi-window-timing-2026-10-11.md](native-multi-window-timing-2026-10-11.md) | current | no-training six-slice autonomous forward timings per run | 冻结64格出生/nearby/v2局部细分；首窗与后续分开、CPU/GPU分项闭合和完整云资源；不含训练loss/backward |
| [native-marker-boundary-exit-2026-10-10.md](native-marker-boundary-exit-2026-10-10.md) | current | explicit nondefault terminal residence approximation; no training | 已记录外边及完整 cubic 求交/两侧核验、末段电荷与五通道沉积、互斥版本与真实完整云复验；临终收敛缺口单独记录 |
| [native-marker-boundary-primary-evidence-2026-10-10.md](native-marker-boundary-primary-evidence-2026-10-10.md) | current | read-only primary source evidence | JOREK完整固定步沉积/overshoot/lost/邻接与失败语义，区别末段驻留数值近似 |
| [native-marker-local-subcycling-2026-10-10.md](native-marker-local-subcycling-2026-10-10.md) | current | explicit opt-in numerical transport and history; no training | 原49步及真实选中云收敛对照、首次完整云边界超限；后续边界修正在独立续篇 |
| [native-nearby-motion-increment-diagnosis-2026-10-10.md](native-nearby-motion-increment-diagnosis-2026-10-10.md) | current | diagnosed transport sensitivity; own-start increment statistics corrected; no training | 三个低密度主误差粒子的秒级逐值重放、扩散/子步反事实；修正第二窗本窗增量与累计差混用，nearby仍opt-in |
| [native-jorek-nearby-locator-2026-10-10.md](native-jorek-nearby-locator-2026-10-10.md) | current | opt-in saved-neighbour nearby workflow; no training | JOREK 限步跨边/坐标转换、独立实际 Fortran oracle；两组源窗口耗时减少36.32%/25.30%，inner 轨迹/投影差异和全局 seam 明确登记 |
| [native-relocation-optimization-2026-10-10.md](native-relocation-optimization-2026-10-10.md) | current | exact CPU work optimization; no training | 保序候选筛选＋调用内几何复用；两组百万槽源推进逐值一致、配对耗时减少25.26%/11.74%，非零历史/源状态续跑补充 |
| [native-ionization-window-review-2026-10-10.md](native-ionization-window-review-2026-10-10.md) | current | 64-cpw default; no-training independent review | 电离增量解释、8物理窗口×8种子、参考抽样噪声与空间/积分误差；条件诊断不等于自主闭环 |
| [native-birth-window-statistics-2026-10-10.md](native-birth-window-statistics-2026-10-10.md) | current | no-training conditional physical-window ensembles | 连续真实字段、完整公共云/history、三臂新生响应、完整direct分解核验及复现命令 |
| [native-ionization-discrepancy-primary-evidence-2026-10-10.md](native-ionization-discrepancy-primary-evidence-2026-10-10.md) | current | read-only primary source audit | JOREK ΔP电离能、单位/历史/ADAS与源码哈希；区分已测归因和待证灵敏度假设 |
| [native-birth-refinement-seed-2026-10-10.md](native-birth-refinement-seed-2026-10-10.md) | current | no-training refinement and debug replay | 独立96×96概率细化实测；128格产物保留，后续用户选择64格；内存/建表成本、显式seed与RNG恢复 |
| [native-birth-projection-primary-evidence-2026-10-10.md](native-birth-projection-primary-evidence-2026-10-10.md) | current | read-only primary source audit | 出生权重、五通道沉积、滤波/history与RNG；概率偏差与投影抽样噪声不可混同 |
| [native-birth-projection-comparison-2026-10-10.md](native-birth-projection-comparison-2026-10-10.md) | current | no-training fixed-field full-cloud birth A/B | 原生拒绝参考/新表与参考/参考噪声；完整源窗口、逐通道投影与seed复现 |
| [native-birth-native-correction-2026-10-10.md](native-birth-native-correction-2026-10-10.md) | current | native mass gate corrected; probability builder default; no training | clip后环向边缘、数量/权重/原locator/rawD初速、版本拒绝；真实全槽出生入口、独立分布误差/统计噪声与修正后的计时 |
| [native-birth-native-contract-evidence-2026-10-10.md](native-birth-native-contract-evidence-2026-10-10.md) | current | read-only primary source audit | 外层marker槽位重试、候选内shard求和；数量/clock/RNG/质量源gate/初始化及n0边缘的原生契约 |
| [native-birth-traditional-comparison-2026-10-10.md](native-birth-traditional-comparison-2026-10-10.md) | current | no-training birth-law validation; native mass gate mismatch unresolved | 独立传统目标积分/抽样噪声与饱和对照；发现两路径共有质量系数差异，injInner 原生概率偏差约16.23%，不改默认或定位 |
| [native-birth-clipping-order-primary-evidence-2026-10-10.md](native-birth-clipping-order-primary-evidence-2026-10-10.md) | current | read-only primary source audit | JOREK逐shard核、injector求和、总接受概率平台及质量归一化调用链；不把逐shard宽度/负值保护当作上限平台 |
| [native-birth-generation-2026-10-10.md](native-birth-generation-2026-10-10.md) | current | opt-in clipped-total-source position sampler; no location/default/training change | n0 环向平均与三维周期抽样接口、闭式条件逆CDF、完整shard真实位置生成计时、概率相对差及小网格恢复 |
| [native-neighbor-locator-experiment-2026-10-10.md](native-neighbor-locator-experiment-2026-10-10.md) | current | oriented-neighbor fallback experiment; no default/training integration | 完整真实云 source A/B、逐通道及轨迹增量相对误差；保留 injInner 离散契约失败，不用性能替代保真 |
| [native-locator-work-diagnosis-2026-10-10.md](native-locator-work-diagnosis-2026-10-10.md) | current | full-cloud source-window work census; no algorithm change | 换单元回退仅0.15–0.42%访问却占42–59%定位时间；候选规模、caller分解与JOREK邻接定位源码依据 |
| [native-full-pipeline-benchmark-2026-10-10.md](native-full-pipeline-benchmark-2026-10-10.md) | current | no-update full-chain timing; exclusive accounting closed | 当前默认 CPU 源＋CUDA 九场/CfC 两前缀两预测；拆开出生/推进/接口准备与逐场指标，保留首次开销、梯度边界与未分类项 |
| [native-coefficient-cache-experiment-2026-10-10.md](native-coefficient-cache-experiment-2026-10-10.md) | current | exact event-local native coefficient experiment; no default/training integration | 原生布局元素系数缓存，不改定位/几何；两真实完整 source 窗口保真，收益小，保留实验而不默认启用 |
| [native-cpu-thread-benchmark-summary-2026-10-09.md](native-cpu-thread-benchmark-summary-2026-10-09.md) | current | bounded native CPU worker benchmark; no training/default integration | 现有保真与实验优化汇总；v400 1/2/4/8 worker 全算子更慢，仅采样双线程小收益；保留 v1 比较器误报、正式 v2 与范围限制 |
| [native-marker-topology-cache-2026-10-09.md](native-marker-topology-cache-2026-10-09.md) | current | experimental static-map refresh and cubic-confirmed lookup; no training | 固定拓扑/求值矩阵同缓冲更新，完整 cubic 采样与候选原生几何确认；独立刷新和真实 source 窗口相对误差，不默认替换 |
| [native-jorek-shard-sampling-evidence-2026-10-09.md](native-jorek-shard-sampling-evidence-2026-10-09.md) | current | read-only primary source evidence | JOREK shard 移动/侵蚀/OLD 场/rocket 顺序，global/nearby/postproc 定位差异与可复用静态对象 |
| [native-marker-lookup-experiment-2026-10-09.md](native-marker-lookup-experiment-2026-10-09.md) | current | experimental memory lookup; not production-qualified, no training | 两组完整真实云单 source 窗口的速度/内存/相对误差；密集微单元解析定位及场表，保留轨迹尾部与额外丢失，不改变默认 backend |
| [native-cpu-locator-optimization-2026-10-09.md](native-cpu-locator-optimization-2026-10-09.md) | current | CPU accepted-trial geometry reuse; full-cloud exact A/B, no training | 保留 Newton/阻尼/容差与事件数量，复用接受试探点几何；两组四事件完整状态/时钟/RNG保真、公开事件回归与独立性能口径 |
| [native-cpu-sampling-optimization-2026-10-09.md](native-cpu-sampling-optimization-2026-10-09.md) | current | CPU sampling optimization; full-cloud exact A/B, no training | marker 消费专用求值、事件局部快照和 ADAS 逐表共享搜索；两组四事件完整状态/时钟/RNG保真与独立性能口径 |
| [native-coupled-backward-performance-2026-10-09.md](native-coupled-backward-performance-2026-10-09.md) | current | no-update current-weight forward/BPTT diagnosis | 最新课程权重严格恢复，两组完整真实云前缀＋两步自主耦合、四项监督与跨步梯度计时；明确 NumPy 梯度边界及 marker 矩阵求解范围 |
| [native-cpu-latency-2026-10-09.md](native-cpu-latency-2026-10-09.md) | current | no-training full-cloud CPU latency optimization | 事件内密度预处理、nonlocal 一次绑定、纯几何定位；同快照计时、真实密度对照及 CUDA 归约重复性，完整状态逐位限制明确登记 |
| [native-cuda-spatial-performance-2026-10-09.md](native-cuda-spatial-performance-2026-10-09.md) | current | shared CPU geometry and rejected GPU feasibility | CPU 共享几何的全量确定性保真、单次同环境延迟对照；GPU 全云定位分歧与更慢诊断保留，禁止据此宣称训练/生产资格 |
| [native-cuda-curriculum-2026-10-09.md](native-cuda-curriculum-2026-10-09.md) | current | bounded CUDA curriculum; machine result authoritative | 无真实场回授、固定节点记忆恢复对照、两窗口小集合 epoch、逐场 loss/相对误差与 CPU/GPU 成本 |
| [native-autonomous-training-pilot-2026-10-09.md](native-autonomous-training-pilot-2026-10-09.md) | current | bounded coupled training pilot | 无真实场回授、两 run 共享参数、train-only 缩放、逐场/源/位置指标与更新边界恢复；执行状态以报告和机器结果为准 |
| [native-autonomous-smoke-2026-10-09.md](native-autonomous-smoke-2026-10-09.md) | current | two-run full-cloud autonomous smoke, untrained | 预测 Ψ 自主诊断、批量粒子路径、无真值回授两事件及逐场/源相对指标 |
| [native-autonomous-equilibrium-evidence-2026-10-09.md](native-autonomous-equilibrium-evidence-2026-10-09.md) | current | supported fixed-boundary double-X diagnostic | 当前 cubic Ψ 的轴/X点/边界磁通，显式 LCFS 初态历史，失败不回退真实 ES |
| [native-training-pilot-data-audit-2026-10-09.md](native-training-pilot-data-audit-2026-10-09.md) | current | six-window data/support audit | 两 run 的 train/val/test 候选支撑隔离、源与场标签时序；资格不等于预测有效 |
| [native-cloud-replay-relative-errors-2026-10-09.md](native-cloud-replay-relative-errors-2026-10-09.md) | current | no training, component replay diagnostics | 原生单位的云分量/增量、身份事件与逐场/投影通道相对误差；区分数值恢复、摘要一致及物理预测精度 |
| [native-psi-feedback-integration-2026-10-09.md](native-psi-feedback-integration-2026-10-09.md) | current | no training, explicit teacher scope | 非零九场完整事件与同刻真实ψ诊断回授分开验证；教师支路不覆盖九场预测，不宣称自主闭环 |
| [native-real-chain-residual-followup-2026-10-09.md](native-real-chain-residual-followup-2026-10-09.md) | current | no training, execution and residual audit | 真实耦合事件后续核查、确定性消融率残差判断与全碎片源诊断成本修复；实际完成范围按报告结果区分 |
| [native-coupling-coordination-2026-10-09.md](native-coupling-coordination-2026-10-09.md) | current | partial, no training | 原子生产诊断/出生接线、预算与定位器；两完整真实云参考源窗口通过，真实生产完整事件仍为0 |
| `model502-schema-and-source-protocol.md` | current | active | authoritative nine-field + source data contract; §3.1a = output timing (ADR-0019) |
| `data-contract-v3.md` | current | active (2026-10-04 frozen; revised 2026-10-07 per ADR-0020) | Phase A contract v3: dual source object, time semantics, granularity, ADR-0019 pairing + registered restart rules, restart-segmented budgets |
| `data-export-spec-v3.md` | current | active (2026-10-04 frozen; revised 2026-10-07 per ADR-0020) | per-fluid-step export spec for the JOREK data-generating side (fields, units, source provenance, freeboundary/axis normalization, registered restarts) |
| `phase-a-coverage-2026-10-04.md` | current | active | Phase A coverage: deliverables, real-data timing evidence, EHL-2 rebuild demo, open questions |
| `phase-a-prime-data-audit-2026-10-04.md` | current | active | A′ field audit of the generation-side sources (no export_v3 delivered yet) + prioritized field-completion list |
| `phase-a-prime-data-interface-2026-10-04.md` | current | active | A′ interface delivery: export_v3 loader (`require_single_step`), acceptance CLI, budget-only readers, evidence |
| `phase-a-prime-native-loader-2026-10-07.md` | current | active | A′ v2: native stepwise layout loader, restart/zero-dt plan-1 handling, native budget/condition reads, acceptance CLI native mode, dedup spec |
| `phase-b0-stencil-boundary-2026-10-04.md` | current | active | B0 stencil operator + JOREK BC layer: geometry/spectral evidence, boundary tests, JOREK-side confirmations |
| `phase-b2-field-propagator-2026-10-04.md` | current | active | B2 nine-field propagator: implicit stencil stability, exact per-term λ_max, shallow FiLM, B0 fix, open items |
| `phase-b2.1-quasi-steady-2026-10-04.md` | current | active | B2.1: per-term metric weights, channel mapping, direct LU implicit solve, quasi-steady fidelity, sheath/Mach-1 replication |
| `phase-b1-source-module-2026-10-07.md` | current | active (erratum 2026-10-07 per ADR-0020) | B1 self-generated source module: `(u_{k-1},u_k,p_{k-1}) -> S_k`, analytic backbone + nonlocal max, split heads, bounded correction, single-step train/val, budget closure; **erratum: recorded projections are the marker-run true source, not a proxy** |
| [marker-persistent-state-implementation-2026-10-07.md](marker-persistent-state-implementation-2026-10-07.md) | current | reference components validated | Native checkpoint initialization, persistent cloud/feedback/time history, model restore and separate JOREK startup; no full closed-loop claim |
| [marker-feedback-consumption-2026-10-07.md](marker-feedback-consumption-2026-10-07.md) | current | Gauss formula components validated | Deterministic model502 five-channel consumption, charge closures/derivatives and electron energy/history; full FEM assembly and nine-field solve pending |
| [native-fem-gauss-primary-evidence-2026-10-08.md](native-fem-gauss-primary-evidence-2026-10-08.md) | current | primary source evidence | Native cubic degrees, signed sizes, geometry, quadrature and ordinary model502 local energy terms |
| [native-gauss-local-marker-loads-2026-10-08.md](native-gauss-local-marker-loads-2026-10-08.md) | current | local reference chain validated | Complete native degree evaluation and two local marker energy loads; global FEM assembly and nine-field coupling unverified |
| [b2-c0-native-state-feedback-design-2026-10-08.md](b2-c0-native-state-feedback-design-2026-10-08.md) | current | design proposal, not implemented | Complete native predictive state, typed load response, charge consumers, atomic rollout commit and next implementation slices |
| [native-state-fidelity-resume-2026-10-08.md](native-state-fidelity-resume-2026-10-08.md) | current | slice A state fidelity and resume | Full native axes/history, explicit units/clocks, atomic model checkpoints and measured phaseA per-step segments; physical feedback and C0 closure unverified |
| [native-joint-field-step-2026-10-08.md](native-joint-field-step-2026-10-08.md) | current | full native learning step structure connected | Joint H=1 all-degree Torch step, raw history/time inputs, supported update mapping and ReferenceRollout checkpoint adapter; physical BC/feedback and C0 unverified |
| [native-window-cfc-curriculum-design-2026-10-08.md](native-window-cfc-curriculum-design-2026-10-08.md) | current | accepted direction; implementation pending | Native residual state advance, compatible values/gradients, parameter routing, replaceable CfC core, causal windows and TF→AR curriculum; no trained/C0 claim |
| `conformance-roadmap-v2-2026-10-07.md` | current | updated 2026-10-08 | roadmap_v2 ↔ implementation/evidence/status; ADR-0021 revision and limited native-step completion |
| `model502-parameter-surface.md` (+ `.json`) | current | active | machine-checked zero-dimension parameter surface |
| `marker-projection-causality.md` | current | active (corrected 2026-10-04) | source-code + production-data evidence for marker/projection timing |
| `v2-doc-isolation-and-dirty-data-audit-2026-10-04.md` | current | active | dirty-data entry points for new work; safe-to-reuse list |
| `implementation-progress.md` | current | active (append-only) | dated log; read the newest entries first |
| `spi-analytic-baseline.md` | v1 evidence (analytic backbone still current) | active evidence | analytic SPI model 3 vs recorded shard ablation |
| `physics-residual-correction.md` | v1 | historical | Fortran-derived operator conventions; physics loss paused (ADR-0017/0018) |
| `ab-condition-2026-10-03.md` | v1 | historical evidence | full 61-dim concat hurts OOD; current rule = physical subset + FiLM (ADR-0018) |
| `roadmap-review-2026-09-29.md` | v1 | historical | review of the v1 roadmap |
| `stage-summary-2026-10-03.md` | v1 | historical (errata added) | LTC interpolation stage results |
| `roadmap-v2-readiness-2026-10-03.md` | v1 | historical (errata added) | readiness review; baseline framing retracted |
| `hyphy-glnn-audit-roadmap.md` | historical (2026-09-04) | historical | audit of the pre-Model-502 repository |

## Removed on 2026-09-24

Nine-field training outputs were deleted (`output/model502_baseline/**`: runs,
checkpoints, training plots, elliptic-surrogate checkpoint, audits).  The
nine-field training workflow scripts and the training-design records were moved
to `/tmp/repo_cleanup_20260924/` (temporary; not durable):

- workflow scripts: `train_stage0_segmented.py`, `train_baseline*.py`,
  `build_frame_cache.py`, `plot_stage0_metrics.py`, `diagnose_stage0_physics.py`,
  `eval_stage0_per_run.py`, `audit_*.py`, `scan_lambda_phys.py`,
  `run_source_surrogate_*.py`, `run_stage0_diag.sh`, ...
- records: `implementation-progress.md`, `current-work-summary.md`,
  `docs/adr/0008`, `docs/adr/0009`, the previous `report/README.md`.

Design documents: `README.md`, `CONTEXT.md` and `new_version.md` were rewritten
on 2026-09-24 for the interpolation-centred route
([ADR-0010](../docs/adr/0010-physics-loss-only-at-interpolated-times.md),
[ADR-0011](../docs/adr/0011-fragment-source-coupling-architecture.md));
`config.yaml` still describes the retired nine-field training pipeline and will
be regenerated together with the new implementation.

- [原生因果窗口与参数路由（2026-10-08）](native-causal-windows-parameter-routing-2026-10-08.md)：切片 W；两个显式 phaseA 片段、完整轴/rawD、split 支撑隔离及参数缺口；场监督资格不等于耦合资格。
- [兼容值/梯度与分组参数（2026-10-08）](native-compatible-features-parameters-2026-10-08.md)：G/P 的完整 H=1 可微求值、训练统计、池化和场步实验路径；不代表预测有效或 C0 闭环。
- [CfC、显式场记忆与 Torch 多步推进（2026-10-08）](native-cfc-memory-torch-rollout-2026-10-08.md)：切片 T；官方核心对照、跨步 BPTT、完整 no-op 和新模型恢复；未训练 field-only 结构，不是 C0。
- [小规模原生场监督课程实测（2026-10-08）](native-field-curriculum-small-2026-10-08.md)：两个 seed 各60更新、TF/AR 固定评估与完整训练恢复；field-only，无反馈消费者，保留 AR 增长与正 dt 连续极限缺口。
- [原生场正步长一致性修复与同预算对照（2026-10-08）](native-field-time-consistency-2026-10-08.md)：外层速率/记忆时间适配、旧权重拒绝、两个seed各60更新；field-only，不代表反馈或C0通过。

- [小规模原生源—场集成参考链（2026-10-08）](native-source-field-coupling-2026-10-08.md)：持久小云、完整 cubic 投影、双源近似响应、连续事件与恢复/回滚；冻结99已明确登记，五通道消费者仍不完整，不代表C0。

- [含源九场联合学习主路径（2026-10-08）](native-joint-source-learning-2026-10-08.md)：ADR-0022；双源/原始历史/电荷闭合的完整原生与Gauss空间编码进入stencil/CfC，冻结容量仅为互斥参考模式；小型事件与恢复通过，不代表物理准确或C0。

- [模型与训练集闭环缺口审计（2026-10-08）](native-closure-readiness-audit-2026-10-08.md)：不训练；有限真实文件与源码审计，定位步长解析、context/源时间防线、真实网格、初态历史及数据资格缺口，给出无训练修复顺序。
