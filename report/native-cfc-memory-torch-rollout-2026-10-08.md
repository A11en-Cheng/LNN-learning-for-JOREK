# 切片 T：CfC、显式场记忆与完整原生 Torch 多步推进

2026-10-08。完成口径：**CfC 残差时间核心与显式记忆已接入，完整原生场的多步可微路径及恢复通过验证。** 没有训练、优化器、CfC 精度/稳定性结论、完整 JOREK 求解器、新反馈消费者或 C0 验收。遵循 ADR-0018/0019/0020/0021 的定向覆盖；不修改 roadmap、ADR 或 JOREK。其他用户修改保留。

## 1. 来源、变体与公式事实

已核查本地 .torch-cuda-env：ncps **1.0.1**，Apache-2.0，安装路径为 .torch-cuda-env/lib/python3.12/site-packages/ncps/torch/cfc_cell.py。使用原版 ncps.torch.CfCCell，未复制或改写单元公式。CfCCell 类源码 SHA256：

    946eded2eb6ce785512295d224eebb7b088998ff9e6051e87be89ea6621c810e

[作者论文 arXiv:2106.13898v2](https://arxiv.org/pdf/2106.13898) 的 Eq.10 / Fig.4 描述共享 backbone 的双分支 sigmoid 插值 CfC；[作者官方实现](https://github.com/mlech26l/ncps/blob/master/ncps/torch/cfc_cell.py) 的 default 分支提供带可学习时间偏置的参数化。这里选择官方实现的 default；不用 pure、no_gate、LSTM mixed-memory，也不调用旧 LTC 指数松弛核心。官方线上 master 只用于核对来源，实际运行版本由本地发行版与类源码 hash 登记。

设 z=backbone([x,h])、a=tanh(ff1(z))、b=tanh(ff2(z))，τ 为网络时间：

    g = sigmoid(time_a(z) * τ + time_b(z))
    h_next = (1-g) * a + g * b

官方单元返回的 output 与 next_hidden 相同。配置固定 mode=default、backbone_activation=lecun_tanh、backbone_layers=1、backbone_dropout=0、无 sparsity mask、无 mixed memory。真实诊断 hidden=12、backbone_units=24；小测试 hidden=8、backbone_units=11。没有声称严格守恒、刚性稳定、半群或单元零时间恒等。

src/models/native_temporal.py::CfCTimeCore 是可替换时间核心的 Interface：接收 [N,hidden] 特征、显式 [N,hidden] 记忆和正 JU 间隔，返回 output/next_memory。NativeFieldStep 的 temporal_core=None 保留既有无记忆候选；不建立插件注册框架。核心宽度须匹配现有 latent 宽度。ncps 版本不符明确拒绝，版本/hash/config 均进入 checkpoint 配置。

## 2. 逐层 shape、残差与时间所有权

G/P 的物理求值、缩放、参数 schema 和池化近似不变，详见[兼容特征报告](native-compatible-features-parameters-2026-10-08.md)。

| 层 | shape / 语义 |
|---|---|
| 完整输入 U、rawD | 各 [9,4,1,N]，原生单位，独立历史 |
| 归一化原生输入 | [N,36] U + [N,36] rawD |
| 时间输入 | [N,5]，当前 JU dt、前 JU dt、比值、theta、zeta |
| 兼容值/梯度 | [E,16,9] 与 [E,16,9,2]，每步由实际输入完整 U 重算 |
| G/P 空间观测 | [N,27]，场景 [4]、eta·∇ψ [N,2]，继续独立路由 |
| 现有编码及残差 stencil | [N,hidden]；CfC 启用时空间残差连接保留 |
| CfC 输入/记忆/输出 | 各 [N,hidden]，显式 recurrent memory |
| 原生解码及映射 | [9,4,1,N] 本步增量 |
| 下一状态 | U_next=U+Δ_applied；rawD_next 由更新映射直接给出 |

CfC 位于原有空间/梯度/参数编码和残差空间主干之后、完整原生解码之前。所有九场四 degree 联合解码，保持完整 U 恒等支路，禁止以有损 latent 重构替换当前场。使用现有 NativeUpdateMapping，含共享原生 index 平均与 keep_n0_const 分支，不从舍入后的 U_next−U 重建历史。

时间仅由显式运行计划提供。SI → JU 的转换位于 NativeFieldStep：h=dt_si/t_norm_si、k=previous_dt_si/t_norm_si。CfCTimeCore 再使用 τ=h/time_scale_ju；真实诊断 time_scale_ju=1，小型独立测试为2。此 scale 是配置，不从标签或未来时间拟合。时间格式仍由原有 fluid_time_coefficients 验证。

当前/前 JU 步长进入原有编码；theta*h 继续作用于学习 latent stencil；τ 进入官方 CfC 门。它们分别是学习特征、空间候选迭代时间和 recurrent 时间，不是重复把物理载荷积分。decoder 输出仍为 **本步原生系数增量**，只乘既有 increment_scale，解码后不再乘 dt；不称为 rate、弱式载荷或 marker forcing。

完整零时间 no-op 在 NativeFieldStep 包装内、空间/特征/单元/解码之前执行，合法 U/rawD/记忆和前步长/格式均先校验，返回相同 U、rawD、memory 引用和零 increment。advance_field 返回同一个 TorchFieldState，时钟、步号和前 dt 不变，即使学习权重及偏置非零。官方 cell 自身不承担 no-op。正步长下 decoder 权重和偏置归零保留 U，但自由更新行的 rawD 更新为零、记忆仍推进；这是与 no-op 不同的契约。

零时间分支不保证 dt→0 的连续极限；本轮只验证精确 no-op，不推定步长一致性、刚性响应或长时间稳定。ReferenceRollout 的真实事件和窗口计划仍要求正步长；no-op 不生成 JOREK restart，也不能解释登记 restart 的重复流体记录/粒子窗口。

## 3. 完整状态与显式记忆契约

src/models/native_torch_rollout.py::TorchFieldState 保存：

- values、raw_deltas：[9,4,1,N]；
- memory：[N,hidden]，场时间学习记忆；
- NativeUnits、SI 端点时间、前 SI dt、内部步号、时间格式；
- 完整模型配置 digest、native-torch-field-v1 版本。

场记忆不等于 raw fluid_deltas、marker 云、projection deltas、marker h_residual 或 shard。模块没有内部可变记忆缓存；单步输入 memory 缺失/轴错/dtype/device 不符/非有限明确失败。

生命周期：
1. initialize_field_state 只从显式当前完整场 checkpoint 冷启动，独立创建零学习记忆；不是伪造空 marker 云，也不是自动在每步调用。
2. 因果 warm-up 从窗口起始锚点冷启动，逐个消费预测起点以前的真实 U/rawD，按真实已声明时间推进记忆，不计目标 loss。到预测起点安装匹配真实 U/rawD，保留暖身记忆。
3. 普通 chunk/模型连续运行传入已提交 TorchFieldState，验证其预测起点时钟/格式/前 dt，不重放 warm-up、不重新初始化。
4. 模型续跑恢复全部 U/rawD/记忆/时钟及模型权重和配置；不附加 JOREK restart。
5. TBPTT 仅在调用者提供的 detach_before 位置 detach U/rawD/记忆的图，数值、时间、步号均不变。没有隐式截断、清状态或课程优化器。

## 4. Torch 窗口循环与旧事件接入

公开 Interface：
- initialize_field_state(model,native_state,previous_dt_si=...,scheme=...)；
- advance_field(model,state,dt_si=...,parameters=...)；
- rollout_field_window(model,window,initial_state=...,teacher_mask=...,detach_before=...)。

TorchWindowResult 返回预测 states、实际每步 inputs、warmup_states、warm-up/loss mask 和明确 field_only_no_feedback_consumer 范围。沿用现有 NativeWindow 和运行计划，循环只是 field-only 计算图，不是新的源/场/shard 调度器。teacher_mask 可显式覆盖窗口策略：
- False 使用前一步预测 U、预测 rawD 和 CfC 记忆；
- True 同时安装该步真实 U/rawD，保留学习记忆，重新计算该实际输入场的 Gauss 值/梯度；
- 当前步固定参数来自窗口当前合法输入，不读取未来参数记录；
- 未来 targets 默认只作为标签，显式教师开关才允许消费已到达步的 U/rawD。改写未来标签不影响 AR 输入和输出。

不使用 ReferenceRollout 的 no_grad/NumPy Adapter 做 BPTT。source teacher policy 非 none、coupled qualification 明确拒绝；本轮没有读取/消费反馈，None 始终表示未提供，未转为零。

既有 ReferenceRollout 增加独立 field_memory，FluidSolveResult 增加可选 field_memory。NativeReferenceSolver 声明支持该参数，显式接收/返回；reference_fluid_event 仍在私有候选上执行原有 solve → old-field shard update → 共同提交 U/rawD/场记忆/时钟。marker h_residual 保持独立。带记忆状态配无记忆 solver 失败；native context 声明 CfC 后缺记忆也失败。

该推理 Adapter 仍只支持来源明确的真实零反馈/零投影历史；feedback=None 或非零反馈拒绝。真实 phaseA 缺投影历史没有被补零，所以真实多步案例走明确 field-only Torch 路径，未声称耦合推进。物理 BC 未验收、node P1 latent stencil、共享 index 平均和 Gauss→节点池化近似继续保留；source_proj 未物理对齐限制不变。

## 5. 保存/恢复与原子性

复用 marker_state.py::save_model_rollout / load_model_rollout；没有另造 checkpoint 格式/调度器。新增 TorchFieldState 的完整配置/状态校验接缝，ReferenceRollout 继续走现有原生 context 校验。保存同目录临时文件后原子替换；恢复先检查 schema/signature/状态、在私有模型验证所有权重，再提交调用者模型。错误配置、记忆缺失/宽度错误在提交前失败。

核心配置含库版本/源码 hash、变体、宽度、网络时间 scale、记忆布局；模型 state_dict 包含官方单元全部权重、G/P 统计/几何及其他权重；状态 payload 包含完整记忆与时间历史。上下文/配置 mismatch 明确拒绝，不在恢复时重拟合统计。失败保存不会覆盖已有文件；失败恢复不修改目标模型。测试为 CPU float64；不宣称此轮已经验证跨设备恢复或 optimizer/RNG课程完整训练续跑。存储不会保存历史 autograd 图，恢复后建立新图；连续数值路径保持一致。

## 6. 独立测试证据与真实诊断

tests/test_native_temporal.py 新增10个公开 Interface 测试，按 TDD 逐项红→绿：
- 官方 CfCCell 同权重/输入/记忆/网络 dt 输出逐位相等。另设独立手算门 oracle：a=.2、b=.8、time_a=.4、time_b=−.1、scale=2，dt=.5 JU 时输出 .5；dt=1 JU 时输出 .2+.6*sigmoid(.1)，float64 atol=2e-16。
- 非均匀 .25/.75 SI 步、JU/network 缩放与 t_norm=1 / 1e-6 的等价输入输出；相同 U/rawD、不同记忆改变增量；历史、场景参数改变仍影响路径。
- 连续两步仅第二步损失，梯度回到第一步 U/记忆、初始 U/rawD/记忆及官方核心参数。显式第二步前截断后，跨切点初始状态梯度为 None，不是清零物理状态。
- 第二步 inputs 是第一步预测状态；同一求值器给出预测场兼容梯度，与真实第二帧梯度不同。未来标签改写不影响 AR；成套教师 U/rawD 和保留记忆测试。
- arbitrary nonzero bias 的完整 no-op、残差分支归零、缺记忆/非法 dt/不支持教师源/错误教师时间、失败不修改输入。
- 小型 ReferenceRollout 真实零反馈 fixture 的连续两事件与保存后新模型恢复一致，包含场记忆；物理零步长事件仍拒绝。
- 模型配置不匹配、坏记忆文件恢复/保存原子拒绝。
- 两个真实完整轴连续两步及跨步 backward、保存后新模型续步；动态路径禁止 detach、NumPy 转换和 HDF5 打开。两个真实案例均执行，未跳过。

真实 scope：v400 4111–4114、injInner 2704–2707；当前场分别4112/2705，预测两步，原生 [9,4,1,4125]、3986单元、16 Gauss点、记忆 [4125,12]。使用既有登记 restart 分段，普通 dt 分别 .05 / 5 JU；保留前 dt。每个独立组件 fixture 的训练统计仅拟合其四条唯一支撑，不使用上一轮 benchmark validation 统计。真实测试从当前场明确冷启动，另用小 fixture 验证 warm-up；不宣称已验证全 run 的暖身选择或模型误差。

真实 U/rawD 的全部九场四 degree 初始梯度非零且有限，第一步中间状态和初始/中间记忆有非零梯度，核心参数梯度有限。恢复后的 U/rawD/记忆用 torch.equal 或 array_equal，零容差；这是同一 CPU float64 确定性程序的续跑一致性，不是预测精度容差。初始冷记忆不能证明真实记忆已拟合或最佳暖身长度。

## 7. 参数量、成本与命令

真实诊断模型共 **3732** 参数：G/P 与原场步1932 + 官方 CfC1800。CfC 为 backbone (输入12+记忆12=24)→24，共600；四个24→12头各300，总计1800。decoder 仍输出36完整增量。没有 mixed-memory LSTM 额外参数。

诊断命令（无 optimizer，固定种子41，CPU单线程，float64，1次预热+3次测量）：

    .torch-cuda-env/bin/python tools/diagnose_native_temporal.py --output-dir /tmp/native-temporal-diagnostics-2026-10-08 --repeats 3

[机器可读诊断](native-temporal-diagnostics-2026-10-08.json) 包含版本、全部36梯度覆盖、配置、参数量及成本。单步与两步均包括完整 G/P 求值、场步、状态校验，反向仅末步损失；两步从当前冷初态起、不计 warm-up。setup 含读取、静态矩阵、训练统计和模型构建。

| 片段 | setup s | 单步 forward/backward s | 两步 forward/backward s | 进程峰值 KiB |
|---|---:|---:|---:|---:|
| v400 | 2.006530 | .041801 / .017299 | .088086 / .033030 | 859076 |
| injInner | 1.792478 | .036304 / .016590 | .085669 / .032664 | 880856 |

峰值是进程累计 RSS，包括 Torch、文件/窗口/静态几何和计算图，不是单一 CfC 分支增量内存；无 GPU 成本、JOREK 加速或生产吞吐结论。

| 验证命令 | 结果 |
|---|---|
| baseline：pytest tests/test_native_field_step.py tests/test_native_features.py tests/test_native_windows.py -q | 53 passed /10.08s |
| pytest tests/test_native_temporal.py -q | 10 passed /8.80s |
| pytest tests/test_native_temporal.py tests/test_native_features.py tests/test_native_windows.py tests/test_native_field_step.py tests/test_native_state.py tests/test_fluid_history.py tests/test_marker_state.py tests/test_native_gauss.py tests/test_marker_feedback.py tests/test_source_module.py tests/test_field_propagator.py tests/test_step_dataset.py tests/test_output_contract.py -q | **286 passed、1 skipped /37.42s** |
| diagnose_native_temporal.py 上述命令 | exit0，两个真实片段完成 |
| python -m compileall -q src/models/native_temporal.py src/models/native_torch_rollout.py src/models/native_field_step.py src/models/marker_state.py tools/diagnose_native_temporal.py tests/test_native_temporal.py tests/test_native_field_step.py | exit0 |
| git diff --check | exit0 |

上述 Python/pytest 均使用 .torch-cuda-env/bin/python。唯一 skip 是既有未交付 export_v3 的真文件测试；新增真实片段和既有 Fortran 参考执行。TDD 中预期失败依次为缺 CfC Adapter、场步缺 temporal_core、缺窗口循环、旧 checkpoint 不接受新完整状态、原推理 Adapter 未传记忆；随后修复。一次事件测试误把既有 (state,diagnostics) 返回值当状态，更正测试解包后通过。旧 zero_dt 拒绝用例按本轮要求改为 negative_dt 拒绝；事件正步长规则未放宽。

## 8. 剩余近似与下一轮训练条件

已验证的是核心来源、完整残差/记忆/历史数据流、兼容梯度重算、跨步反传与数值恢复。仍缺物理 BC 验收、非零五通道消费与完整耦合初始化/源推进、刚性/准稳态、长时间稳定性、训练后的预测误差和跨参数辨识。节点空间近似、共享 index 均值、固定几何 H=1 限制不变；完整系数保存不能使 latent/池化成为充分物理表示。

后续小规模场监督训练需独立实现课程/优化器：明确 L/K、场教师 mask、源教师策略和截断计划；只从训练 split 拟合统计；保存 optimizer/scheduler/采样器/RNG/课程位置，分别验收完整系数、兼容值/梯度及纯 AR 累计误差。当前推理 checkpoint 不冒充完整训练续跑协议。正式 coupled/C0 仍需真实可恢复 marker/shard/反馈/历史/时钟及声明消费者，不能借 field-only 诊断绕过。本轮未训练，也未更改已接受决策。
