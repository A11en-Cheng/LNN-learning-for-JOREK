# 小规模原生场监督课程实测（2026-10-08）

## 验收边界

本交付限定为**课程训练机制与短程 field-only 学习实测**。全部实验产物登记 `field_only_no_feedback_consumer`。未增加反馈消费者、PDE residual loss、完整 JOREK 求解器或训练核心搜索；没有修改 JOREK、roadmap 或已接受 ADR。遵循 ADR-0018/0019/0020/0021 的定向覆盖关系，沿用已验收原生状态、兼容求值和 CfC 路径。

反馈缺失保持 `None`，未补零。模型可能从场及原始历史中学习未显式提供的驱动影响，因此损失下降不证明正确源响应。物理 BC 未验收、Gauss 特征池化与节点 stencil 学习近似、source_proj 未物理对齐、非零反馈消费者缺失继续有效。本实验不是 C0、预测精度或长期稳定验收。

## 接口与实现

- [训练核心](../src/training/native_curriculum.py)：损失、课程、采样、优化及原子训练检查点。
- [数据准备](../src/data/native_training.py)：先审计显式片段，再生成独立 split 窗口，重新拟合统一训练统计。
- [固定评估及 dt 诊断](../src/training/native_evaluation.py)。
- [运行入口](../tools/train_native_field_curriculum.py)、[汇总入口](../tools/summarize_native_curriculum.py)、[唯一实验配置](../config/native-field-curriculum-phaseA-small.json)、[独立测试](../tests/test_native_curriculum.py)。

复用 NativeFieldStep、NativeFeaturePath 和 rollout_field_window，不经旧 teacher-source 训练器、旧 manifest 或 ReferenceRollout 的 NumPy/no_grad 训练适配器。当前完整状态 U/rawD 为 `[9,4,1,4125]`；3986 单元、每单元16 Gauss 点；场学习记忆为 `[4125,12]`。预测每一步都从实际输入 U 重新求值及求梯度。教师替换 U/rawD 成套执行，沿用已登记保留学习记忆的策略；本轮窗口间独立因果 warm-up，窗口内记忆连续。

沿用官方 ncps 1.0.1 default CfCCell（隐藏宽12、backbone24），来源、公式和单位转换见 [T 报告](native-cfc-memory-torch-rollout-2026-10-08.md)。官方 cell 源码 SHA256 为 `946eded2eb6ce785512295d224eebb7b088998ff9e6051e87be89ea6621c810e`。时间输入沿既有 JU→网络时间接口，time_scale_JU=1；输出仍为本步原生系数增量，不额外乘 dt。模型实际 **3732 参数，其中 CfC 1800**，24 个参数张量。本轮没有修改 cell 公式。

## 数据审计与初始化资格

先指定下列有限片段及独立时间计划，再读取内部时间/步号/当前历史步长进行一致性核查，没有扫描或加载全 run。

| split | run / 内部记录索引 | 记录数 | 声明预测步长 JU |
|---|---|---:|---:|
| train | phaseA v400 4111–4142 | 32 | 0.05 |
| 同 run 时间验证 | v400 4150–4161 | 12 | 0.05 |
| test | v400 4169–4180 | 12 | 0.05 |
| 跨 run 验证 | phaseA injInner 2704–2713 | 10 | 5 |

run 的完整路径、内部端点时间、原始文件、单位、注册 restart 与参数证据在 [manifest](../output/native-field-curriculum-2026-10-08/data/manifest.json) / [coverage](../output/native-field-curriculum-2026-10-08/data/coverage.json)。总计66记录；训练唯一支撑32，验证22，测试12。v400 两个未加载隔离带分别为4143–4149、4162–4168；完整窗口支撑（初始化、warm-up、目标）跨 split 互斥。L=1、stride=4、isolation=7，K=1/2/4 单独建 bank。

| K | train | 同 run 验证 | 跨 run 验证 | test |
|---:|---:|---:|---:|---:|
| 1 | 8 | 3 | 2 | 3 |
| 2 | 8 | 3 | 2 | 3 |
| 4 | 7 | 2 | 2 | 2 |

共45窗口；这些明确兼容块内候选排除原因为空。所有窗口仅具备 field_supervised 资格，**coupled_rollout 数为0**。66份对应投影均缺独立 projection deltas；本轮既不消费它们，也不伪造投影历史或 marker 空云。相关缺口登记在 coverage。

这些片段是逐步流体记录，不概称跨多流体步。v400 4111 是注册 restart 锚点，保存的前步长为0.5 JU；下一步计划0.05 JU，warm-up 保留比值0.1。重复启动场时间不被解释为零粒子采样窗口，也不触发训练 chunk 的额外 restart。损失覆盖的 v400 预测步均为0.05 JU，injInner 为5 JU，预测步长比均为1。后者提供大步长跨 run 诊断，并非训练过的参数/时间范围。

有限检查另发现具体例外 injInner 2714：内部时间8427.05，前一记录2713为8427，步长从5变为0.05 JU。当前生产 namelist 表不足以无歧义说明全部重启片段；故跨 run 实验明确止于2713，不用未来标签反推计划作为模型输入。此例外不能推广成全 run 数据结论。

所有完整系数/rawD 与 Gauss 值/梯度尺度均重新从**训练窗口唯一支撑记录并集**拟合，未复用两个诊断 fixture 的单独尺度。统计及版本见 [normalizer](../output/native-field-curriculum-2026-10-08/data/normalizer.json)、[feature-statistics](../output/native-field-curriculum-2026-10-08/data/feature-statistics.json)。验证与测试不参与拟合。

[参数路由](../output/native-field-curriculum-2026-10-08/data/parameter-routes.json)保留来源、单位、缺失状态，元数据不进入数值特征。66×61项中 declared=792、unconfirmed_default=396、unresolved_unit=2442、audit_only=264、duplicate_legacy_entry=132；仅既有确认路由进入模型。所选 nspi、pellet density、eta 均恒定；变化主要是 t_now/tstep。不能从本轮推出参数可辨识性或跨参数泛化。

## 损失与课程

四项权重依次为 coefficient=1、raw history=0.25、Gauss value=0.25、compatible R/Z gradient=0.1。U 与 rawD 按 field/degree 的独立尺度归一化，直接监督保存的原始 rawD，绝不从端点相减重建。Gauss 场与梯度先用物理单位求值，再各自归一化；真实梯度来自真实完整记录，预测梯度来自预测完整系数。

系数/历史在 harmonic、节点内均方后，对九场×四 degree 均匀平均；这是原生槽位监督，不等同于 FEM 能量范数。Gauss 项使用归一化正权 `w_s w_t |J| R`，分别对九场、九场×R/Z均匀平均。各步损失再均匀平均。没有容量矩阵、PDE loss 或插值真值。所有 unweighted/weighted 项、逐 field/degree 归一化 MSE 及原生单位平方误差均进入机器曲线，避免只报混合标量。

| 阶段 | 更新数 / seed | horizon | 场教师概率 | 源策略 |
|---|---:|---:|---:|---|
| 单窗口优化冒烟 | 1 | 1 | 1 | none |
| 局部 TF | 19 | 1 | 1 | none |
| 混合短段 | 20 | 2 | 0.5 | none |
| 纯 AR | 20 | 4 | 0 | none |

固定 seed 41、73，各60更新，总120，冒烟计入预算。AdamW lr=0.001、weight_decay=0，ExponentialLR gamma=0.995，显式 gradient clipping max_norm=10。未搜索配置，未根据测试或验证选 checkpoint；最终固定更新60。采样器对训练 bank 有放回抽样，教师 mask 用独立保存的 RNG。无 TBPTT 切点，四步图完整反传；接口支持显式切点 detach，不清物理状态或 restart。本课程是轨迹监督与 scheduled sampling，不是 DAgger 或偏离轨迹专家重解。

## 恢复与失败原子性

检查点保存模型/统计/核心配置、optimizer/scheduler、私有采样器和 Python/NumPy/Torch/CUDA RNG、课程位置、更新号、完整日志、最近完整 U/rawD/时钟/学习记忆。训练数据内容摘要与配置绑定。仅在 optimizer 完成更新边界保存，不序列化未完成 autograd 图。

加载先在独立对象上校验配置、静态统计、全部模型及 Adam 张量/步号、scheduler、记忆布局和 RNG，成功才提交；异常回滚。配置不匹配、损坏 Adam 状态/策略、损坏记忆均有无污染测试。连续训练与中途新模型恢复的下一窗口、教师 mask、损失、权重、优化器与记忆逐位一致。两个真实 seed 均在更新20保存并**实际新建模型加载后**继续训练。最终60检查点另在新对象中重新加载验证，没有增加 optimizer 更新。

检查点：[seed41 边界](../output/native-field-curriculum-2026-10-08/seed-41/boundary.pt)、[seed41 最终](../output/native-field-curriculum-2026-10-08/seed-41/final.pt)、[seed73 边界](../output/native-field-curriculum-2026-10-08/seed-73/boundary.pt)、[seed73 最终](../output/native-field-curriculum-2026-10-08/seed-73/final.pt)。最终重载证据见 [checkpoint-verification.json](../output/native-field-curriculum-2026-10-08/checkpoint-verification.json)。这些是可信本地产物，使用 Torch pickle 读取。

## 实测结果

所有120更新参数梯度均存在且有限，36个 field/degree decoder 行均有梯度覆盖，每次 optimizer 都产生非零权重变化。无 NaN 或反馈补零。

| seed | 同一冒烟窗口更新前 loss | 更新后 loss | 权重变化 L2 |
|---:|---:|---:|---:|
| 41 | 0.394825 | 0.335679 | 0.060982 |
| 73 | 1.066076 | 0.797801 | 0.061004 |

每个固定评估组/horizon 只取排序后前两个窗口，保持所有更新时刻相同；并非评价所有候选窗口或全 run。预测起点 train=4112/4116、同 run 验证=4151/4155、test=4170/4174、跨 run 验证=2705/2709。评估更新0/1/20/40/60；test **仅在最终60**计算，未参与课程或 checkpoint 选择。

同 run 验证 K=4 的固定窗口平均 weighted loss：

| 更新 | seed41 TF | seed41 AR | seed73 TF | seed73 AR |
|---:|---:|---:|---:|---:|
| 0 | 0.146760 | 0.984819 | 0.962711 | 6.407276 |
| 1 | 0.080959 | 0.481113 | 0.649816 | 4.436961 |
| 20 | 0.034140 | 0.116056 | 0.033146 | 0.134730 |
| 40 | 0.026538 | 0.089912 | 0.023992 | 0.095719 |
| 60 | 0.019600 | 0.038830 | 0.018661 | 0.038179 |

最终 horizon/组别完整结果：
| group / K | seed41 TF | seed41 AR | seed73 TF | seed73 AR |
|---|---:|---:|---:|---:|
| train / 1 | 0.195402 | 0.195402 | 0.203974 | 0.203974 |
| train / 2 | 0.176585 | 0.248252 | 0.181411 | 0.257242 |
| train / 4 | 0.156024 | 0.405772 | 0.159331 | 0.407522 |
| same_run_validation / 1 | 0.018711 | 0.018711 | 0.019178 | 0.019178 |
| same_run_validation / 2 | 0.018570 | 0.022191 | 0.018635 | 0.023163 |
| same_run_validation / 4 | 0.019600 | 0.038830 | 0.018661 | 0.038179 |
| cross_run_validation / 1 | 7.273459 | 7.273459 | 7.289849 | 7.289849 |
| cross_run_validation / 2 | 4.992801 | 5.100310 | 5.064189 | 5.310533 |
| cross_run_validation / 4 | 3.317325 | 3.613990 | 3.381748 | 4.346418 |
| test / 1 | 0.017037 | 0.017037 | 0.017772 | 0.017772 |
| test / 2 | 0.017038 | 0.020141 | 0.017383 | 0.022001 |
| test / 4 | 0.018278 | 0.038693 | 0.017535 | 0.035783 |

固定同 run 验证 AR 随预测时刻的误差仍增长；两固定窗口按相对时刻汇总：

| 相对时间 JU | offset | seed41 loss | seed73 loss |
|---:|---:|---:|---:|
| 0.05 | 1 | 0.018711 | 0.019178 |
| 0.10 | 2 | 0.025672 | 0.027147 |
| 0.15 | 3 | 0.042873 | 0.042077 |
| 0.20 | 4 | 0.068063 | 0.064314 |

最终同 run K4 AR 各项加权贡献（coeff/rawD/value/gradient）：seed41 为 0.011049/0.015118/0.002660/0.010001，seed73 为 0.011838/0.015222/0.001558/0.009560。Phi 梯度归一化 RMS 最大约0.916，不能被小幅静态场的低误差掩盖。逐九场×四degree系数与历史误差、各场 Gauss/R/Z 误差见 [field-degree-errors.csv](../output/native-field-curriculum-2026-10-08/field-degree-errors.csv)；原生单位平方误差与所有固定时刻/桶见各 seed evaluations.json 和 [metrics-summary.json](../output/native-field-curriculum-2026-10-08/metrics-summary.json)。

预测步 dt 桶：v400 为 JU<0.1，injInner 为 JU>=1；损失步的 dt/prev_dt 都在0.5–2桶。只有 warm-up 覆盖restart后比值0.1，不能声称已覆盖一般变步长预测；独立小测试覆盖可变 dt。跨 run 误差明显较大，且与 dt 分布变化混杂。固定 train 前两个窗口靠近 restart，误差高于后段验证，并不能推出泛化。不同 horizon 的逐步均值也不应当作同一任务性能排名。

![训练、固定验证与误差增长](../output/native-field-curriculum-2026-10-08/field-only-curves.png)

![全部 field/degree 误差](../output/native-field-curriculum-2026-10-08/field-only-degree-errors.png)

## 正 dt 趋零诊断：未通过连续极限

固定一个合法 warm-up 后 U/rawD/记忆/prev_dt，只改变正 dt；基准0.05 JU，倍率1至1e-6，另单独测试精确0。不改 CfC 公式，不混入未来标签。

| seed | dt倍率 | 原生增量/增量scale RMS | 记忆变化 RMS |
|---:|---:|---:|---:|
| 41 | 1 | 0.201685 | 0.350871 |
| 41 | 1e-6 | 0.202767 | 0.348204 |
| 41 | 精确0 | 0 | 0 |
| 73 | 1 | 0.191580 | 0.290608 |
| 73 | 1e-6 | 0.192149 | 0.288592 |
| 73 | 精确0 | 0 | 0 |

**正 dt 缩小时，场增量与记忆变化趋向非零值；精确0结构分支仍保持场、rawD、记忆、时钟完全不变。**因此 no-op 测试不能证明连续时间极限；不能声称时间一致性、半群或任意小步稳定。现有离散增量解码/CfC 包装在这一点存在明确缺口，下一轮如调整必须显式设计与验证，不能静默改公式。所有倍率、逐field/degree物理增量和 rawD 变化在 seed 目录 dt-limit.json。

![正dt诊断](../output/native-field-curriculum-2026-10-08/field-only-dt-limit.png)

## 成本与可复现命令

Torch 2.11.0+cu130，实际 CPU float64、单线程、确定性算法；未使用 GPU。seed41/73 训练前向+反向+更新分别18.77/18.06秒，固定评估45.08/40.77秒；含检查点与评估的各 seed 主循环65.91/60.88秒（不含初始数据准备）。峰值进程 RSS 2,233,120/2,237,616 KiB，包含数据、求值、评估和检查点副本，且是进程累计高水位，不等同模型增量显存。实际产物约153 MiB。没有同工作单元 JOREK 对照，不宣称加速。

命令（仓库根目录）：

```bash
.torch-cuda-env/bin/python -m pytest tests/test_native_temporal.py tests/test_native_windows.py -q
.torch-cuda-env/bin/python -m pytest tests/test_native_curriculum.py -q
.torch-cuda-env/bin/python tools/train_native_field_curriculum.py --config config/native-field-curriculum-phaseA-small.json --output-dir output/native-field-curriculum-2026-10-08 --audit-only
.torch-cuda-env/bin/python tools/train_native_field_curriculum.py --config config/native-field-curriculum-phaseA-small.json --output-dir output/native-field-curriculum-2026-10-08
MPLCONFIGDIR=/tmp/native-curriculum-mpl .torch-cuda-env/bin/python tools/summarize_native_curriculum.py --output-dir output/native-field-curriculum-2026-10-08
.torch-cuda-env/bin/python -m pytest tests/test_native_curriculum.py tests/test_native_temporal.py tests/test_native_features.py tests/test_native_windows.py tests/test_native_field_step.py tests/test_native_state.py tests/test_fluid_history.py tests/test_marker_state.py tests/test_native_gauss.py tests/test_marker_feedback.py tests/test_source_module.py tests/test_field_propagator.py tests/test_step_dataset.py tests/test_output_contract.py -q
```

结果依次为：基线26 passed；新增7 passed；审计成功66记录/45窗口；两个 seed 各60更新成功；汇总成功；最终回归 **293 passed, 1 skipped / 40.41s**。跳过项是既有未交付 export_v3 用例，真实数据案例未跳过。新增测试包含独立常量场损失手算、真实两片段训练统计隔离、真实优化更新、变步长课程、逐位训练恢复、损坏加载原子性及全局 RNG 恢复。TDD 从缺失接口/实现的失败开始，未仅比较同一损失实现两次。另运行新文件 compileall 与 git diff --check。

可恢复指定 seed 的未完成边界：在同一 config/output 下加 `--seed 41 --resume-checkpoint output/native-field-curriculum-2026-10-08/seed-41/boundary.pt`；恢复会丢弃晚于边界的旧评估，按保存采样/RNG继续。此命令会重新执行边界之后的训练，**交付预算已完成，本轮未再次运行**。完成预算的 final 检查点由 load API 检查/推理，CLI 拒绝额外训练；实验配置不匹配在写产物前拒绝。

## 剩余问题与下一步条件

已验证训练更新、短程跨步反传、TF/AR机制和更新边界恢复；固定同 run 短程损失下降，但 AR 误差增长、跨 run 大误差和正 dt 连续极限缺口均保留。下一轮应先决定离散增量与小 dt 行为的支持域，再扩展独立时间/参数覆盖。物理 BC、真实非零反馈及其历史/marker恢复资格、闭合消费者和耦合评估仍需分别实现验收；不能由本轮训练替代。结论仅为**课程训练机制与短程 field-only 学习实测**。
