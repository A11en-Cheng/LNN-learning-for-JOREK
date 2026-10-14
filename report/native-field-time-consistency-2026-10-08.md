# 原生 field-only 正步长一致性修复与同预算对照（2026-10-08）

## 范围与复现

范围始终为 `field_only_no_feedback_consumer`。保留完整 `[9,4,1,N]` U/rawD、同场值/梯度、已确认分组参数、空间残差 stencil、官方 ncps 1.0.1 default CfC 及显式记忆。没有修改 JOREK、物理事件顺序、roadmap 或已接受 ADR，没有增加反馈消费者。

使用 diagnosing-bugs 构造公开 NativeFieldStep 接口的失败测试，再用 tdd 修复。固定合法 U、rawD、前步长及非零记忆，设置非零 decoder bias=0.7，对 seed 3/41/73 比较 dt=1e-4/1e-6/1e-8。修复前增量范数在 dt=1e-8 仍分别约13.1478/13.6363/13.3818；三个测试均因增量不随 dt 收缩失败（2.25秒）。复现命令：

```bash
.torch-cuda-env/bin/python -m pytest tests/test_native_temporal.py -k positive_time_limit -q
```

排查假设：①步增量解码缺时间因子；②直接提交 CfC 候选记忆造成跳变；③时间格式特征奇异。实际 Gears 特征 `zeta=dt/(dt+prev_dt)`、`delta_scale=dt/prev_dt` 在固定正 prev_dt 下有限并趋零，排除③。官方 cell default 的候选在 dt→0 通常不等于输入记忆；直接赋值加上独立步增量解码解释①②。独立零权重 cell 案例候选为0，旧记忆为1，单独证实记忆问题。未改官方 cell 公式，也未冻结记忆。

## 时间一致性适配契约

[TimeConsistentResidualAdapter](../src/models/native_temporal.py) 是**外层学习时间适配器**，不是官方 CfC 原公式或 JOREK 精确积分器。[NativeFieldStep](../src/models/native_field_step.py) 返回接口仍是本步已应用原生增量，内部 decoder 的含义变成无量纲速率。

令 `h=dt_SI/t_norm_SI`（JU），`q=h/τ_JU`，`c=CfC(x,m,q)`：

- `m_next = m + [-expm1(-q)] (c-m)`；decoder 实际读取推进后的 `m_next`。
- `r = decoder(m_next)`，无量纲速率。
- `ΔU_candidate = history_scale * r * q`；经原生共享DOF/支持约束映射后，`U_next=U+ΔU_applied`。
- 对支持的自由更新行，正步长 `rawD_next=ΔU_applied`，直接保存计算增量，不从已舍入 U 相减重建。冻结行沿已有 JOREK 更新规则保留历史。

τ 复用核心已有 `time_scale_ju` 配置，本实验固定 **1 JU**，没有搜索或拟合。rawD 的 field/degree 分组 history_scale 仍仅由训练窗口唯一支撑拟合；它是每参考时间的速率尺度基准，并非宣称对真实速率方差重新拟合。原生速率尺度为 `history_scale/τ_JU`（原生系数/JU），SI速率为 `history_scale/(τ_JU*t_norm_SI)`。Gauss值/梯度统计和参数路由不变。无时间步数据统计泄漏。

若候选与 decoder 在固定有限输入下有有限极限，则门控为 `q+O(q²)`，所以 `ΔU=O(h)`、`Δm=O(h)`，局部变化率有限。记忆正步长仍可变化且反传，不以冻结解决问题。空间层原有 dt 控制的是空间特征算子；这里只对新定义的学习速率积分一次，未对任何已积分物理载荷再次乘 dt。

精确0保留已有独立 no-op：U/rawD/记忆/时钟均不变，不调用正步长物理事件。**正步长 rawD 趋于0，与精确0保持旧 rawD 是两条不同规则**；原始历史是上一步增量，不要求它自身在这种固定历史的探针中连续。普通预测和 chunk 不触发 restart。没有要求严格半群或据局部一致性推出刚性/长期稳定。

## 版本与恢复

模型配置从 `native_field_step=v1` 改为 `v2_rate_memory`，新增 output_semantics，绑定 `native-rate-memory-v2`、速率定义、τ、场/记忆因子及尺度来源；官方 core 配置及源码哈希未变。完整 TorchFieldState 仍保存相同布局，但配置摘要包含新语义。模型/训练检查点均按完整配置核对，旧步增量权重不能自动作为速率恢复。

没有提供通用迁移：即使 decoder 除一个参考步长，记忆更新与后续输入轨迹也改变，不能恢复等价变步长行为。本轮按要求从头训练；旧产物完整保留。模型状态字段及物理更新顺序不变；domain-modeling 仅给 CONTEXT 增补学习速率与增量/载荷的区别，不修改 ADR 决策。

## 测试证据

- 三 seed 正步长极限、有限局部速率，非零记忆与解码偏置；修复后全部通过。
- 独立常速率/记忆手算：τ=2 JU、r=0.7、history_scale=3，1 JU 增量=1.05；零候选记忆从1变为 exp(-0.5)=0.6065306597126334。两半步在这个特例满足手算，但最后rawD是半步增量，不能误等同整步历史。
- 原有SI/JU等价、变步长/三时间格式的独立权重案例更新为速率积分口径；不是删除旧断言来忽略差异。
- 第二步损失回传第一步 U/记忆、初始记忆和核心参数；显式切点后梯度消失；TF的 U/rawD 配对和实际输入重新求梯度继续通过。
- 新模型连续/恢复逐位一致、旧语义/不同τ拒绝且不污染模型；训练优化器/RNG/采样与课程边界恢复回归继续通过。
- 新 subdivision 探针对同一初态连续推进1/2/4/8子步，传递预测 rawD/记忆和更新前步长，记录差异而不要求一般模型严格半群。
- 两真实 phaseA 完整轴多步反向测试、特征/窗口/状态/事件回归保留，未跳过真实案例。

定向回归46 passed / 13.14秒；完整相关回归 **299 passed, 1 skipped / 43.74秒**（既有 export_v3 缺失跳过）。compileall、git diff --check 通过。新增临时诊断日志未留入模型。

## 同预算实验设计

沿用[旧实验](native-field-curriculum-small-2026-10-08.md)的66记录、45窗口、训练唯一支撑32记录，warm-up=1、horizon=1/2/4、stride=4、隔离7；同 run 时间验证与跨 run 验证分开。新配置只增加 `model.output_semantics=rate_memory_v2`；原配置/产物保留。统一训练统计重新按同一支撑计算并在比较工具中核对逐项相等。

seed41/73，各60更新含冒烟：1次H1冒烟、19次H1 TF、20次H2混合p=.5、20次H4 AR。optimizer/scheduler/loss/模型宽度均不变。中间20更新实际保存、新建模型恢复。固定评估0/1/20/40/60，test仅最终；不选优、不追加搜索。反馈仍为None，无补零。跨 run injInner dt=5 JU，相对训练v400 dt=.05 JU相差100倍，不能称纯参数泛化。

新产物：[目录](../output/native-field-time-consistent-2026-10-08/)，旧产物：[目录](../output/native-field-curriculum-2026-10-08/)。比较工具另检查数据/statistics/参数路由、评估支撑、实际采样窗口与教师mask一致，防止把预算相同误称实验条件相同。

## 对照结果

两个 seed 均完成60更新，无 NaN，所有参数梯度有限且每次 optimizer 产生非零权重更新。冒烟同窗口 loss：seed41 0.240658→0.240619；seed73 0.241008→0.240962。每个组/horizon仍只评估固定排序前两个窗口，不代表全run覆盖。

### seed 41：最终固定窗口平均加权损失

| 组 / K | 旧TF | 新TF | 旧AR | 新AR |
|---|---:|---:|---:|---:|
| train / 1 | 0.195402 | 0.185631 | 0.195402 | 0.185631 |
| train / 2 | 0.176585 | 0.167530 | 0.248252 | 0.239869 |
| train / 4 | 0.156024 | 0.146992 | 0.405772 | 0.389694 |
| same_run_validation / 1 | 0.018711 | 0.008155 | 0.018711 | 0.008155 |
| same_run_validation / 2 | 0.018570 | 0.007949 | 0.022191 | 0.009242 |
| same_run_validation / 4 | 0.019600 | 0.007728 | 0.038830 | 0.013133 |
| cross_run_validation / 1 | 7.273459 | 215.019823 | 7.273459 | 215.019823 |
| cross_run_validation / 2 | 4.992801 | 213.284169 | 5.100310 | 319.079936 |
| cross_run_validation / 4 | 3.317325 | 212.082334 | 3.613990 | 619.780221 |
| test / 1 | 0.017037 | 0.006370 | 0.017037 | 0.006370 |
| test / 2 | 0.017038 | 0.006237 | 0.020141 | 0.006690 |
| test / 4 | 0.018278 | 0.006077 | 0.038693 | 0.007717 |

### seed 73：最终固定窗口平均加权损失

| 组 / K | 旧TF | 新TF | 旧AR | 新AR |
|---|---:|---:|---:|---:|
| train / 1 | 0.203974 | 0.185678 | 0.203974 | 0.185678 |
| train / 2 | 0.181411 | 0.167574 | 0.257242 | 0.239939 |
| train / 4 | 0.159331 | 0.147059 | 0.407522 | 0.389859 |
| same_run_validation / 1 | 0.019178 | 0.008186 | 0.019178 | 0.008186 |
| same_run_validation / 2 | 0.018635 | 0.007971 | 0.023163 | 0.009261 |
| same_run_validation / 4 | 0.018661 | 0.007767 | 0.038179 | 0.013121 |
| cross_run_validation / 1 | 7.289849 | 271.725639 | 7.289849 | 271.725639 |
| cross_run_validation / 2 | 5.064189 | 269.351351 | 5.310533 | 476.467185 |
| cross_run_validation / 4 | 3.381748 | 268.596095 | 4.346418 | 741.167104 |
| test / 1 | 0.017772 | 0.006401 | 0.017772 | 0.006401 |
| test / 2 | 0.017383 | 0.006262 | 0.022001 | 0.006715 |
| test / 4 | 0.017535 | 0.006123 | 0.035783 | 0.007738 |

同 run 新版 K4 初始→最终 AR：seed41 0.018625→0.013133；seed73 0.020604→0.013121。旧版分别0.984819→0.038830、6.407276→0.038179。新版未训练时已因时间适配缩小输出，比旧版初始损失低很多；不能把最终差异全部解释为学到了更准确的动力学。新版seed73 TF从更新40的0.007754到60的0.007767略升，保留固定最终检查点。

固定同 run K4 AR 的逐步累计误差（相对时刻0.05/0.10/0.15/0.20 JU）：

| offset | seed41旧 | seed41新 | seed73旧 | seed73新 |
|---:|---:|---:|---:|---:|
| 1 | 0.018711 | 0.008155 | 0.019178 | 0.008186 |
| 2 | 0.025672 | 0.010330 | 0.027147 | 0.010337 |
| 3 | 0.042873 | 0.014152 | 0.042077 | 0.014103 |
| 4 | 0.068063 | 0.019894 | 0.064314 | 0.019858 |

同 run 短程 TF/AR 改善，但 AR 误差依然随步数增长。**跨 run K4 AR 严重退化：3.613990→619.780221、4.346418→741.167104。**该组dt=5 JU，训练为0.05 JU；速率外层在大步长不再具有旧解码的幅度限制，且时间/历史分布改变。这里登记观察与合理结构解释，不把它证明为唯一因果或通过数据驱动调参掩盖；没有追加搜索或预算。局部趋零修复不保证大步长保真/稳定。

新版最终同 run K4 AR 四项加权贡献（coeff/rawD/value/gradient）：seed41 0.004873/0.006809/0.001051/0.000400；seed73 0.004856/0.006832/0.001034/0.000399。完整逐field/degree、Gauss值/RZ梯度归一化误差在[field-degree-errors.csv](../output/native-field-time-consistent-2026-10-08/field-degree-errors.csv)与[误差图](../output/native-field-time-consistent-2026-10-08/field-only-degree-errors.png)；逐时刻原生单位平方误差与各项未加权/加权值在 seed evaluations.json。所有 horizon/dt/步长比桶仍独立登记，损失步比值均为1，不能宣称真实数据已覆盖一般变步长预测。

![同预算固定评估](../output/native-field-time-consistent-2026-10-08/time-contract-comparison.png)

## dt 极限和细分实测

每个模型固定自身 warm-up 后合法状态、历史、记忆、参数、prev_dt。旧/新训练后的 warm状态可能不同，故下表是各自局部诊断；严格固定同一输入的前后机制证据来自失败回归测试。

| seed / 版本 | h=0.05 JU 增量scale RMS | h=5e-8 JU 增量scale RMS | 小h记忆变化 RMS | 小h增量局部率/JU | 小h记忆局部率/JU |
|---|---:|---:|---:|---:|---:|
| 41旧 | 0.201685 | 0.202767 | 0.348204 | 随h缩小发散 | 随h缩小发散 |
| 41新 | 0.007997 | 7.761732e-9 | 2.706026e-8 | 0.155235 | 0.541205 |
| 73旧 | 0.191580 | 0.192149 | 0.288592 | 随h缩小发散 | 随h缩小发散 |
| 73新 | 0.006825 | 6.812756e-9 | 2.405097e-8 | 0.136255 | 0.481019 |

两个 seed 新版场/记忆变化都随正 h 收缩，局部率收敛为有限非零值。精确0两项严格为0且保留原历史/时钟；记忆并非冻结。最低h约对应2.04e-14秒，远小于本轮记录步长，仅作数值/契约诊断。

![正步长极限](../output/native-field-time-consistent-2026-10-08/field-only-dt-limit.png)

同一初态总时长0.05 JU，8子步相对1步的场差异（state_scale RMS）seed41=8.72572e-5，seed73=8.95308e-5；记忆差异为0.005802/0.005265。1→8子步的最后rawD RMS分别0.007997→0.001001、0.006825→0.000853，因其表示最后子步增量，不能要求与整步历史相等。完整1/2/4/8诊断在各seed subdivision.json；这些差异反映重求值、记忆、前步长及原始历史共同改变，不是严格半群或收敛阶证明。

## 成本、检查点与复现

模型参数仍3732，CfC1800，没有新可训练参数。CPU float64单线程，Torch2.11.0+cu130，核心版本/源码哈希与旧版一致。新训练前向+反向+更新18.87/17.85秒，旧18.77/18.06秒；新固定评估44.87/41.28秒；新各seed主循环含恢复/评估66.66/61.96秒。新进程峰值RSS2,245,040 KiB，旧2,233,120/2,237,616 KiB；RSS为累计进程高水位且包含数据/检查点副本，时间存在运行噪声。不作JOREK加速结论。

两seed在更新20真实保存并新建模型/optimizer恢复。最终检查点另重新加载，实际旧训练final加载均明确 `training configuration mismatch`，拒绝后模型、采样器和更新号不变；审计无optimizer更新：[checkpoint-verification.json](../output/native-field-time-consistent-2026-10-08/checkpoint-verification.json)。训练与推理配置拒绝路径、完整连续恢复一致性由小型确定性测试覆盖。

新最终检查点：[seed41](../output/native-field-time-consistent-2026-10-08/seed-41/final.pt)、[seed73](../output/native-field-time-consistent-2026-10-08/seed-73/final.pt)；边界检查点、完整曲线、manifest、统计与参数路由均在新目录。旧目录未覆盖。[comparison.json](../output/native-field-time-consistent-2026-10-08/comparison.json)逐项确认两臂数据/统计/参数路由相同、实际窗口和teacher masks相同、每seed每臂60更新。

复现命令（仓库根目录；预算已经完成，训练命令记录用途，再执行会产生新训练预算）：

```bash
.torch-cuda-env/bin/python -m pytest tests/test_native_temporal.py -k positive_time_limit -q
.torch-cuda-env/bin/python -m pytest tests/test_native_temporal.py tests/test_native_field_step.py tests/test_native_curriculum.py -q
.torch-cuda-env/bin/python tools/train_native_field_curriculum.py --config config/native-field-curriculum-time-consistent.json --output-dir output/native-field-time-consistent-2026-10-08 --audit-only
.torch-cuda-env/bin/python tools/train_native_field_curriculum.py --config config/native-field-curriculum-time-consistent.json --output-dir output/native-field-time-consistent-2026-10-08
MPLCONFIGDIR=/tmp/native-curriculum-mpl .torch-cuda-env/bin/python tools/summarize_native_curriculum.py --output-dir output/native-field-time-consistent-2026-10-08
MPLCONFIGDIR=/tmp/native-curriculum-mpl .torch-cuda-env/bin/python tools/compare_native_time_contract.py --baseline output/native-field-curriculum-2026-10-08 --new output/native-field-time-consistent-2026-10-08
.torch-cuda-env/bin/python tools/audit_native_time_checkpoints.py --baseline output/native-field-curriculum-2026-10-08 --new output/native-field-time-consistent-2026-10-08
.torch-cuda-env/bin/python -m pytest tests/test_native_temporal.py tests/test_native_features.py tests/test_native_windows.py tests/test_native_field_step.py tests/test_native_state.py tests/test_fluid_history.py tests/test_marker_state.py tests/test_native_gauss.py tests/test_marker_feedback.py tests/test_source_module.py tests/test_field_propagator.py tests/test_step_dataset.py tests/test_output_contract.py tests/test_native_curriculum.py -q
```

结果：复现修复前3 failed→修复后3 passed；定向46 passed；审计66记录/45窗口；训练两个seed各60成功；汇总/比较/实际检查点审计成功；全相关299 passed、1 skipped。compileall覆盖本轮修改的模型/数据/评估/工具/测试文件，git diff --check通过。

验收限于**原生field-only正步长局部一致性修复及同预算课程对照实测**。未解决物理BC、缺失反馈/marker耦合资格、空间池化/stencil学习近似和大步长外推失败，不宣称反馈接通、长时间稳定、预测有效或C0完成。
