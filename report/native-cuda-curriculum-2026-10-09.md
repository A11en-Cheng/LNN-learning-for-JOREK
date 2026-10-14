# CUDA 原生耦合课程与逐场指标（2026-10-09）

## 范围与执行状态

用户要求继续训练、尽量利用 RTX 5070 Ti，并提供逐场 loss 和相对误差。
本轮不恢复任何未来真实场（含 Ψ）的回授；真实节点仅用于监督和指标。
初态仍需读取认证的完整真实状态。源、shard、平衡诊断使用预测场；
控制沿用显式记录步长回放，不能称为自主控制预测。

有界执行目录：`output/native-cuda-curriculum-2026-10-09/seed1-run3/`。
执行结果以该目录 `result.json` 为准：已正常完成，`success=true`，新增 6 次更新，3 个小集合 epoch。
前两次尝试 `seed1/`、`seed1-run2/` 均在验证初态读取失败，优化更新为 0，产物保留。

上一轮只有两个训练窗口各一次更新，不是全部 phaseA 的一个 epoch。
本轮预算为 **3 个小集合 epoch，每 epoch 遍历两个声明的训练窗口，共新增 6 次更新**，
监督 rollout 长度依次为 1、2、2 个完整 base 事件。沿用上一轮 `update-2.pt` 权重和 Adam 状态。
不遍历整个 phaseA，不追加搜索、不裁剪真实云、不为提高 GPU 利用率修改物理步长或扩大模型。

### 与正式训练方案的差距（用户追问后的明确界定）

当前序列 batch size 为 **1**，两个窗口串行、各自立即更新一次；没有滑窗 Dataset/DataLoader，
也没有跨窗口梯度累积或 GPU 批处理。所谓 epoch 只是重复这两个固定窗口。
`[1,2,2]` 是运行前写定的预测长度，不是逐轮观察结果后临时调参。
本轮不构成“确定正式 burn-in → 全训练区间切窗 → mini-batch 训练”的完成证明。
尤其本轮临时选 B=0 后，实际执行为两步完整无梯度预测前缀、重置记忆、再监督 H 步；
这与保留 B 步 warm-up 记忆的正式 burn-in 窗口并不等价。

正式扩展应先冻结以下契约，再增加训练量：

1. 先按物理序列划定 train/val/test，再在各自区间内构造“完整初态/因果支撑＋B步 warm-up＋H步预测”。
   不跨 restart，整个支撑不跨 split；缺少窗口初态的历史/云不能补零。
2. warm-up 不监督、不反传但保留其记忆和物理状态，预测段不读未来真实场；
   B 候选应在固定预测起点和时长下单独验证，不与本轮记忆重置实验混称。
3. 同一训练阶段固定 B、H、窗口清单与采样规则；一个 epoch 明确定义为遍历该训练窗口清单。
   单步到多步的切换是阶段变化，不以不等长平均 loss 拼成一条收敛曲线。
4. 分阶段计时 CPU source/birth/projection/shard/ES、传输、GPU forward/backward；
   先对冻结权重的独立窗口验证 CPU-worker/GPU-owner 流水化，再扩展训练。
   同一窗口仍必须等待其预测场与状态提交；不得跨权重更新复用预测缓存。

这些是下一步工作范围，**本轮未实现正式滑窗批训练或流水化**。

## 固定预测节点与记忆恢复对照

两个 run 的 train 锚点为 4110/5680，val 为 5870/6440，test 为 6870/7220。
test 本轮不计算指标。读取扩展目标时检查内部索引、实际时间与注册 restart；
训练/验证沿用原先隔离的完整支撑区间。

每个窗口用当前权重推进两个完整事件，形成 C0/C1/C2 的完整状态与 RNG 快照。
以 C2 对应的固定时刻为预测起点，比较：

- 连续记忆：从 C2 直接预测，不重置记忆。
- B=0：C2 仅重置 CfC 记忆，随后预测。
- B=1：C1 仅重置记忆，完整推进一次后在同刻预测。
- B=2：C0 仅重置记忆，完整推进两次后在同刻预测。

历史场、rawD、marker 云、shard 和源历史均不清零，也不装入中途真值。
B=1/2 的冷启动会改变后续物理状态，所以这不是“同一预测初态、只比较隐藏状态”的实验，
更不是最优物理历史长度或 BPTT 长度的证明。连续记忆是必要的参照。
仅按两个 val 窗口的两步平均 loss 暂选记忆恢复设置；1e-6 相对平局取较短设置。

完整前缀绑定权重/配置指纹，每次权重更新后重新计算，禁止沿用旧权重的预测源缓存。
若 C0 记忆本来为零，B=2 可复用本次刚计算的同权重前缀；这是同一条实际执行路径，
不是省略物理事件。证明字段明确记录复用与状态差异。
训练前后评估使用相同 seed、同一节点、同一时长；epoch 间训练 RNG 不同，曲线另行标注。

## 指标定义

验证对照已完成：v400 连续/B0/B1/B2 loss 分别为
25.1164634718 / 25.1164635054 / 25.1164634990 / 25.1164634718；
injInner 分别为 6.79874116338 / 6.79874112037 / 6.79874112053 / 6.79874116338。
差异远小于 1e-6 相对平局容差，临时设置取 B=0。
这是当前近初始权重下对记忆重置不敏感的结果，不能据此宣布正式 burn-in 为 0。

九场顺序：`psi, Phi, Z_j, omega, rho, T_i, v_par, rho_imp, T_e`。

逐场 loss 是该场对总 loss 的实际贡献，四组权重均为 1：完整原生系数、
rawD、Gauss 场值、兼容 R/Z 梯度。使用仅两个 train 初态拟合的缩放，
不是物理单位 MSE，也不是把不同量纲原始误差相加。
完整 JSON 同时保留各组归一化误差和原生单位平方 MSE。

主要相对误差直接在 JOREK 单位下计算：
`sqrt(sum(w*(pred-truth)^2)/sum(w*truth^2))`，Gauss 权重为 `R*abs(J)*quadrature`。
零参考范数返回未定义及原因，不以 epsilon 伪造有限相对误差。
Pearson r 另报，不是 R²；常量场相关度未定义。
全系数/历史按 degree、梯度按方向、marker 五通道投影与 shard 身份/位置指标，
保留在逐事件 JSON，九场概览不能替代这些细项。
独立 shard density 没有导出的监督标签，其指标显式为 `missing`；
双源进入执行链不等于双源都已得到定量验收。

`per-field.csv` 保存每个事件的逐场 loss、相对 L2、Pearson r。
epoch 图使用相同第一预测节点，避免将 epoch1 一步和 epoch2/3 两步均值直接作趋势比较。
固定训练前后表另报两步平均 loss 与第二步相对 L2。
本轮预测节点比上一轮 pilot 更靠后，禁止将两轮总 loss 直接当成收敛曲线。

## CUDA 与资源解释

Torch 场网络、CfC、Gauss 求值、loss 与反向传播使用 CUDA FP64；模型为 3147 个参数。
源/出生/shard/部分投影与平衡诊断仍为 CPU NumPy/SciPy 路径，源与 shard 停止梯度，
不能宣称完整端到端可微。
完整真实云不裁剪，源采样使用现有批量后端。

默认沙箱拒绝访问 GPU；授权的 CUDA 执行实际识别 RTX 5070 Ti。
真实 CUDA 两事件/跨设备 checkpoint 专项通过（1 passed）；不能据此宣称整轮课程中断恢复通过。
记录 Torch 自身显存峰值、场 forward CUDA stream 时间、backward 墙钟时间与源 advance 墙钟时间。
`source_seconds` 不包含出生，不能当作完整 CPU 成本。
NVML 每 2 秒采样为整卡指标，包含其它进程，不能归因于本训练独占利用率。

epoch 保存参数/优化器以及独立 controller JSON；旧 pilot cursor 固定为 0、不表示课程进度。
本轮没有实现并验证课程 CLI 自动断点续训，恢复必须同时解释 controller 和完整前缀重建边界。

## 验证初态的 restart 时间舍入

injInner 6441 的保存历史仅在已登记的 6440→6441 边界发生微小倒退：
10358.00000000064 → 10358.000000000215 JOREK 时间，
差为 −4.256435204297304e-10 JOREK 时间，即 −1.7390837508587155e-16 s。
6440 以前的 `xtime/R_axis_t/Z_axis_t` 与前帧逐位一致，末项与当前场 `t_now` 严格相等；
无负时间或非有限值。LCFS 初始记录和选择未改变。

只在读取器明确收到注册 restart 索引时，允许既有 1e-14 s 事件时钟容差内的该类舍入；
普通倒退、大于容差的倒退、未注册边界仍拒绝。原始时间数组不排序、不改写，
实际允许的边界与 SI 差值写入 provenance；默认读取保持严格。
JOREK HEAD `54c675bcc39c33b082c0c35046e28122dc53b3aa` 的
`core/mod_jorek_timestepping.f90:328` 启动时由 `sim%time/sim%t_norm` 重建 `t_now`，
与所见舍入相符；这不是对生产二进制逐指令执行的证明。
`models/equil_info.f90:1295` 使用初始化历史前缀作 LCFS 判断。

使用 diagnosing-bugs skill：先复现真实失败、检查原始时间与注册边界，再增加拒绝/允许对照测试，
只修复有证据的读取容差，不更换验证窗口或放松普通时钟检查。

## 复现与测试

### 本轮实测结果

固定两步任务的训练前后总 loss：

| run | 更新前 | 更新后 |
|---|---:|---:|
| v400 | 0.204668470870366 | 0.20466842056721984 |
| injInner | 1.3076675370444812 | 1.307572936649437 |

两者都只有极小变化，不能宣称收敛、预测有效或泛化通过。
下表 loss 为更新后两步平均的逐场贡献，误差为第二预测节点的 Gauss 相对 L2（百分比）。
loss 与相对误差的聚合不同，不能要求它们逐项同向变化。

| 场 | v400 loss贡献 | v400 相对L2 % | injInner loss贡献 | injInner 相对L2 % |
|---|---:|---:|---:|---:|
| psi | 1.45984e-4 | 0.019727 | 0.00438580 | 0.130049 |
| Phi | 0.0508134 | 208.841 | 0.493304 | 167.758 |
| Z_j | 0.00316787 | 0.609533 | 0.107174 | 4.80588 |
| omega | 0.149467 | 169.419 | 0.632940 | 150.822 |
| rho | 6.88127e-6 | 0.227988 | 0.0153330 | 0.672028 |
| T_i | 4.72460e-5 | 0.0325939 | 0.00331141 | 0.319185 |
| v_par | 0.000985735 | 0.662946 | 0.00189422 | 0.869803 |
| rho_imp | 7.35444e-6 | 0.429392 | 0.0153922 | 0.698069 |
| T_e | 2.69062e-5 | 0.0416664 | 0.0338376 | 0.446108 |

Phi/omega 的相对误差明显过大；其它场的总状态误差较小，不能代替场增量/历史/梯度质量判断。
完整数值见 [field-summary.json](../output/native-cuda-curriculum-2026-10-09/seed1-run3/field-summary.json)、
[逐事件逐场 CSV](../output/native-cuda-curriculum-2026-10-09/seed1-run3/per-field.csv)、
[loss 曲线](../output/native-cuda-curriculum-2026-10-09/seed1-run3/per-field-epoch-losses.png)、
[相对误差曲线](../output/native-cuda-curriculum-2026-10-09/seed1-run3/per-field-epoch-errors.png)。
曲线展示训练时的第一预测节点，与上表的固定复评第二节点不同。

资源实测覆盖构建、验证对照、训练与前后评估，不是纯优化器阶段：

- 总墙钟 2188.70 s（36.48 min）。
- CPU `source.advance` 1156.12 s；出生/shard/ES/构建/指标等未完整分项，不能全部归到该项。
- 60 次场 forward 墙钟共 6.34 s，CUDA stream 计时 6.33 s；6 次 backward 墙钟共 0.509 s。
- 本进程 Torch 峰值分配显存 606982656 bytes（约 579 MiB）。
- 整卡采样利用率平均 9.53%，P95 18%，最大 61%，含其它进程；没有实现 GPU 跑满。

瓶颈证据更支持“CPU 物理链串行等待＋极小序列 batch/模型”，不是 GPU 算力不足。
流水并行的收益尚未测量，不承诺简单增大 batch 即可达到高利用率。
本轮结束后未继续增加 epoch 或启动下一项训练。

```bash
timeout 2400 .torch-cuda-env/bin/python -m tools.run_native_cuda_curriculum \
  --output-dir output/native-cuda-curriculum-2026-10-09/seed1-run3
.torch-cuda-env/bin/python -m tools.report_native_cuda_curriculum \
  output/native-cuda-curriculum-2026-10-09/seed1-run3
```

输出目录要求不存在；再次运行请使用新目录，不覆盖本轮记录。

- 读取器、autonomous entry、equilibrium：26 passed，1 xfailed。
- memory burn-in、pilot、source batch、base training、coupled metrics：34 passed，1 skipped（CPU 环境的 CUDA 项）。
- 实际 CUDA 两事件/跨设备 checkpoint 专项：1 passed。
- 逐场报告字段/节点及未定义指标保留：2 passed。

未修改 roadmap、ADR 决策、JOREK 或原始数据，保留用户既有修改。
