# 无真实场回授的完整云冒烟（2026-10-09）

## 本切片结论

两个 phaseA 各完成两个连续、完整的源—九场—shard 事件；没有真实未来 Ψ 或其它场回授。
这仍是 **未训练网络的流程冒烟**，不是预测精度或长期稳定验收。后续训练结果另记。

- 初态使用注册 poststartup 的完整原生 U/rawD、100 万 marker 槽、原始零投影和全部 shard。
- 后续全部九场来自残差学习场步；源采样、OLD 场 shard 更新、非局部诊断均消费当前预测状态。
- 磁轴、两个 X 点、边界磁通由当前 cubic Ψ 重算。没有冻结真实 ES，也没有搜索失败后回退真值。
- 时间控制仍是场文件内的已记录 schedule，不宣称网络自主选择步长。
- 真值场/投影在完整事件提交后才打开计算指标，不替换状态。

自主诊断的源码依据与支持域见 [专门报告](native-autonomous-equilibrium-evidence-2026-10-09.md)。
数据及候选 split 见 [审计报告](native-training-pilot-data-audit-2026-10-09.md)。

## 实测

|运行|初始活 marker|连续事件|总 wall/s|首事件粒子推进次数|首事件输运与投影/s|观测峰值 RSS/GB|
|---|---:|---|---:|---:|---:|---:|
|v400/4110|11,168|4112、4113|90.796|447,880|10.411|1.907|
|injInner/5680|52,604|5682、5683|120.510|4,265,327|70.637|1.908|

两进程并行运行，耗时并非独占资源 benchmark。源第一事件分别执行 40/81 个真实粒子子步，第二事件执行 4/16 个；没有减小 dt 或裁云。
v400 出生 29、2 个 marker；inner 出生 56、11 个。完整云包含损失、位置与身份状态，不只比较新生增量。

机器结果：

- `output/native-autonomous-smoke-2026-10-09/v400-run2/result.json`
- `output/native-autonomous-smoke-2026-10-09/injInner-run2/result.json`

每事件另存 `event-1.json` / `event-2.json`，包括逐场四 degree、Gauss 值/梯度、rawD、五通道投影与 shard 位置的相对 L2/Pearson。
归一化仅为指标自身相对范数；物理输入与误差在原生 JOREK 单位中计算，不先换算为 SI 场量。
零范数/常量相关系数保持未定义状态，不加 epsilon 掩盖。

作为**未训练**基线，第二事件 Gauss 相对 L2：

|场|v400|inner|
|---|---:|---:|
|Ψ|0.00013|0.00092|
|Φ|0.66490|0.86901|
|Z_j|0.00236|0.01963|
|ω|0.94942|0.77938|
|ρ|0.00128|0.00450|
|T_i|0.00021|0.00219|
|v_parallel|0.00320|0.00549|
|ρ_imp|0.00240|0.00468|
|T_e|0.00029|0.00327|

短时段静态场相对误差小不意味着动力学已学会；Φ/ω 的误差尤其明显。
第二事件投影五通道相对 L2：v400 约 `0.05995/0.02878/0.0008846/0.002395/0.010007`；
inner 约 `0.13968/0.05068/0.001402/0.002428/0.005779`。
两步 shard 坐标误差为零（当前短步及记录精度下），不能外推为长期轨迹精确。

## 批量源路径

新增显式 `spatial_backend='batched_v1'`：完整 cubic 值/梯度批量求值、保持粒子顺序的载荷 scatter、同单元 Newton 批量定位。
跨单元或快速定位未收敛的粒子仍走原 scalar 搜索。只批处理执行，不改消融、电离、辐射、速度、粒子步长或投影矩阵。
默认 scalar 及旧配置指纹保留；启用 batch 会进入新的配置字段，不能静默恢复不同配置检查点。

两个真实网格各抽取 512 个已有活 marker，仅用于数值等价审计而非裁云推进：
值、梯度、几何、基函数、Jacobian 与 scalar 相对 L2 全为 0；对 R/Z 各移动 `1e-7 m` 后，定位单元和局部坐标也完全一致。
公共完整源事件、解析多点、重复 deposit 和域外定位专项均通过；不宣称任意布局下始终逐位一致。

## 验证与复现

```bash
timeout 900 .torch-cuda-env/bin/python -m tools.smoke_native_autonomous \
  --run data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400 \
  --anchor 4110 --output-dir output/NEW-EMPTY-v400-autonomous

timeout 900 .torch-cuda-env/bin/python -m tools.smoke_native_autonomous \
  --run data/EHL-2/phaseA_dataset_1d22_ang15_massloss_neon_v400_injInner \
  --anchor 5680 --output-dir output/NEW-EMPTY-inner-autonomous
```

采用外层 900 秒 timeout、源内部 900 秒/8192 MiB 限额、2048 marker/chunk；超限不更改物理配置。
最初两个 `v400`/`injInner` 目录是主线程在发现指标字段命名不匹配后主动中断的尝试，没有作为成功证据；修正 `RecordedNode.object_times_si` 后重新执行 `*-run2`。

相关回归两批 **58 passed** 和 **80 passed**；分别覆盖新增自主/批量入口及既有生产源、联合场、CfC、指标和训练序列接口。`git diff --check` 通过。
使用 TDD 在既有 provider / 完整源推进 / real builder 公共接口先记录缺失能力的失败，再实现并复验。

本报告不声称端到端源/shard 可微：NumPy 源/shard/ES 仍 stop-gradient；训练可反传的是 Torch 联合场/CfC 路径。
生产 MPI 出生流、历史 rank-lag nonlocal 差异及 learned-open-boundary 近似未被此冒烟消除。
