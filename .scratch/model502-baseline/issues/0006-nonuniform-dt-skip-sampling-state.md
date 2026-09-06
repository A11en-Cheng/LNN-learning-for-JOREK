# 0006 非均匀 dt + 跳采样 + 状态连续性

Blocked by: #0001, #0005

交付：每步 `dt=t_{t+1}-t_t`（含每帧 `tstep`）从 trainer 传入 CfC 的 timespan 与 PDE loss；
跳采样 `(s_t, s_{t+k})`（`k=1,2,4,8,...`）以真实 Δt 训练，不重采样；隐藏状态跨步传递、BPTT 段边界 detach；
修掉 trainer 每步丢弃 state 的行为。

验收：变 dt 前向/反向无 NaN；dt 契约测试；两段 rollout 与一段完整 rollout 状态/输出一致（容差内）。

