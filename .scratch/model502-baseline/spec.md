# Spec：Model 502 九场单体基线与含源扩展（new_version Phase 0–2）

## Problem Statement

当前管线在 d5a44f0 已把 schema、encoder/decoder/loss 迁到九场，但训练闭环仍有三处断裂：
数据契约沿用自定义假设而非 JOREK/marker_spi_source 的真实接口；训练用固定 `dt=1.0`，与实测非均匀时间步
（耦合阶段帧间 dt 0.45–50）不符；零维参数与源项从未输入网络。因此无法给出可信的无源单体基线，也无法进入
含源项训练。

## Solution

按 gated 路线图收敛：先锁数据契约与参数通道（Phase 0），打通无源九场单体基线并过量化 gate（Phase 1），
再接入 marker_spi_source 真实源项做二维含源训练（Phase 2）。网络学习目标永远是原生九场；源项与零维参数
只作为控制输入；hyper-* 高阶截断由网络隐式吸收。

## Seams（测试挂点）

唯一高层 seam 是一步残差契约：`x_{t+1} = x_t + delta_x`，decoder 只输出 `delta_x [N,9]`；
processor 的输入是 `(latent, controls, time_steps)` 并返回 `(output, state)`。所有 ticket 的测试都挂在这条
seam 上，不新增第二套状态传递路径。

## User Stories

1. 作为数据工程师，我要 schema fixture 锁死真实 HDF5 形状（`values (9,4,1,N)`、`x (2,4,1,N)`）与谐波维语义，
   这样 parser 无法静默破坏契约。
2. 作为数据工程师，我要序列级 train/val/test manifest（附数据版本/hash），且统计量只由训练 split 计算，
   这样任何 run 都可在固定随机种子下复现。
3. 作为数据工程师，我要协议 v2 以 marker_spi_source 真实输出为准，并为每个字段标注"原生/导出/控制"，
   这样网络不会把导出量当学习目标。
4. 作为物理研究员，我要 mod_elt_matrix(_fft).f90 的参量面审计，列出每个零维参数与闭包并标注
   "已导出/需导出/由网络吸收"，这样条件集收敛到最小基底。
5. 作为数据工程师，我要 parser 读取全部零维参数并带 mask（缺失=0 且 mask=false，禁止 1.0 回退冒充），
   这样 condition 数据永远可溯真伪。
6. 作为 ML 工程师，我要 trainer 把真实每步 dt 传入 CfC 与 PDE loss，并支持跳采样 `(s_t, s_{t+k})`
   多尺度训练，这样网络能拟合非均匀时间演化。
7. 作为 ML 工程师，我要隐藏状态跨步传递并有明确的 BPTT 截断边界，这样两段 rollout 与一段完整 rollout
   在容差内一致。
8. 作为 ML 工程师，我要固定种子单样本过拟合 loss 下降 ≥99% 且无 NaN、reload 可复现，
   这样网络容量底线被证明。
9. 作为 ML 工程师，我要 checkpointing 梯度等价 ≤1e-5 与峰值显存报告（含并发占用），
   这样 hidden_dim=64 不会 OOM。
10. 作为研究员，我要 free-run rollout 逐场误差与质量漂移，至少一个 horizon 优于 persistence，
    这样无源单体基线成为 MoE 的对照。
11. 作为研究员，我要源项按训练集统计归一化后经 control_encoder 注入，无源帧拒绝零填充，
    这样强制训练与无源基线可公平比较。
12. 作为研究员，我要条件消融：固定状态改变零维参数时输出/latent 有可检测响应，且 no-condition 对照成立，
    这样条件信息被证明有用。
13. 作为研究员，我要含源 rollout 在强源项下无 NaN 且相对无源基线可解释改善，
    这样二维含源训练闭环被证明。
14. 作为 JOREK 开发者，我要在 mod_export_restart.f90 补导出 `ZK_i_par/ZK_e_par`、`D_perp` 基准等缺失标量，
    这样参数通道不再依赖辨识兜底。

## Out of scope

MoE 与梯度 firewall（Phase 3，gate 后）；ntor11 多环向与 3D 频谱；完整九方程 charge-state closure；
JOREK runtime 算子替代；hyper-* 系数的辨识。

