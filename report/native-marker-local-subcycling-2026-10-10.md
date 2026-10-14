# marker 局部细分实现与边界未通过项（2026-10-10）

后续外边界末段处理与最终独立复验见 [续篇](native-marker-boundary-exit-2026-10-10.md)。下文保留初版及未通过案例，不能把后续版本结果倒写到旧代码证据中。

结论：**已实现显式、非默认的局部细分输运；敏感存活粒子的运动与投影对照明显改善，但 injInner 完整云第一窗仍未通过，不能切换正式训练或宣称整体修复完成。** v400 两个给定场源窗口与同臂恢复通过；含源九场 Torch 双事件、跨步反传及模型保存/恢复在解析 fixture 通过。未训练、未更新 optimizer、未改 JOREK/原始数据/训练产物/roadmap；已有用户修改保留。

使用 `diagnosing-bugs` 复现原49步红灯并建立真实入口回归；`codebase-design` 将调度和历史封装在共享推进 seam，避免复制电离/沉积公式；`domain-modeling` 登记 [ADR-0023](../docs/adr/0023-explicit-marker-local-subcycling.md) 与独立数值历史词汇。复用了此前源码核查子代理，完成两组初态覆盖审计、injInner 完整窗口尝试及状态消费者核查；主代理独立复现失败并核验外边界拓扑。

## 1. 接入和不变项

- [marker_transport.py](../src/models/marker_transport.py)：`LocalSubcycling`、`TransportHistory`、显式原生初态初始化、二进制时间调度、变间隔方向推进。
- [marker_state.py](../src/models/marker_state.py)：`MarkerRuntime.transport_history`；共享 `advance_window` 的每个接受步仍执行电荷更新、旧位置五通道沉积、运动、速度历史、定位。不是只细分位置。
- [native_coupling.py](../src/models/native_coupling.py)：CPU `batched_v1` 的实际 `NativeMarkerSource.advance` 可显式安装 policy；由当前完整 cubic 值/梯度得到筛选指标，无粒子编号白名单、无未来场或对照臂输入。
- [native_real_chain_builder.py](../tools/native_real_chain_builder.py)、[profile_native_coupled_latency.py](../tools/profile_native_coupled_latency.py)：`marker_transport=local_subcycling_v1` / `--marker-transport local_subcycling_v1`；默认 `native_fixed`。旧 checkpoint 与新配置不得混用。

源扩散公式、原始梯度、`ni` 截非负及 `1e12 m^-3` 分母下限均未改变；未限幅速度、关扩散、放宽定位或按异常编号绕过。核对只读 JOREK `particles/examples/marker_spi_source.f90:500–519,625–667` 和 `particles/pushers/mod_fieldline_euler.f90:56–78`：旧位置沉积 → flow＋扩散推进 → nearby 定位；固定步长使用当前速度模长乘当前/前方向的 `1.5/-0.5` 组合。

## 2. 数值约定与历史

定义因果无量纲指标

`χ = base_dt × (D_ju/t_norm) × |grad_ni_raw|² / max(max(ni_raw,0),1e12)²`。

当前 policy：`χ>0.1` 触发；最少64细步，按单细步指标不超过1继续细分，最多1024细步；每窗最多100万次细分粒子步，另受原 wall/RSS/chunk 预算约束。历史 level 对该 lifetime 只升不降，避免在密度极值处梯度趋零后立即恢复粗步。**该筛选不检测完整 Hessian、E×B 或平行输运刚性，不是局部误差估计、稳定性证明或普遍精度保证。**

每粒子保存上次实际接受间隔 `h_prev`；新间隔 `h` 的方向外推系数为 `1+h/(2h_prev)` 与 `-h/(2h_prev)`，仍乘当前速率。等间隔显式调用原 pusher，普通未触发粒子保持原结果；不能将此称为完整速度的变步 AB2 或普遍二阶积分器。

时间调度用整数二进制 ticks，不以累加浮点子时钟决定结束；每个存活粒子到原 base 终点或丢失。电离能量按实际电荷增量，辐射与三个粒子矩按实际 `h` 累积；最终仍除原窗口 `n×particle_dt` 的 gather。事件时钟、残余时间、窗口数和源/流体/shard 顺序不变。

历史包含 float64 前间隔、细分等级、lifetime 与 policy/import 声明；按槽复用重置新身份，保留旧云所有原生数组。原生 HDF 的 `v` 不记录独立前采样间隔，首次导入显式声明原固定间隔；新生 `v` 沿用已有出生时刻采样约定。模型续跑保存完整历史；缺失、dtype/level/import/config 错误明确拒绝。新增历史不能静默转成原生 JOREK restart。compact birth-only 重建入口明确拒绝新模式，不丢历史后冒充完整响应。组件摘要、状态差异报告与精确 digest 纳入新增历史。

## 3. 原红灯与存活粒子对照

真实 injInner 捕获保留原场/rawD/ADAS/几何/初始 `V_prev`；可信本地 pickle 禁止外部输入。原49子步例本轮再次运行退出1：速度增量跨定位相对 L2 `4.500260446e-4`。最终新模式退出0、代码哈希不变：`1.366805337e-5`；`1e-4` 仍只是原诊断红灯，不是生产 gate。

22个原始持久粒子完整81子步对照，走实际 source 入口及完整五通道稀疏/滤波投影；这是**选中子云投影，非整个真实云投影**。

| 对照（同原初态） | R/Z 位移相对 L2 | 速度增量相对 L2 | 原子权重速度增量相对 L2 | 五通道投影相对 L2 上界 |
|---|---:|---:|---:|---:|
| 原定位局部模式 vs nearby 局部模式 | 5.88e-9 | 1.20e-5 | 1.19e-5 | 1.24e-8 |
| nearby 最少128细步参考 vs 最少64细步 | 1.94e-13 | 9.44e-12 | 9.37e-12 | 1.59e-6 |

后一行电离通道约 `5.16e-8`、辐射0、三个矩约 `1.20e-6/1.11e-6/1.59e-6`；电荷向量相对 L2 `6.55e-11`。两种128/64模式仍会自适应升到128/256/512细步，不能将其误称为全程固定64 vs 固定128。共同身份全部匹配。

度量为 `||candidate-reference||₂ / ||reference||₂`；本窗增量各减自己的起点；权重范数乘 `sqrt(atoms)`。R/Z 与 phi 分开，不把米和弧度混合当物理距离。投影与 deltas 逐通道比较原生 JOREK 储存量，不做额外 SI 缩放；零参考按现有明确 undefined/全零约定报告，不用 epsilon 分母。

## 4. 完整云覆盖与 v400 两窗

初态只读覆盖：v400 11,168初活中0触发，`χmax=0.00144720`；injInner 52,604初活中5触发（0.009505%），`χmax=11.3727`，槽 `500558/973/400633/900629/600505`。初态均没有命中负密度或分母 floor。这个比例不是整个推进过程中触发率；22持久粒子诊断没有覆盖原运行丢失粒子，不能由它推断完整云资格。

v400 百万槽、实际出生、两个给定记录场源窗口4111/4112，两臂独立携带云/投影历史/RNG，并分别序列化恢复第二窗，共六次 source advance。最终36.77s、RSS 2.81 GiB；未裁剪云，代码和原始文件不变。两臂各自连续/恢复完整状态逐值一致；离散事件/RNG 相同。跨定位速度本窗增量相对 L2 `3.29e-10/1.44e-9`，五通道投影分别不超过 `1.30e-9/2.59e-9`。

这是**给定记录场的源过程验证，不是 NN 预测—源—场—shard 的真实自生成闭环**；v400 没有证明触发细分的完整云对照。完整 Torch 耦合接线在实际触发细分的解析 fixture 通过，NumPy source/shard 的 stop-gradient 不变。

## 5. injInner 外边界失败，必须保留

完整云试验第一窗失败，窗口/resume 完成数为0，候选不提交。最终只读观察独立复现同一失败，代码/输入文件不变；未提高阈值、减少粒子或绕过该身份。

- 旧粒子 slot973、lifetime1，非本窗新生；时间 `0.004113560398902941 s`，元素3377，`st=(0.99858916266,0.16800825293)`。
- `ni=5.66086e14 m^-3`，**尚未命中1e12 floor**；原始梯度约 `(9.15918e19,-3.32986e15) m^-4`，密度长度尺度 `ni/|grad_ni|≈6.18 μm`。
- 当前速度约 `(-1.98002e6,755.127,0.679) m/s`；`χ≈1615.97`，请求level11即2048细步，超过level10。
- 捕获真实拓扑的邻居 `[3376,0,3378,3361]`，s=1一侧确为外部零邻居，边上两个节点 boundary code 为2；同t的边坐标约 `(0.462647,0.15462634525)m`。当前R仅比边高约6.18 μm；该R差不是严格法向距离或精确退出时刻。

这是外边界的陡密度下降/退出情形，不能用三个主要存活误差粒子的成功替代它，也不能声称只要更细就普遍稳定。**后续需定义并验证外边界穿越/终止事件与最后一段电离、辐射和沉积时间权重**；定位失败还需与确认域外区分。不能把超过上限的步直接夹到1024、看到locator失败就当成功退出，或无限追逐边界零密度。

## 6. 验证、复现与成本

最终相关回归 **290 passed in 23.24s**（25文件；一条既有 profiler cycle 提示）。覆盖普通路径逐值不变、真实源公式触发、接受步电荷/旧位置沉积、丢失和出生身份、变间隔方向、配置拒绝、预算及场时间守卫、模型恢复、实际触发的双事件 Torch BPTT、真实 Fortran nearby 与已有训练执行接口；无 optimizer 更新。编译与差异检查通过。新增测试见 [test_marker_local_subcycling.py](../tests/test_marker_local_subcycling.py)。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .torch-cuda-env/bin/python -m tools.diagnose_native_nearby_motion --capture output/native-nearby-motion-2026-10-10/capture-inner.pickle --slot 600505 --steps 49 --local-subcycling --assert-consistent --output /tmp/local49-new.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .torch-cuda-env/bin/python -m tools.validate_marker_subcycling_capture --capture output/native-nearby-motion-2026-10-10/capture-inner.pickle --output /tmp/local-selected-new.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .torch-cuda-env/bin/python -m tools.validate_native_nearby_history --run-index 0 --marker-transport local_subcycling_v1 --output /tmp/local-v400-new.json
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .torch-cuda-env/bin/python -m tools.diagnose_marker_subcycling_limit --output /tmp/local-inner-limit-new.json
```

输出均独占创建，复跑使用新路径。产物位于 [output/native-marker-local-subcycling-2026-10-10](../output/native-marker-local-subcycling-2026-10-10/)：`original-red49.json`、`local-green49-guards.json`、`selected-refinement.json`、`selector-coverage.json`、`v400-full-two-window-final.json`、`inner-limit-final.json`、`inner-limit-final-failure-detail.json`、`inner-boundary-geometry.json`。保留原失败和早期试跑；`local-green49.json` 在编辑期间执行，不作为最终代码哈希证据。selected-refinement 数值执行于后续dtype/摘要守卫之前，文件保存该执行版本哈希；未改变其数值算法。

新模式有真实成本：22粒子64模式实际约70,996粒子步，其中69,457次细分、69,138个 callback group；两种定位验证约53/60s，128参考约60s。这些是共享主机并发验证计时，**不是受控性能 benchmark**，不计算加速比，也不以小触发率推断开销小。后续在边界语义通过后，重新测完整耦合 CPU/GPU 分布并优化细分调度/小批量沉积，才考虑默认启用或训练。
