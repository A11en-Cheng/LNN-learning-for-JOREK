# 两个 phaseA 的闭环初态、历史与窗口认证（2026-10-08）

本轮使用 research 组织一手证据，读取 AGENTS、ADR README、roadmap_v2、ADR-0018～0022 与指定入口/sparse 报告及两份 JSON。只读原始数据和 JOREK，没有训练、启动生产 JOREK、改导出代码、裁剪 marker 或覆盖已有修改。未改变数据加载 Interface、模型或领域契约；新增的是独立研究审计脚本及结果。资格依据是现行 ADR，旧 ADR 不作门槛。

结论：**v400 4110 的注册 restart 初态成立，下一记录回放控制与配置材料已绑定；injInner 2710 的普通时钟/投影历史仍待证；找到并核验更强的 injInner 5680 注册 restart 替代初态。** 两个 restart 的核心初态认证不等于完整生产配置同构、模型支持或完整窗口认证。

## 一手证据和可重现产物

- [生产源码与构建证据](native-closure-production-provenance-2026-10-08.md)：按投影更新/输出、滤波与边界、marker 时钟、restart、shard、辐射六子系统分开记录事实、差异与缺证。生产投影 RCS 为 `v2.25.12.0-1042-gb4165893f-dirty`；当前检查源码 HEAD 为 `54c675bcc39c33b082c0c35046e28122dc53b3aa`，本地缺生产 commit 对象。不同 HEAD 未被当作相关逻辑不同，相同 dirty 字符串也未被当作源码等同。
- [最终锚点清单](../output/native-closure-certification-2026-10-08/certified/anchors.json)：三个完整快照、所有所用场文件、投影与控制见证；每项初态字段登记 `saved / verified_reconstructed / pending_evidence / unavailable`、来源、算法、容差及事件相位。本轮未执行完整状态因果回放，故没有字段冒用 `causal_replayed`。
- [最终窗口清单](../output/native-closure-certification-2026-10-08/certified/windows.json)：逐窗口保存 split、初始化/历史/回放/预热/预测支撑、条件源标签及四类资格/拒绝原因。目录根、`final/` 与 `reviewed/` 是迭代中的研究快照；**本报告计数只取 certified/**，目录名表示本次认证结论定稿，不表示所有候选合格。
- [独立时钟算术见证及 2710 候选](../output/native-closure-certification-2026-10-08/clock-witnesses-final.json)：直接解析后段生产日志 9291–9293 的子步数/residual，并与投影事件时间交叉核验；包括普通历史恢复条件表。

所有锚点所用 HDF5、marker、namelist、时间表均登记字节数和 SHA256。保存完整 namelist 原文摘要与解析值、场 HDF5 的全部 scalar 参数；解析值不是完整生效配置证明，尤其数组与未保存默认值仍需生产构建绑定。未来场/投影只标为控制回放见证或标签，不进入物理初态恢复。

## 逐锚点结果

| 项目 | v400 4110 | injInner 2710 | injInner 替代 5680 |
|---|---|---|---|
| 边界 | 注册 restart 4111 前 | 普通 after-fluid/before-particle 候选 | 注册 restart 5681 前 |
| 完整场/rawD/前场/shard/marker | saved，读取校验通过 | saved，读取校验通过 | saved，读取校验通过 |
| 槽数 / 活 marker | 1,000,000 / 11,168 | 1,000,000 / 19 | 1,000,000 / 52,604 |
| 初始 feedback/history | registered zero，verified_reconstructed | 前投影 values saved；rawDP pending_evidence | registered zero，verified_reconstructed |
| 启动/普通时钟 | verified_reconstructed，startup_pending | pending_evidence | verified_reconstructed，startup_pending |
| 导入 tstep / 上一流体步（JU） | 0.5 / 0.5 | 5 / 5 | 1 / 1 |
| 下一控制（JU） | 0.05，recorded_schedule_replay | 5，recorded_schedule_replay | 0.2，recorded_schedule_replay |
| 初始 marker 子步（秒） | 5.107219032439791e-9 | 候选 5.007077482784109e-9，未认证 | 5.044166945619547e-9 |
| residual | restart 初始为 0 | 候选约 1.0079692072326174e-19，未认证 | restart 初始为 0 |
| 下一粒子事件实际 gather | pending_evidence，不拿流体 dt 替代 | pending_evidence | pending_evidence |
| 完整生效配置/生产同构 | pending_evidence | pending_evidence | pending_evidence |

两个 restart 通过 `initialize_native_checkpoint` 读取全场与全部槽位；marker 内部时间匹配前场，零投影内部时间匹配 marker 事件时间，注册重复流体时间通过断言。零 rawDP 是已接受的 restart 初始化重建规则，绝非“保存的 rawDP”。不再笼统说 4110 所有投影历史和时钟缺失。

5680 是本次实际读取认证的替代初态，属于**更晚物理阶段**，不能用来证明或替代 2710 的普通初态。可见 injInner 最早 part 为 2710，2703 没有匹配快照；没有 part 不能说明空云，2704 零投影也不能说明空云。因此未宣称已有注入前空云锚点。

模型限制单独报告：两个 restart 的活 marker 都超过旧参考 10,000 上限，这是资源/Adapter 支持问题，不是快照不完整。三个初态的 million-slot/P_imp/v/identity 均保留；没有挑选 19 或 10,000 粒子伪造另一个同名状态。

## 控制、时钟与配置绑定

直接读取每个所检场的 `tstep/index_now/t_now/t_norm`，核对内部 index 与时间表，普通时间差与保存 tstep 相符；时间容差为 1e-7 JU，SI 比较使用 1e-14 秒下限。没有读取 namelist 步长数组作为控制，也没有恒步长回退。

4111 与 5681 是重复流体时间输出，保存的 tstep 仍分别为导入的 0.5、1 JU，**不能把它们当下一调度控制**。从首个正区间 4111→4112、5681→5682 与末端保存 tstep 双重核验得到 0.05、0.2 JU；2710→2711 同样核验为 5 JU。机器清单保存后续逐事件控制及各自 witness index；restart startup 的首个控制按现行调度语义绑定首段控制，生产实现同构仍单列 pending。

这些计划是 `recorded_schedule_replay`，可以组织无训练记录事件复核；不是外部自主预测时已知的控制。所有 `autonomous_input_eligible=false`，未收到 `external_control`。首个 startup 求解中的 `tstep_prev` 按下一控制设置，不能把导入步或零时间重复帧差当成它；普通上一流体步、marker 子步、residual、gather 分开登记。

配置已绑定完整 namelist 文件哈希及保存 HDF5 scalar，但无法凭现有资料把编译默认值、dirty 源码、ADAS 文件及全部运行时改动补齐为完整生效配置。源分支实际为 abl3、rocket6、nonlocal；滤波非零；ns_radius_min 两 run 不同。配置缺证与模型未实现这些分支分别进入数据/模型资格，不相互代替。

## 2710 的因果时钟与投影历史恢复

只读当前源码 `marker_spi_source.f90:350–355` 的规则是：真实粒子起点 `sim_time-residual`；按 `int((target-start)/particle_dt)` 取整数子步；新 residual 保留；实际 gather 为子步数乘粒子步。标称粒子步不能直接代替导入步整除调整后的实际步。

独立案例使用 injInner `logfile_spitest:10937–10974,15025–15063,19114–19152` 的 9291–9293：日志给出完整精度的 particle_step_time、n_steps、residual，复算子步数精确相等，residual 差小于 1e-22 秒，投影输出时间配对差小于 1e-14 秒。首个事件 16 子步，后两个 8 子步。打印精度仅 8 位的 target_time 未被用于取整。验证范围仅此后段实际日志，不跨段推定生产 dirty 一致。

从注册 reset2703/事件2704 到 projections2709，用此前保存投影时间与导入 tstep 算出 2710 时钟候选，末端与 part02710 的内部 `/time` 相符；没有使用 2710 之后真值补历史。它仍依赖：早期生效 nominal substep 配置、生产时钟/调度源码、投影时间的输出变换、全部更新/调度事件无遗漏。后段日志未覆盖 2704–2709，浮点取整对微小时间差敏感，故候选保留 pending_evidence，未送入普通初始化器，也未运行百万槽完整物理回放。

普通投影 rawDP 候选规则为 **本次更新后的 P 减实际上一次更新前的 P**，不是任意相邻文件差。当前源码 `mod_uncoupled_projection.f90:258–294` 支持该赋值顺序；但还需生产对应代码、同一节点对象/布局、连续实际更新、无重复输出/漏更新/restart、滤波与边界在两个操作数中的阶段、无损输出变换及 marker/projection 事件配对。HDF float64/no-scaleoffset 只解决其中精度子条件。

本次独立可核验的 rawDP 特例仅限注册 restart：原始零 values 见证 + 接受的清零规则 + 初始化输出零 values/rawDP 的实际断言。未找到目标普通段的独立 saved rawDP 或 update-level witness，不能验证普通差分规则后推广；**没有 deltas 文件不等于原则上不可恢复**。2710 的 P2709/P2708 不作未经认证的差分输入，past-zero 特例也不推广到任意普通窗口。

## 窗口资格与隔离

新增清单是明确有限范围的**资格研究清单，不是已可训练 manifest**。先定义 train/validation/test blocks，再从锚点构造每个候选的完整支撑；初始化前场/投影历史、anchor、startup/回放、一个 warmup transition、两个预测 transition 全计入隔离。未来条件源投影只作为条件/监督标签列出，不补初始化历史。

实检范围：v400 4108–4120、injInner 2708–2717 与 5678–5687；三个锚点共 **14 个候选窗口，8 个完整支撑处于同 split，6 个隔离拒绝**。包括局部目标位于 test 的 4117/4118 候选：虽目标在 test，仍因初始化和回放来自 train 被拒绝。所有候选逐条保留拒绝原因，数量由机器实际结果累加。

| 资格 | 合格窗口数 | 原因/范围 |
|---|---:|---|
| 数据状态完整生产资格 | 0 | 生效配置/生产对应关系待证；后移窗口尚无因果回放状态；2710 另有普通时钟/rawDP 待证 |
| 模型生产分支支持 | 0 | rocket6、nonlocal、出生/核、滤波/约束、PRB、物理 BC；restart 大云另有资源上限 |
| 记录回放控制来源 | 14 | 控制逐事件有内部记录见证；只说明来源资格，隔离失败不会被它豁免 |
| 自主外部控制来源 | 0 | 未提供 external_control |
| 合并条件源窗口 | 0 | 标签存在不等于初始化历史/完整支撑已认证 |
| 合并自生成闭环窗口 | 0 | 数据、模型、控制与隔离缺口分别保留 |

两个 restart **核心初态数量为 2**，与窗口合格数 0 是不同对象。后移窗口的因果状态尚未生成不能冒标 causal_replayed；未拿目标期记录重置 marker/feedback/场来凑窗口数。模型支持缺口没有被改写成原始数据损坏。

## 验证与需要补充的证据

```bash
PYTHONPATH=. .torch-cuda-env/bin/python tools/certify_phase_a_closure_evidence.py --output-dir <新的输出目录>
PYTHONPATH=. .torch-cuda-env/bin/python tools/check_phase_a_clock_witnesses.py --output <新的JSON路径>
.torch-cuda-env/bin/python -m pytest -q tests/test_native_checkpoint_control.py tests/test_native_context_guard.py tests/test_fluid_history.py tests/test_marker_feedback.py
# 77 passed in 1.29s；无训练/optimizer 测试。
```

精确需补的**现有材料**见[生产证据报告最后一节](native-closure-production-provenance-2026-10-08.md#精确需要补充的现有材料无需重跑生产)：生产 b416 的完整提交/源码与 tracked dirty diff、构建中 untracked/generated 文件；对应二进制 SHA256/编译宏与 run-segment 对应记录；覆盖 4109–4112、2703–2711、5679–5682 的原始事件日志与配置变更；实际 ADAS 文件校验和。若要自主闭环资格，还须外部控制；记录回放计划已单独组装。现有资料不足之处已定位，本轮不自行重跑生产或修改导出。
