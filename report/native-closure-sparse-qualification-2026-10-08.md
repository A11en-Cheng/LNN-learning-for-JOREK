# 原生闭环入口续篇：稀疏空间路径与实际资格

本次使用 codebase-design、tdd；窗口资格词汇使用 domain-modeling。未训练、未运行优化器、未修改 JOREK、roadmap 或 ADR。保留旧实验及检查点。本报告是分切片交付，不是生产闭环完成证明。

## 实现与证据

- `NativeSourceMesh` 分离局部完整 cubic 求值与全局投影。支持 dense、sparse、none：dense 保留原小网格参考上限；sparse 按共享索引组装局部 16×16 一致质量块，CSC 稀疏分解；none 不构造全局矩阵。没有通过提高稠密上限支持真实网格。
- 稀疏求解采用对角对称均衡。初次未经均衡的真实网格常数系数恢复最大误差约 2.659e-4，不能通过验收；均衡后为 2.381e-9。小网格以保留的稠密参考独立对照，另用常数和坐标 R 的解析值/导数检查。不是仅重复同一求解。
- `TorchMarkerLoads` 用局部基函数与共享索引 scatter 产生弱式载荷，公式特征不依赖全局稠密基矩阵。给定源的场路径仍可微；NumPy 源事件没有变成端到端可微链。
- mesh 配置版本为 `native-source-mesh-v2`，载荷消费者为 `native-torch-marker-loads-v2_local_scatter`，记录投影后端、均衡、边界及资源限额。冻结电子容量仍仅接受 dense 参考模式，未恢复为主路径。旧配置不是精确续跑的新配置。
- 窗口 schema 升为 `native-causal-window-v2`，旧 schema 保留。`WindowSources` 保存五通道全轴、原始 projection deltas、独立两类密度率及时间/单位/来源；`RecoveredCoupledAnchor` 检查完整 ReferenceRollout 状态、历史、时钟、控制计划、配置和网格。资格分别为场监督、给定合法源、已注册小规模自生成参考。缺历史不补零；未经认证的相邻文件差分拒绝。
- 初始化与源历史支撑必须包含在 split 隔离后的完整窗口支撑内。每个被排除候选保存位置、支撑和具体原因。自生成资格不再硬编码为零，但仅支持已实现的小规模参考配置；不认证目标生产配置。
- 新 fixture 经公开源/场/事件接口实际连续推进两步；完整状态恢复检查包括新模型、重新构造的稀疏因子、云、投影历史、shard、记忆及时钟。真实数据不能因此获得参考 fixture 的资格。

## 有界真实数据验证

机器证据位于 `output/native-closure-entry-2026-10-08/`。

| 验证 | 实际结果与范围 |
| --- | --- |
| 两个真实网格 | 各 3986 单元、4125 节点、16419 唯一自由度；CSC 578583 非零，约 7,008,676 字节，LU 合计 7,619,650 非零 |
| 构造/因子耗时 | 两片段约 1.190 / 0.903 秒；求值、组装、求解约 0.412 / 0.391 秒 |
| 内存 | 稀疏诊断进程峰值 RSS 785384 KiB，含 Python/Torch，不是纯矩阵或完整云的增量内存 |
| v400 锚点候选 4110 | 一百万槽位、11168 活 marker，粒子载荷 172000000 字节；读取/验证约 1.453 秒 |
| inner 锚点候选 2710 | 一百万槽位、19 活 marker，同样 172000000 字节；读取/验证约 0.666 秒 |
| 初始化结果 | 两者均未获生产自生成闭环资格；具体字段状态见 `anchors-final.json` |
| 有界窗口审计 | v400 4108–4120、inner 2701–2708：train 2、validation 2、test 1 场监督窗口；条件源/自生成窗口均 0；6 个跨 restart 候选被排除 |
| 小规模稀疏两事件 | 未训练参考，约 0.037 秒，2691 参数；中途保存/新模型与新因子恢复逐位一致 |

v400 4110 仅依注册程序 restart 认证零投影历史及启动时钟，不能推广到普通记录。inner 2710 虽有粒子文件，真实 gather/residual 与更新前投影历史仍未认证。未从未来记录补初态，未执行百万槽位长回放。

## 源码确认事实与边界

只读源码 HEAD：`54c675bcc39c33b082c0c35046e28122dc53b3aa`。数据投影记录 RCS 为 `v2.25.12.0-1042-gb4165893f-dirty`，生产构建与当前源码完全对应尚缺证据。

| 位置 | 核查事实及影响 |
| --- | --- |
| `particles/examples/marker_spi_source.f90:98,146–148,384–437,547,670–675` | 非平衡分支、全部投影滤波参数、带拒绝采样的出生分布、辐射调用、按真实 gather 归一化。当前中心出生参考不是生产出生分布。 |
| `particles/projection_functions/mod_radiation.f90:52–76` | 此处 get_Lz 使用 PLT+PRB；当前 PLT-only 缺少实际需要的 PRB。不能无依据把 CX 全部设为本切片前置。 |
| `models/pellet_module.f90:464–479,624,684–725,864–889` | abl_model=3 与 rkt_model=6 是不同配置；nonlocal 查询磁面平均并参与烧蚀，rocket6 更新速度。现有小参考未实现后两者。 |
| `particles/diagnostics/mod_uncoupled_projection.f90:38–48,258–294,1010–1018,1070,1090–1130` | H=1 仍消费 n=0 滤波；原始 deltas 在写入新投影前由新旧值产生；含滤波和边界惩罚，边界 if/elseif 顺序不能简化为集合并集。 |
| `particles/diagnostics/mod_project_particles.f90:154,241–268,520–581` | 默认应用 Dirichlet，投影求解后输出系数。所测 HDF 是 float64 且无 scaleoffset；仍不足以单独认证生产更新前后对象、过滤阶段与构建一致。 |

本次 sparse 是**未滤波、无约束的一致质量投影参考**，不是生产 JOREK 投影。几何/全轴及单位保持，但不能宣称生产滤波、物理边界或九场 FEM 算子对齐。真实 marker 定位仍为逐单元参考搜索；未验证大云推进成本。

两个目标均 abl3、rocket6、nonlocal 开启，滤波非零；所测配置还发现 ns_radius_min 分别 0.05 / 0.01，不能称仅空间位置和步长不同。详见 `source-support.json`。

## 无训练尺度诊断及资格缺口

仅隔离原编码器绝对 eventJU 输入列进行诊断，两片段该列产生的 tanh 饱和比例均为 1。它不是填零历史后的完整真实源前向，也不是训练结果。本轮未擅自改变时间原点、尺度或检查点语义；完整真实源预激活诊断还需合法投影历史。

| 缺口 | 状态 |
| --- | --- |
| 共享 context、事件因果防线、显式下一控制步长 | 已修复，见上一切片报告与回归 |
| 完整局部求值/无全局矩阵特征/稀疏参考投影 | 已实现并验证；生产滤波/Dirichlet 未实现 |
| 真实初态清单、活粒子数、动态资格及拒绝原因 | 已实现；两生产锚点资格仍待证据 |
| 普通投影差分历史认证 | 待生产版本及输出链证据；未默认恢复 |
| rocket6、nonlocal、生产出生/核、PRB | 未实现；已查明实际分支，不要求所有 JOREK 配置 |
| 目标物理 BC/自由度约束 | 未实现；共享索引平均不能替代边界处理 |
| 真实配置两个完整自生成事件 | 尚不具资格，未运行，不用真值重置替代 |

可执行补采要求：在目标生产构建同一事件的源推进前、投影后、流体提交后分别输出事件相位、sim/time、粒子真实时间、step_rest_time、实际 gather、index_now、tstep/tstep_prev、完整 U/rawD/前场、完整 P/rawDP、marker 全状态及 shard；保存 namelist 与源码/构建标识，外部控制提供下一 dt。现有无 rawDP 输出必须补充保存或先用同构建确认过滤前后与旧新对象对应，再认证差分；不能跨 restart 推断。nonlocal 中间量可否从 OLD 场重建还需移植对应算法核验，不先认定为永久缺失。

## 测试与复现

逐切片记录：projection 参数未实现时失败→小网格稀疏通过；formula-only 缺全局基矩阵时失败→局部 scatter 通过；窗口双源/锚点类型缺失时失败→动态资格通过；零槽位虽空云但无法出生的例子暴露资格不足→增加资源/槽位守卫。非零源、历史、移动位置、梯度及恢复的已有回归保留。

```bash
.torch-cuda-env/bin/python -m pytest -q tests/test_native_conditional_windows.py tests/test_native_sparse_source.py tests/test_native_context_guard.py tests/test_native_checkpoint_control.py tests/test_native_joint_source.py tests/test_native_coupling.py tests/test_native_field_step.py tests/test_native_temporal.py tests/test_native_features.py tests/test_native_windows.py tests/test_native_state.py tests/test_marker_state.py tests/test_fluid_history.py tests/test_native_gauss.py tests/test_marker_feedback.py --junitxml=output/native-closure-entry-2026-10-08/final-regression-v2.xml
# 227 passed in 30.76s；最后补强锚点配置/单位检查后相关 2 tests 再通过。
PYTHONPATH=. .torch-cuda-env/bin/python tools/diagnose_native_sparse_source.py --output /tmp/native-real-sparse.json
PYTHONPATH=. .torch-cuda-env/bin/python tools/audit_native_source_anchors.py --output /tmp/native-anchor-inventory.json
PYTHONPATH=.:tests .torch-cuda-env/bin/python tools/diagnose_native_source_support.py --output /tmp/native-source-support.json
PYTHONPATH=.:tests .torch-cuda-env/bin/python tools/diagnose_native_joint_source.py --projection-backend sparse --output /tmp/native-sparse-two-events
PYTHONPATH=. .torch-cuda-env/bin/python tools/audit_native_windows.py --spec config/native-window-phaseA-audit.json --output-dir /tmp/native-window-v2-audit
```

前三个工具使用代码内显式登记的两个有界片段；实际机器结果分别为 `real-sparse-final.json`、`anchors-final.json`、`source-support.json`。manifest 已使用 Python jsonschema 对 v2 schema 验证通过。未运行课程 optimizer 测试。

本切片结论限于：可扩展参考空间路径和实际状态资格入口已补齐，小规模事件与续跑回归通过；生产配置闭环仍有上述具体缺口，无预测精度、稳定性或 C0 完成结论。
