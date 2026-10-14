# 切片 A：完整原生状态保真与检查点续跑（2026-10-08）

验收范围仅为**完整状态保真与续跑**。使用tdd，以既定原生状态/读取/ReferenceRollout/模型保存加载Interface逐切片红→绿；没有训练、容量矩阵、电荷闭合消费者或九场矩阵实现，没有宣称物理反馈接通或C0闭环完成。roadmap/JOREK未修改，已有用户改动保留。

依据：现行ADR-0018/0019/0020，以及[B2→C0设计](b2-c0-native-state-feedback-design-2026-10-08.md)。本轮纠正该设计中的泛化帧间隔描述，不改变ADR决策。

## 状态与兼容接口

- `src/models/native_state.py::NativeFieldState`：`values/raw_deltas=[9,4,H,N]`，原生浮点dtype，显式NativeUnits、SI时间、原生内部步号、完整mode计划及`native-field-v1`。独立保存原始fluid_deltas，不用端点差重建。
- `NativeFeedbackState`：`values/raw_deltas=[5,4,H,N]`、相同单位上下文、投影事件SI时间、history_provenance、可选真实gather_time及`native-feedback-v1`。历史来源不可省略；未知gather_time保持None，不用流体时间间隔冒充。
- `to_node_layout(history=...)` / `from_node_layout(...)`：可逆 `[field,degree,harmonic,node] ↔ [node,degree,harmonic,field]` Torch/NumPy副本，保留dtype及负零。属于参考/存储Adapter，反向调用会detach，不承诺训练梯度路径。`node_dof0()`是显式有损观察，不是重建入口。
- `NativeFeedbackState.to_projection_feedback()`完整接现有事件驱动，保存历史来源和gather_time。未改变冻结共享S的节点视图、现有FieldPropagator或source_proj未物理对齐标识。
- `NativeUnits`明确场/投影归一化约定、t_norm、n_norm、central_mass、gamma；能量归一化可由gamma无损计算。读入时校验文件中的时间/密度/质量归一化，不隐式把SI时间当JU。gamma等未完整导出的配置必须由调用者明确解析并声明，不能从潜特征猜测。

## 全轴读取和原始历史

`src/data/native_state.py`提供：

| 入口 | 必需输入及行为 |
|---|---|
| `read_native_field_state(path, units, modes)` | 显式单文件；核对jorek_model=502、cubic、n_tor、内部整数index；完整读取values/deltas；field t_now乘一次t_norm转SI。缺deltas失败，绝不减另一帧。 |
| `read_native_feedback_state(path, units, modes, ...)` | 投影t_now本身为SI；完整5场全部轴。文件有deltas则直接读取；无deltas必须显式提供已保存raw_deltas和历史来源。不会自动搜索相邻、前后或未来文件，也不保证外部历史仅凭来源字符串已被独立验证。 |
| `read_native_context(path, units, modes, configuration)` | 从原文件读取x、signed size、vertex、mode数量/plane/period/坐标配置、boundary/freeboundary及可用拓扑；调用者提供已解析physics/tables、numerics、model及BC选项。所有已声明配置内容参与指纹，数组含dtype/shape/content。路径名称不作为网格内容身份。 |

普通projection文件无deltas时，不提供零补或“任意相邻输出差分”的恢复选项。真实登记JOREK restart由已有初始化适配器明确产生零投影/零历史，来源标为`jorek_restart_zero_projection`；这与普通缺历史不同。程序restart不清流体node deltas。

原有`native_run`节点读取/StepDataset接口保留供兼容，不能借本轮声称它们已变成完整原生训练数据接口。调用者应显式使用新全轴入口；有mode轴不代表已支持非轴对称物理推进。

## 复用ReferenceRollout

没有新建调度器。`ReferenceRollout.with_native_context(context)`或`reference_rollout_from_checkpoint(..., native_context=context)`绑定完整原生契约；`native_field_state()`和`native_feedback_state()`返回类型化副本。持久的真实数组仍只存一份，避免包装状态与旧数组两份不同步。

严格native路径验证：九场/五场全部degree/harmonic轴、原始增量dtype/有限性、版本、前场端点、反馈与marker事件时间、流体端点时间、余留范围、正particle_dt、正流体计划及游标、前步长，以及声明的时间scheme。startup必须零反馈/零投影历史/零余留；连续路径缺前场或历史失败。旧降阶/不透明回调接口兼容保留；即使未绑定context，只要声明native_full，仍验证完整轴及投影历史，不能绕过缺历史拒绝规则。但未绑定context时不能调用类型化native状态入口，也不获得完整配置一致性保证。

沿现有驱动执行：已持有场端点→粒子/新反馈→流体候选（明确接原始D_u和时间系数）→在旧场更新shard→联合提交场/历史、shard、marker/反馈、时间、前dt和游标。提交前再次验证，任一步异常不返回candidate、不修改输入状态；这不覆盖回调对外部全局对象的任意副作用，回调仍须遵守无外部提交契约。

模型续跑不会调用jorek_restart或execute_startup；保存待startup状态时保留原标志，保存已执行startup状态时不会再执行。小型固定递推仅证明状态链，不是物理流体求解。

## 完整模型检查点

现有`save_model_rollout/load_model_rollout`升级`marker-rollout-v2`：完整rollout及原生schema、全部物理数组/marker槽/反馈/时钟/步长游标/残差记忆、模型state_dict、逐子Module运行模式、结构/extra_repr、已声明原生配置指纹、Python/NumPy/Torch CPU及已初始化CUDA的全局RNG状态。

native恢复必须传`native_context=expected_context`，比较单位、mode、网格及BC、physics/tables、numerics、model声明；不以记录里自己的配置与自己比较。配置不匹配、未知/缺失schema、模型结构或tensor dtype不匹配、缺历史均拒绝。模型state_dict先在私有模型上验证，避免PyTorch部分拷贝失败污染调用者；最终加载有权重/模式/RNG回滚。CPU已测试；CUDA恢复逻辑有设备数量检查，但本轮没有CUDA续跑数值测试，不宣称跨设备逐位一致。

保存使用同目录临时文件、flush/fsync和原子replace，验证或序列化失败不会替换原检查点，临时文件清理。是可信本地推理checkpoint；不含训练优化器语义，也不是不可信pickle加载器。兼容读取旧v1，旧文件不得被当作缺失native配置的完整新契约。

配置保证以调用者**明确声明**的manifest为范围：物理表应提供内容digest/版本，不能只写文件名；隐藏在自定义Python属性而未进入state_dict/extra_repr/manifest的配置无法自动发现。独立随机生成器需由调用者作为持久状态携带，全局RNG保存不能替代它。

## 当前phaseA逐步片段实测

通过`audit_native_segment`读取明确请求的文件及macroscopic表，仅用于诊断，不作为rollout推理输入。不读取后续场补齐历史。以下时间表四舍五入至足够辨别步长的JU；原始数值对比流体表容差1e-7 JU、投影/marker容差max(1e-14 s,t_norm*1e-7)，保存恢复本身用逐位比较。

| run片段 | 内部步号 | 流体t_now（JU） | H5 tstep（JU） | projection t_now/t_norm（JU） | marker /time/t_norm（JU） |
|---|---:|---:|---:|---:|---|
| v400 | 4108 | 8601.0 | .5 | 8601.0 | 文件不存在 |
| v400 | 4109 | 8601.5 | .5 | 8601.5 | 文件不存在 |
| v400 | 4110 | 8602.0 | .5 | 8601.5（全零） | 8601.5 |
| v400 | 4111（restart） | 8602.0 | .5（导入步长） | 8602.0 | 文件不存在 |
| v400 | 4112 | 8602.05 | .05 | 8602.05 | 文件不存在 |
| v400 | 4113 | 8602.10 | .05 | 8602.10 | 文件不存在 |
| v400 | 4114 | 8602.15 | .05 | 8602.15 | 文件不存在 |
| injInner | 2701 | 8372 | 5 | 8372 | 文件不存在 |
| injInner | 2702 | 8377 | 5 | 8377 | 文件不存在 |
| injInner | 2703 | 8382 | 5 | 8377（全零） | 文件不存在 |
| injInner | 2704（restart） | 8382 | 5 | 8382 | 文件不存在 |
| injInner | 2705 | 8387 | 5 | 8387 | 文件不存在 |
| injInner | 2706 | 8392 | 5 | 8392 | 文件不存在 |
| injInner | 2707 | 8397 | 5 | 8397 | 文件不存在 |

两个片段内部index_now与文件请求步号一致；流体按步输出，普通时间间隔匹配步长，只有表中登记restart重复时间。v400原始重复时间存在约3.3e-11 JU舍入差，不能按严格float相等判断。macroscopic表精度也与H5不同。全部流体shape=(9,4,1,4125)，原始D_u完整读取。projection时间字段为SI，与流体JU分开校验。marker稀疏文件不能补零成每步marker初态。

这一结论限定上述片段，不把全run扫描或全部其他数据版本声称为已验证。已改设计报告的泛化“跨多流体步”说法；即便逐步输出，舍入/约束意味着D_u仍独立保存。restart登记JSON的旧meta中“退化source窗口”措辞不作物理依据，沿ADR-0020和已有源码报告解释正时长粒子窗口。

## 测试证据与命令

`tests/test_native_state.py`：完整3-harmonic/4-degree逐位Torch roundtrip（含负零）、相同dof0不同高阶状态、原始历史缺失失败、显式投影历史来源、单位拒绝、原生时钟/shape/schema校验、不同配置组及模型dtype拒绝；固定非平凡增量递推在两种dt下直接执行和序列化执行一致；混合子Module模式/RNG恢复；shard失败输入不变、保存中途pickle失败旧文件不变、坏权重加载模型/RNG不变；真实两run字段读取及v400百万槽restart完整保存恢复（不推进百万粒子）。

时间递推fixture是逐系数`D_new=.5*D_old+.25*dt`，初始D=.125，dt=.5→D=.1875，再dt=1→D=.34375，独立字面期望；不是u+1事件顺序测试。marker事件也只是确定性状态fixture，不是物理反馈闭环。真实字段用独立HDF5读取逐字节对照。

| 命令（仓库既有环境） | 结果 |
|---|---|
| `.torch-cuda-env/bin/python -m pytest tests/test_fluid_history.py tests/test_marker_state.py -q`（改动前） | 20 passed / 3.94s |
| `.torch-cuda-env/bin/python -m pytest tests/test_native_state.py -q`（tdd） | 状态入口缺失1 failed→1 passed；读取入口缺失1 failed/1 passed→2 passed；context入口缺失及真实字段名修正后→3 passed；续跑keyword缺失→4 passed；RNG失败→5 passed；audit/context入口缺失2 failed→7 passed；未知schema未拒绝1 failed→9 passed；反馈metadata Adapter缺失1 failed→11 passed；startup零规则缺失1 failed→12 passed；dtype损失未拒绝1 failed→补拒绝 |
| `.torch-cuda-env/bin/python -m pytest tests/test_native_state.py tests/test_fluid_history.py tests/test_marker_state.py -q`（中途冻结代码） | 31 passed / 6.70s |
| `.torch-cuda-env/bin/python -m pytest tests/test_native_state.py tests/test_fluid_history.py tests/test_marker_state.py tests/test_native_gauss.py tests/test_marker_feedback.py tests/test_source_module.py tests/test_field_propagator.py tests/test_step_dataset.py tests/test_output_contract.py -q`（首轮扩大回归） | 222 passed, 1 skipped / 21.60s |
| `.torch-cuda-env/bin/python -m pytest tests/test_native_state.py -q`（兼容路径补测） | 1 failed, 13 passed / 6.01s：未绑定context的native_full漏验投影历史；随后补基础全轴/历史校验 |
| 上述9个文件的扩大回归命令（最终冻结代码） | **223 passed, 1 skipped / 19.54s**；新增14项均执行，真实读取/百万槽保存恢复未skip |
| `.torch-cuda-env/bin/python -m compileall -q src/models/native_state.py src/data/native_state.py src/models/marker_state.py src/data/marker_checkpoint.py tests/test_native_state.py`；`git diff --check` | exit0 |

诊断命令使用`.torch-cuda-env/bin/python -`，显式对v400 `range(4108,4115)`和injInner `range(2701,2708)`调用`audit_native_segment(run, indices, restart_events=registry[run]['restart_events'])`；exit0，输出为上表。初次探索请求不存在的`part04109.h5`曾报FileNotFoundError；随后诊断明确登记其缺失，没有将其补零或当作可用初始化点。

本轮所有验收仍限“完整状态保真与续跑”；完整反馈响应、容量/压力/电阻率/交换消费者与C0闭环均未实现或验证。
