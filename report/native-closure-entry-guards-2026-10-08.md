# 闭环入口完善：P0与显式步长控制切片（2026-10-08）

## 交付范围

本报告是用户允许的逐切片交付，**不是整体闭环资格任务完成**。本切片修复共享context漏检、绑定含源Torch事件时钟、解除真实初始化对namelist步长数组的依赖，执行无训练回归和两组有限记录审计。没有训练、optimizer更新、超参数搜索、源回放或JOREK修改。联合学习仍为主模式，冻结容量仅为参考。

## 已复现的问题与实现

新增 `tests/test_native_context_guard.py` 首次运行6 failed/2 passed：错误位置、signed size、共享索引、单位、边界policy和model配置均能绕过耦合入口。现在 `native_field_step.validate_reference_context` 是场入口和耦合入口通过NativeReferenceSolver共用的防线；构造及调用时验证，保存完整context指纹，拒绝运行中包括参数来源、数值/边界声明的漂移。耦合入口必须提供mesh.index；native读取器保存该轴。完整几何仍通过cubic求值器的几何/单位/基函数指纹绑定。

原参考fixture只传节点位置或修改units后不重建求值器，先前因漏检而通过。现在补全其几何/index并重建正确单位的求值器，保留原解析容量、事件顺序、恢复及失败回滚测试，没有降低校验。

新增 `FieldEventContext(field_time_si, solve_time_si, phase, provenance, gather_time_si)`，由当前场的所有者显式提供。SourceEventTime不再单独充当因果依据。公开含源Torch入口校验源端点、实际gather及phase：

- ordinary：正dt，field/solve/source端点一致且gather与上下文一致。
- zero_time_probe：dt精确0，合法普通源上下文；完整U/rawD/学习记忆不推进。
- registered_restart：正控制dt，solve使用marker事件时钟，可早于导入场端点；必须为注册零投影事件，不是普通窗口或模型恢复。

ReferenceRollout在已有求解事件中从自身场/marker时钟生成上下文，没有新增调度器。源反馈仍在当前源事件生成后传入；下一事件的源仍消费预测场。Torch调用者对上下文来源负责：类型和来源声明不能证明外部文件真实性；入口不从标签或源时间反推场时间。当前实现核对声明的gather，不另外重演粒子子步来证明它。

版本 `native-source-field-v3_bound_events` 纳入检查点配置；旧v2或field-only不能作为新模式精确续跑。新增v2 schema和bound-events配置，旧schema、配置及实验产物保留。现有保存/恢复测试验证新模型恢复一致。

## 步长控制与源码核查

JOREK HEAD `54c675bcc39c33b082c0c35046e28122dc53b3aa`，只读核查：

| 源码 | 已确认语义 |
|---|---|
| core/mod_jorek_timestepping.f90:310–325 | 本次控制get_tstep_n提供dt_jorek并赋tstep；前一步控制赋tstep_prev；首次用当前dt，SI转换乘t_norm |
| 同文件:327–335 | startup把t_now设为sim%time/t_norm，重置index_start并增加内部步号 |
| 同文件:418、424–429 | OLD场update_spi；提交values；update_deltas；更新时间与tstep_prev |
| particles/examples/marker_spi_source.f90:79–86 | 导入tstep用于把标称粒子步调整为整除导入流体步，不是下一控制计划 |
| 同文件:311–313、348–355 | restart清residual并启动事件；真实粒子起点sim%time−residual；覆盖至target的子步数与新residual分别计算 |

初始化直接读取HDF5 tstep/index_now/t_now/t_norm；index_now现在必须存在且等于指定内部事件号。raw fluid_deltas仍直接读取，不能由时间差/端点重建。保留导入tstep及来源、内部时间、粒子标称步。不再读取namelist tstep_n/nstep_n作为控制；通用数组解析器本轮未修改。

`reference_rollout_from_checkpoint(..., control_plan_si=..., control_origin=...)`：

- restart和普通续跑均要求显式正有限下一步控制；缺失失败。
- external_control是调用者提供的控制；旧continuous_remaining_plan_si保留为显式兼容参数，两者不可同时提供。
- 从记录文件整理的计划必须声明recorded_schedule_replay；它不代表自主推理已知未来计划。
- 控制来源保存在运行状态provenance，计划及cursor由原模型检查点保存。
- 导入tstep是历史；startup前dt仍按JOREK采用当前控制dt；普通续跑前dt使用导入步长。没有恒步长回退。

## 实际验证与命令

工作目录仓库根，Python为`.torch-cuda-env/bin/python`。所有命令不调用训练器。

```bash
.torch-cuda-env/bin/python -m pytest -q tests/test_native_context_guard.py
# 最初：6 failed, 2 passed；时间上下文切片另有5个失败测试。
.torch-cuda-env/bin/python -m pytest -q tests/test_native_checkpoint_control.py
# 最初：2 failed，复现数组依赖/缺少显式控制接口。
.torch-cuda-env/bin/python -m pytest -q tests/test_native_context_guard.py tests/test_native_checkpoint_control.py tests/test_native_joint_source.py tests/test_native_coupling.py tests/test_native_field_step.py tests/test_native_temporal.py tests/test_native_features.py tests/test_native_windows.py tests/test_native_state.py tests/test_marker_state.py tests/test_fluid_history.py tests/test_native_gauss.py tests/test_marker_feedback.py --junitxml=output/native-closure-entry-2026-10-08/regression.xml
# 219 passed，27.71s；在最后补充gather/指纹防线之前。
.torch-cuda-env/bin/python -m pytest -q tests/test_native_context_guard.py tests/test_native_joint_source.py tests/test_native_coupling.py tests/test_native_field_step.py tests/test_native_checkpoint_control.py tests/test_marker_state.py tests/test_fluid_history.py tests/test_native_state.py --junitxml=output/native-closure-entry-2026-10-08/final-slice.xml
# 101 passed，10.15s；包含最终生产代码与新增gather测试。
.torch-cuda-env/bin/python -m pytest -q tests/test_native_context_guard.py tests/test_native_checkpoint_control.py --junitxml=output/native-closure-entry-2026-10-08/guards-final.xml
# 17 passed，1.08s；进一步确保错误时刻测试提供有效gather，避免因别的错误先失败。
PYTHONPATH=. .torch-cuda-env/bin/python tools/audit_native_coupling.py --output output/native-closure-entry-2026-10-08/phaseA-metadata.json
PYTHONPATH=. .torch-cuda-env/bin/python tools/audit_native_closure_entry.py --output output/native-closure-entry-2026-10-08/qualification.json
# 两条命令成功；新审计拒绝覆盖已有输出，复现时换输出路径。
```

已有测试覆盖两事件预测场反馈、OLD场shard、模型恢复、三阶段失败回滚、完整degree、跨步反传及显式截断。沿用原数值容差；时间校验max(1e−14秒,t_norm×1e−7)，用于HDF5输出精度，不把重复restart时间解释为零gather。没有运行包含optimizer smoke的test_native_curriculum.py。

## 真实数据证据与资格边界

范围严格限定v400 4109–4112、injInner 2702–2705，不泛化全run。两者完整场为[9,4,1,4125]、unique index为16419，已检查保存rawD。普通时间差与记录tstep匹配，4111/2704是注册重复流体时间；前投影为滞后零投影。本切片使用现有真实v400完整marker初始化读取回归；另一run只完成有限元数据/历史存在性审计，不冒称完整marker锚点读取。

v400 part04110：1000000槽、11168活粒子，所需marker数组未压缩字节数172000000。这个数不是进程峰值内存或推进成本。未裁剪活粒子，未执行超过当前10000活粒子限制的参考回放。所选injInner路径无匹配part，不推断全run无part或注入前空云。

机器清单按实际字段/时间检查报告资格和原因；它是**记录入口审计，不是窗口manifest或已认证锚点**。projection文件无deltas标saved=false，同时差分重建标pending_evidence；不直接判定永远不可恢复。

## 审计缺口状态与后续切片

| 缺口 | 本轮状态 | 剩余工作 |
|---|---|---|
| namelist截断影响导入控制 | 已修复 | 外部调用者必须提供下一步控制；历史旧调用需显式迁移 |
| 耦合绕过完整context校验 | 已修复 | 物理BC仍未验证；当前只接受明确learned_unverified策略 |
| Torch源/场时间与phase未绑定 | 已修复 | 外部上下文真实性仍需数据资格认证；gather核对不等于粒子时钟重演 |
| 真实16419 DOF稠密投影 | 未实现 | 先分离局部求值与全局矩阵，再共享索引稀疏投影；不提高128上限 |
| 两run完整锚点/投影历史 | 待证据 | 精确补齐前场、saved clock/residual/actual gather、更新前后投影及滤波阶段/精度、外部控制、源配置相位；不以任意相邻差分补齐 |
| production abl3/rocket6/nonlocal/filter/原子分支 | 未实现本轮核查/接入 | 当前小源明确no_rocket/no_nonlocal/未滤波/PLT-only；不能认证目标生产配置，不把rkt6叫abl6 |
| 真实物理边界约束 | 未实现 | 对实际BC码、轴和配置建立源码映射及解析验收，不能把共享索引平均当BC |
| 真实源尺度/绝对eventJU饱和 | 未实现 | 先无训练幅值/预激活/灵敏度诊断，禁止沿用本fixture尺度作为生产统计 |
| 窗口manifest动态四类资格 | 未实现 | 本轮机器清单只覆盖记录审计；需源/初态支撑进入split隔离和窗口检查，旧硬编码数量尚待替换 |
| 真实配置两事件/恢复 | 未实现 | 先真实稀疏空间路径、源配置、BC和锚点认证；不使用真值重置/补零取得运行 |

可执行补采要求：在同一生产运行的after-fluid/before-particle锚点，一并输出完整当前/前场与rawD、shard旧烧蚀/运动状态、全部marker槽及P_imp/v/身份、投影values及原始deltas、sim时间/真实粒子时间/particle_dt/residual/实际gather、事件phase和完整生效配置；保存滤波前后定义及输出精度。只新增projection deltas不够消除全部资格缺口。下一切片优先推进真实共享索引稀疏投影及锚点认证，再扩展窗口资格入口。

结论仅为：**入口共享校验、源事件绑定和显式控制步长切片通过无训练验证**。不宣称整个P0/P1/P2或闭环训练集任务完成，不宣称物理BC、真实生产源闭环、预测精度、长期稳定或C0通过。
