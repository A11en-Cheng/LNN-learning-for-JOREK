# 完整原生联合学习场步结构与事件接口（2026-10-08）

交付结论仅为**完整原生学习场步结构与事件接口接通**。未开展训练，未实现JOREK九场矩阵、容量矩阵或完整闭合消费者；不代表预测精度、五通道物理耦合或C0闭环通过。现有`FieldPropagator.source_proj`仍未物理对齐，未借本轮改变其标识。

依据现行ADR-0018/0019/0020、[B2→C0设计](b2-c0-native-state-feedback-design-2026-10-08.md)及[切片A报告](native-state-fidelity-resume-2026-10-08.md)。使用codebase-design确定一个联合场步Module、原生更新映射和现有solve_fluid Seam；使用tdd按已指定Interface逐切片红→绿。没有新建调度器，没有修改roadmap、ADR决策或只读JOREK；保留其他用户改动。

## 源码确认：原始增量与更新映射

只读源码 `/home/ac_desktop/JOREK_code/jorek`，HEAD `54c675bcc39c33b082c0c35046e28122dc53b3aa`。本轮重新核对普通非JECCD分支，未混用FFT或其他模型。

| 事实 | 位置 | 本轮规则 |
|---|---|---|
| keep_n0_const将harmonic循环起点改为2 | `communication/update_values.f90:32–33,88–96` | H=1且显式冻结时不更新任何场系数，保留旧node%deltas；默认生产测试未冻结 |
| 非constrained节点按degree的共享index取求解增量，加到values并直接写deltas | 同文件 `35–42,88–94` | 返回原始求得增量，不做values-old_values；共享index必须取得同一增量 |
| treat_axis且axis_node时先变换轴自由度并恢复共享RHS条目 | 同文件 `42–58,100–113` | 主动轴变换未实现，treat_axis=True明确拒绝 |
| constrained节点从父元素/父节点分别插值values与deltas，导数系数含3/9与size | 同文件 `120–145,250–314` | 悬挂节点插值未实现，任何constrained=True明确拒绝，不能简单冻结该节点或补零历史 |
| update_deltas从node%deltas打包求解向量，不相减场端点 | `communication/update_deltas.f90:21–39` | 模型历史直接来自更新映射 |
| JECCD最终变量有特殊冻结例外 | `communication/update_values.f90:60–85,148–247` | 显式jeccd=True拒绝；本轮只支持standard_nonJECCD声明 |
| Dirichlet等物理边界是在矩阵中处理，涉及field/degree方向、freebound和Mach开关，不是所有节点/degree统一冻结 | `models/model502/mod_boundary_conditions.f90:224–248,319–367` | 本轮未执行该物理BC装配，显式登记learned_unverified；请求jorek_physical或未声明策略拒绝 |

源码SHA256：`update_values.f90=e54072564f2a57e91989ea9696ecbc2956ebccef14ef96279a24e698c8fd7578`；`update_deltas.f90=345ef8d448275f86559e6b4327edc1f43f063d3196e27bdf8ebeb406273c3545`。`models/preset_parameters.f90:600,873`分别默认keep_n0_const=False、treat_axis=False；文件中treat_axis仍以实际保存值为准，keep_n0_const由调用者显式解析/声明。

### 当前phaseA元数据勘误与支持域

真实测试文件为 `phaseA_dataset_1d22_ang15_massloss_neon_v400/jorek004112.h5`，内部步号4112；流体t_now约8602.05 JU，tstep=.05 JU，完整values/deltas均float64、shape=(9,4,1,4125)。时序与登记restart4111依据切片A已核查片段。

**该文件实际存在**`index/constrained/axis_node/axis_dof/treat_axis`：index=(4,4125)，正整数范围1..16419且有16419个唯一index；4125节点全部constrained=F；68节点axis_node=T，但treat_axis=F。旧报告/ADR背景中的“axis_node不存在”等数据缺失描述不能套用到当前文件；这是本片段事实勘误，不改动ADR决定，也不据此宣称主动轴处理已实现。约束读取不能用节点空间坐标是否R=0代替这些元数据。

## 可微联合场步Interface

实现：[native_field_step.py](../src/models/native_field_step.py)，读取Adapter追加至[native_state.py](../src/data/native_state.py)。

```
NativeFieldStep(geometry, update_mapping, units,
                state_scale[9,4], increment_scale[9,4], center[9,4], ...)
forward(values[9,4,1,N], raw_fluid_deltas[same],
        dt_si, previous_dt_si, scheme, time_coefficients=None)
  -> TorchFieldStepResult(values, fluid_deltas, increment)
```

支持生产H=1的全部4个degree和九场；H>1、不完整轴、缺历史、非有限量或dtype/device不匹配拒绝。默认float64，调用者需明确决定模型dtype。不是只对dof0预测：每节点按`field*4+degree`排列全部36个系数，联合encoder同时输入36个当前系数、36个历史系数和5个显式时间特征；decoder输出全部36个本步增量。psi高阶系数可以影响Te高阶增量，九场不分7+2。

核心Torch路径只用可微permute/reshape、线性/tanh、既有implicit stencil matvec/Richardson迭代和共享索引index_add。没有调用切片A用于存储的to_node_layout/from_node_layout，没有把当前场或历史detach/转NumPy后推进。构造时的静态NumPy网格/稀疏算子以及配置指纹不属于场计算图。时间/格式是非可学习metadata，不承诺对dt求梯度。

### 缩放、逆变换和时间所有权

| 对象 | 定义 |
|---|---|
| 当前场输入 | `U_normalized=(U-center)/state_scale`；逆为`center+state_scale*U_normalized` |
| 原始流体历史输入 | `D_normalized=D_raw/increment_scale`；不减center、不先乘dt比；逆为`increment_scale*D_normalized` |
| decoder输出 | 无单位标准化**本步增量**；`ΔU_proposed=increment_scale*decoder(latent)` |
| 最终更新 | 自由行`U_new=U_old+ΔU_applied`，`D_new=ΔU_applied`，其中ΔU_applied已通过共享index映射；不重建U_old、不由舍入后的端点差恢复D_new |

每个field/degree有独立正scale，center可带符号；都作为buffer保存并绑定检查点配置。degree1/2/3（零起始）的系数不是正密度/正温度，不能照搬节点正热力学量的symlog/截断。这里选择可逆线性缩放，没有声称它是已标定的训练统计；测试的显式scale只是结构验证配置。

时间特征为 `[dt_JU, previous_dt_JU, dt/previous_dt, theta, zeta]`，SI步长各除一次t_norm。Euler/CN/Gears沿现有`fluid_time_coefficients`生成，Gears zeta=dt/(dt+previous_dt)。若ReferenceRollout传入系数，必须与相同显式步长/格式的计算结果一致。stencil使用theta*dt_JU作为近似latent隐式步参数；decoder之后**不再乘dt**。本Module不是rate网络，不把latent forcing或弱式载荷混入原生增量。

原始D_u持久保存不被缩放覆盖。网络显式得到raw历史和时间系数，不声称其数学运算已经实现JOREK全部Gears时间行；时间项的联合物理响应仍是待学习/验证对象。

## 空间与共享自由度：明确登记的近似

复用`implicit_stencil.build_term_operators`与`ImplicitStencilLayer`，使用现有四边形拆三角P1/lumped节点几何，encoder先在节点内混合所有field/degree，latent各通道进行隐式stencil交换，decoder回到全部原生系数。因此完整场存储没有丢degree，但**空间学习算子不是原生cubic FEM全degree块算子**：未逐项使用signed size/完整试探测试基构造高阶空间矩阵，也不宣称继承JOREK全部刚性稳定性/准稳态性质。

仅使用Torch Richardson路径；调用者将solver改为direct等不支持路径会失败，避免隐式转入NumPy求解。复用注册term标签eta只为取得节点核；所有latent通道用显式`stencil_strength`，它是学习空间系数，**不是**model502物理电阻率或已接通的电荷闭合。

共享原生index处，decoder先提出节点增量，再按同index取均值并广播，使同一求解自由度的更新一致。**取均值是本次学习投影选择**；JOREK实际解的是唯一RHS自由度，不会平均多个独立节点预测。此近似需后续对共享轴/高阶响应标定；它不是悬挂节点父元素插值，也不是主动轴变换的替代。keep_n0_const=True的H=1仅用于冻结规则组件验证，不是默认将高阶系数冻结。

边界策略必须显式声明`learned_unverified`，其含义是边界系数由未验证的学习映射更新，**没有执行物理BC**。本轮不调用旧节点BC层去冒充全degree物理边界；不宣称通过roadmap的BC或刚性验收。请求真实物理BC、JECCD、主动轴变换、悬挂节点或非正index配置明确拒绝。

## ReferenceRollout与检查点接入

`model.reference_solver(context)`返回现有solve_fluid Interface的Adapter：

```
solve_fluid(old_fields, shards, feedback, dt, previous_dt,
            fluid_deltas=..., time_coefficients=...)
  -> FluidSolveResult(values, fluid_deltas)
```

沿现有ReferenceRollout执行粒子/反馈→本次场候选→旧场shard更新→场/历史与时钟共同提交；没有改变调度顺序或新增平行rollout框架。该NumPy驱动的Adapter用于推理，以no_grad调用核心Module，最后转换提交数组；**不承诺跨ReferenceRollout事件反向传播**。需要学习梯度时直接调用Torch forward，本轮真实前后向测试使用该路径。

本轮未消费marker五通道或shard密度源，Adapter只接受显式全零feedback及全零projection history；任何非零反馈拒绝，不能静默忽略教师或自生成反馈。shard参数只是保持既有Interface；本轮没有其物理流体源消费者。因而事件接通测试是无反馈结构测试，不是自生成源闭环。

Context需绑定units、mode=(0,)、正确网格、明确时间格式、learned_unverified策略及`configuration['model']=model.checkpoint_configuration`。配置记录完整缩放、共享index、冻结分支、几何指纹、架构、迭代格式/次数/系数；现有checkpoint签名加入该声明。保存时模型/rollout配置不符直接拒绝，未写出部分文件。加载沿切片A的完整权重、模式、RNG、物理状态/历史及原子加载机制；不会附加JOREK restart。

## 独立测试与真实数据证据

测试：[test_native_field_step.py](../tests/test_native_field_step.py)，23项。

- 所有36个field/degree输入和历史都具有非零梯度，所有field/degree得到非零候选更新；逐degree扰动改变输出；显式检查psi高阶系数影响Te高阶增量。encoder/decoder和stencil全部可学习参数的梯度有限且非零。
- 受控权重、不带空间强度时，用独立标量tanh计算oracle：U_centered/scale=.5，D/scale=.5，h=.5，theta=1，zeta=1/3，输出为各degree scale乘`tanh(17/6)`；变步长为`tanh(11/3)`，去历史为`tanh(7/3)`。Euler/CN/Gears各3个步长比另用独立公式核对。不调用实现本身生成期望，也不是u+1顺序测试。
- 共享index的两个提议1/3得到同一增量2；自由行`1e16+1`舍入后values不变但历史仍为1；显式H=1冻结保持旧历史.25且本步增量为0。悬挂节点/主动轴/JECCD拒绝。
- 真实`jorek004112.h5`完整U/D_u及实际共享index读入，只有该当前文件被允许打开；下一dt=.05 JU由测试显式提供，前dt使用当前H5已保存tstep，不读取下一帧决定输入。CPU float64、hidden12、一层2次Richardson，state_scale全10、increment_scale全.001为声明的诊断配置。forward/backward中禁止Tensor.numpy和存储layout Adapter，全部degree更新与输入/历史梯度非零且有限。未读part/projections或下一场作为输入。
- 连续两个变步长事件与中间保存、在不同随机初始权重的新模型加载后继续，场和原始历史逐位一致；保留游标和非startup状态。shard失败不改已提交场/历史、marker时钟或shard；非零反馈、缺历史/错误轴、非有限输入/归一化溢出、错误格式/系数和错误网格拒绝。

独立float64标量/代数oracle容差`rtol=2e-14, atol=1e-14`；这是浮点求和与tanh舍入容差，不是预测误差门槛。续跑同平台/dtype使用字节一致；未验证跨设备逐位一致或GPU性能。

## 命令与结果

使用仓库既有`.torch-cuda-env/bin/python`，无optimizer step或正式训练。

| 命令 | 结果 |
|---|---|
| `.torch-cuda-env/bin/python -m pytest tests/test_native_state.py tests/test_field_propagator.py tests/test_fluid_history.py -q`（改动前） | 48 passed / 9.53s |
| `.torch-cuda-env/bin/python -m pytest tests/test_native_field_step.py -q`（tdd逐切片） | 入口缺失1 failed→1 passed；共享索引规则1 failed→2 passed；时间系数keyword缺失1 failed→4 passed；checkpoint/Reference Interface缺失1 failed→5 passed；原生update读取入口缺失1 failed→6 passed；归一化溢出漏拒绝1 failed→7 passed；补时间/非法输入至21 passed；物理BC/JECCD请求漏拒绝1 failed→22 passed；非Torch求解及错缩放保存漏拒绝1 failed→23 passed |
| `.torch-cuda-env/bin/python -m pytest tests/test_native_field_step.py tests/test_native_state.py tests/test_fluid_history.py tests/test_marker_state.py tests/test_native_gauss.py tests/test_marker_feedback.py tests/test_source_module.py tests/test_field_propagator.py tests/test_step_dataset.py tests/test_output_contract.py -q` | **246 passed, 1 skipped / 20.15s**；新增真实读取/反向未skip |
| `.torch-cuda-env/bin/python -m pytest tests/test_native_field_step.py -q`（最终补全部参数梯度断言，Implementation未改变） | 23 passed / 1.15s |
| `.torch-cuda-env/bin/python -m compileall -q src/models/native_field_step.py src/data/native_state.py src/models/marker_state.py tests/test_native_field_step.py`；`git diff --check` | exit0 |

另外只读`rev-parse HEAD`、上述两个源码文件sha256以及真实文件index/axis/constraint/shape元数据均exit0，结果已记录本文。没有执行完整求解器、正式训练或未来帧推理回放。

## 后续缺口

完整原生状态与联合学习输出现在可用，但未学习预测精度；节点P1空间近似、共享index均值投影、物理BC/主动轴/悬挂节点、高阶几何块、容量响应、压力/电阻率/交换闭合、五通道反馈与独立shard源、真实粒子自生成源以及C0闭环均须分别接入/验证。下一步可按既有设计实施局部载荷的有单位响应与闭合消费者，不能把当前输出当成已实现的九场物理解算。
