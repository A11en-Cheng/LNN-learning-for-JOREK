# 大流体步的源—shard反馈时序核查（2026-10-08）

范围：按用户要求只读核查本地JOREK源码，研究共同base时间、跳步与场反馈滞后；不训练、不改模型、不改训练决策或窗口划分。使用research，shard更新由后台研究并行核对。本报告给出源码事实和设计建议，建议不是已验收能力。

检查源码HEAD：`54c675bcc39c33b082c0c35046e28122dc53b3aa`。结论针对该实现，不以此声称与生产dirty构建逐字一致；版本缺证不阻止分析已经明确的更新结构。

## 1. 源码事实

### 三种时间不能混用

1. 流体控制步 `dt_jorek=get_tstep_n(istep)`，转SI乘`t_norm`；下一流体事件按当前dt排程，投影事件与流体事件同目标时刻。见 [mod_jorek_timestepping.f90:310](../../JOREK_code/jorek/core/mod_jorek_timestepping.f90:310)、[340](../../JOREK_code/jorek/core/mod_jorek_timestepping.f90:340)。
2. marker子步在初始化时由导入流体`tstep`与标称`tstep_particles`整除调整，`h_p=h_import/floor(h_import/h_nominal)`。不是每次流体变dt都重新调成同一个base。见 [marker_spi_source.f90:79](../../JOREK_code/jorek/particles/examples/marker_spi_source.f90:79)。
3. 每次源窗的真实粒子起点为`sim_time-residual`，`N=int((target-start)/h_p)`，gather=`N*h_p`，新residual=`target-start-gather`。因此流体间隔、gather和粒子子步不是同一个量。见 [350](../../JOREK_code/jorek/particles/examples/marker_spi_source.f90:350)、[443](../../JOREK_code/jorek/particles/examples/marker_spi_source.f90:443)。

共同base可用于调度/监督对齐，但不能在没核查residual的情况下假定每个源窗都恰好含整数个标称粒子步。非均匀记录应累加真实步长；不能以文件编号差乘固定dt替代。跨事件边界尤其restart不能无条件拼成跳步。

### marker场采样使用已算出的场，不要求未来真值

流体步结束设置 `fields%time_prev=sim%time`、`time_now=sim%time+dt`，供随后粒子窗使用；见 [timestepping:567](../../JOREK_code/jorek/core/mod_jorek_timestepping.f90:567)。粒子子步用其实际时刻`t=start+(k-1)*h_p`调用`calc_NeTe_imp`等，见 [marker:486](../../JOREK_code/jorek/particles/examples/marker_spi_source.f90:486)。

在线线性采样实现读取当前values与原始deltas，执行

`U_sample(t)=U_now-((time_now-t)/(time_now-time_prev))*D_now`。

见 [mod_fields_linear.f90:66](../../JOREK_code/jorek/particles/mod_fields_linear.f90:66)、[mod_linear.f90:19](../../JOREK_code/jorek/tools/mod_linear.f90:19)。原始D不能随意用已舍入场端点相减替代。该正常时间分支没有把插值系数夹在[0,1]，residual可使首个粒子采样略早于场区间起点；另有静态/极短区间分支，不能用一个无条件线性公式替代全部源码行为。

取消“双端点真值插值训练”不等于删除这种数值采样：预测场一旦已由因果求解生成，便是后续源推进可用状态，不是未来记录标签。

### shard不是每个marker子步更新

流体事件先组装/解场候选，再调用`update_spi`，然后才`update_values/update_deltas`；因此shard看到的是更新前节点场。见 [timestepping:393](../../JOREK_code/jorek/core/mod_jorek_timestepping.f90:393)、[418](../../JOREK_code/jorek/core/mod_jorek_timestepping.f90:418)、[424](../../JOREK_code/jorek/core/mod_jorek_timestepping.f90:424)。不是解完场立刻把新场传给同一事件的shard。

`update_spi`用已有速度更新位置，用已有消融率侵蚀半径；再从OLD场求新消融率，并按rocket模型更新速度。见 [pellet_module.f90:270](../../JOREK_code/jorek/models/pellet_module.f90:270)、[307](../../JOREK_code/jorek/models/pellet_module.f90:307)、[704](../../JOREK_code/jorek/models/pellet_module.f90:704)、[864](../../JOREK_code/jorek/models/pellet_module.f90:864)。这些更新含流体`tstep*t_norm`，不是marker粒子子步。

源码循环每个源窗先用当前shard位置/半径/组分/消融率组织出生区域与出生量，之后进入marker子步；该粒子内循环没有调用`update_spi`。见 [marker:392](../../JOREK_code/jorek/particles/examples/marker_spi_source.f90:392)、[423](../../JOREK_code/jorek/particles/examples/marker_spi_source.f90:423)、[480](../../JOREK_code/jorek/particles/examples/marker_spi_source.f90:480)。因此粒子小子步并不能自动提高shard与场之间的反馈频率。

## 2. 大步与多小步不是同一映射

设共同耦合base为δ，总目标时长H=Kδ。只是把输入dt改为H，意味着一次流体/投影/shard事件，期间marker仍可有很多子步；执行K个δ事件则会K次重算场、源、消融及火箭响应。

| 方案 | 同一H内场—shard交换 | 可以宣称什么 |
|---|---|---|
| K次base闭环 | K次 | 保持base反馈节奏的多步rollout |
| 一次H大步 | 1次 | 改变耦合节奏的大步近似，需独立验证 |
| 每K步才算loss/输出，内部仍K步 | K次 | 稀疏监督/输出，不是单次大步加速 |
| 宏步生成中间预测场并内部反馈/校正 | 依新算法 | 新的多速率/宏步方法，当前未据此验收 |

推论：大步会延长OLD场、旧速度/旧消融率的使用间隔，缺失中间重求值。不能说误差必定单调增长或固定延迟恰为Kδ，但可以确定反馈刷新次数减少、最大使用时间尺度增大。其影响可能随强冷却、烧蚀和rocket变化而显著。后台独立核查进一步区分：新场最早进入下一次shard调用的新率/新加速度计算，新率和新速度在后续调用才影响半径/位移；不是所有量具有同一个简单一拍延迟。

隔离时序的手算例子：固定方向、恒定加速度a，代码先位移后更新速度。一次H得到`x=x0+H*v0`，K次δ得到`x=x0+H*v0+a*δ²*K*(K-1)/2`，即使末速度一样，末位置也不同。这不是rocket6恒加速度假设，而是说明更新顺序足以使两种推进不等价。

一个大步的准确流体终点也不能唯一确定中间shard轨迹、出生、投影历史。因此共享base是时间对齐的必要设计约定，不是耦合准确性的充分条件。预测终点的线性连接仍是一种路径近似；不得使用真实远端场给中间源推进提供未来信息。

## 3. 对训练设计的建议（未执行）

- 用户最新明确：若插值是在区间内多步演化、最终回归真实后端节点，则可以保留。建议将其定义为**因果区间内演化＋真实端点监督**：从左端及过去状态初始化，右端真值仅计算loss，不输入推进；有真实内部记录则逐点监督，无记录的内部状态不是插值生成的伪标签。保留这个训练任务不等于采纳双端点条件输入，也不是取消动力学任务。尚未据此修改roadmap/ADR或启动训练。
- 加噪用于因果历史/当前输入的鲁棒性训练：在完整原生系数上定义扰动，由同一扰动场重新求梯度和闭合；不随意独立扰动rawD/投影历史来造不一致状态。噪声模型和物理支持域需在训练设计中单列。右端监督保持原始真值；预测端点与真值计算loss不等于每段后自动用真值重置状态。
- 若一段包含多个真实base步，该任务是base多步推进到真实节点；若在两个相邻真实base节点内加更细子步，细分会改变源/shard离散更新，需明确这是新近似并与base轨迹比较，不能自动称作精确复现原离散过程。只有端点loss不能唯一识别中间轨迹，应充分使用已有真实内部记录和源/位置监测。
- 分别命名δ_couple、h_marker、H_target与监督采样间隔。跳步H与base对齐；记录变步长时用已声明控制序列的累计时间，不伪造均匀标签。
- 先以相同base的完整rollout建立参照，再决定真正大步模型是否需预测中间场/阶段或其它因果校正。不要把内部K步悄悄改名为跳步加速，也不要把增加dt当成完成新接口。
- 用户主指标不变：五通道投影相关度/相对误差、九场相关度/误差、按身份匹配shard位置相关度/误差、逐场loss。比较相同物理时长，并同时看base时刻轨迹和远端结果；只报远端温度不足以发现中间源反馈失真。
- 额外诊断仅为解释误差：每个事件的消融率/速度、源总量、反馈更新时间、gather/residual与更新次数；不替代用户主指标，不预设性能阈值。

本轮未跑JOREK、未训练、未改模型，没有声称跳步架构已经建成。研究回答的是时序机制；实际可用宏步大小和误差必须在明确算法后实测。
