# 原生因果窗口与参数路由（2026-10-08）

结论限定为**原生因果窗口与参数路由可用（显式支持域）**。本轮没有训练、CfC、梯度新求值器、求解核心修改或九场矩阵；不是预测有效性、物理反馈接通或 C0 闭环验收。现有 source_proj 未物理对齐、原生场步仅接受零反馈的限制不变。

遵循 ADR-0018/0019/0020/0021：0021 定向修订参数/时间核心/课程，0020 继续定义源、状态与 restart。codebase-design 将窗口生成、文件 Adapter 和参数清单集中在数据 Seam；tdd 通过公开 Interface 验证。domain-modeling 仅澄清原生弹丸材料数密度单位，向 CONTEXT 增补一个词条；没有修改 ADR、roadmap 或只读 JOREK，没有覆盖其他用户修改。

## 已实现的 Interface

- `src/data/native_windows.py`：`FieldRecord / SplitBlock / WindowConfig → build_windows → WindowCollection`。先检查 split blocks 及隔离，再仅在每块内部生成窗口；L/K/stride/isolation 均必填，不继承旧 manifest。完整支撑 `[n-L,n+K]` 含场初始化锚点、warm-up 和目标。回放尚未支持，`replay_steps>0` 明确拒绝，因此不会漏隔离隐含回放记录。
- `NativeWindow.current_input()` 返回独立 `CurrentFieldInput`，包含当前完整 U/rawD、当前/前 dt、格式、计划端点、声明的计划来源、可选同号完整 S/rawD_S、独立 source interval、分组参数。没有 future targets / label accessor / run ID 输入字段。数据对象自身可持有未来标签；该视图不将未来场、marker、投影或目标时间差用作输入。
- `targets` 是 K 份完整 NativeFieldState（含独立原始历史），不是端点差分标签；`warmup` 不计目标 loss。manifest 记录 warmup/loss/valid target masks、独立场/源 teacher masks、horizon 和 truncation。没有随机 teacher 抽样器或训练器；mask 表示配置的确定性策略，不声称实现 scheduled sampling 或 DAgger。
- `src/data/native_window_io.py::read_segment` 复用全轴单文件读取、生产 namelist 解析链和 `audit_native_segment`；显式范围、不自动扫描或补齐未来文件。U/rawD 原生 dtype、全部 `[9,4,H,N]` 均保留；真实首轮 H=1。兼容指纹含完整几何/index/边界、模式、单位和非数值时间参数；参数跳变默认切断。普通 dt 变化保留。
- `NativeScaler.fit({'train': records, ...})` 只看唯一 train 支撑记录，逐 field/degree/harmonic 拟合当前场中心/标准差和原始历史 RMS。`U'=(U-c)/s_U; D'=D/s_D`，逆变换显式；原始 D 不减中心、不乘 dt 比、不被覆盖。零方差组使用正单位 scale，绝不补缺失状态。保存版本、训练身份支撑、单位 digest、模式与所有统计；encode 拒绝单位/模式不符。线性缩放数学可逆，浮点逆变换不保证逐位；读取原始系数则逐字节验证。
- `src/data/native_parameters.py` 复用 H5→生产 namelist→旧默认取值链，但不复用旧全集 concat。每项提供 name/group/unit/provenance/available_at/normalizer/consumer/supported_range/value/status。场景、物理系数、数值时间分组；消费者细分至磁扩散、离子/电子热传导、密度输运、边界、归一化、pellet 条件等，均明确哪些尚待接入。没有把声明路由当作已执行消费者。

参数未知来源明确失败。未知单位、未确认默认值、旧重复项及局部数组 padding 保留在审计清单，排除在 `current_input.parameters` 外；不静默补值。旧 derived Gears zeta=.5 只作 audit，不作可变步长求解输入；实际格式/系数由 dt 对及已有时间契约计算。来源路径属于审计 metadata，不是给网络的 ID 编码。后续模型必须只编码逐项声明值/组，禁止拼入 provenance/path/name/计数身份。

## 时间、教师反馈与初始化资格

时间计划由调用者的 `declared_run_control` 或 `declared_segment_schedule` 提供；独立计划对未来记录端点作**一致性验证**，不由标签反推输入 dt。当前 previous_dt 来自已保存当前文件 tstep；restart 帧保留导入步长。H5 field t_now 为 JU→SI，projection t_now 本身 SI；内部步号与 macroscopic 表一起核验。普通间隔容差沿已有 audit：1e-7 JU / max(1e-14 s,t_norm*1e-7)。重复时间仅登记 restart 可解释。

source teacher policy=`recorded_full_history` 要求每个消费步的 S_n、原始 projection deltas、单位/完整轴、同号时间和正时长 source interval；五通道全部保留。缺历史明确失败，不差分相邻文件、不补零。gather time 单独保存，可未知；未知不能变成流体 dt，后续需要它的消费者必须拒绝。restart 上未知采样窗口不推导为零。field-only policy=`none` 的 feedback=None 表示未提供，不表示物理反馈为零。

普通窗口不触发 JOREK restart。跨 restart 的候选排除，允许窗口以事件帧为首个支撑锚点；没有重置 marker 云、反馈或物理时钟的运行代码。场 TF 的替换单位是完整 U+rawD，实际输入改变后必须重新计算兼容梯度；目标场不作为模型偏离轨迹的专家一步重标注。

当前资格入口只开放 `field_supervised`。`required_qualification='coupled_rollout'` 明确拒绝；该 Dataset 尚未把已有完整检查点初始化 Adapter 转成可验证的窗口资格，也未实现 marker 因果回放。仅有路径、稀疏快照或初始化字典不能通过资格检查；缺 marker 不伪造空云。这个限制是登记缺口，不是断言所有真实 run 都无法耦合。场监督窗口可准备数据，不代表当前 zero-only 场步已能正确消费真实非零反馈。

## 真实覆盖与排除

可复现配置：`config/native-window-phaseA-audit.json`。L=1、K=2、stride=1、isolation=1，field teacher=`paired_U_rawD`、source teacher=`none`、truncation=2。gamma=5/3 是显式组件审计配置并记录来源；不是从未来标签或缺失 H5 猜得。固定配置依据用户确认：两组 phaseA 各自运行中除步长表外配置不变。

| 显式读取片段 | 条数 | split blocks | 有效窗口预测起点 | 排除 |
|---|---:|---|---|---|
| phaseA v400 4108–4120 | 13 | train 4108–4115；test 4117–4120 | train 4112/4113；test 4118 | 3 个跨 restart 4111 候选 |
| phaseA v400_injInner 2701–2708 | 8 | validation 2701–2708 | validation 2705/2706 | 3 个跨 restart 2704 候选 |

总计 **21 条读取记录、11 个候选、5 个场监督窗口、0 个耦合资格窗口**。train/validation/test=2/2/1；完整支撑分别 train4111–4115、validation2704–2708、test4117–4120，train/test 之间隔离4116。未被任何合格窗口使用的记录不参与统计。此划分只用于读取/隔离组件验收，不是有代表性的训练/泛化评测。

所测流体全部内部 index 连续、普通步长时间一致，shape=(9,4,1,4125)，U/rawD 原生完整。v400 普通 dt .5→restart 后 .05 JU；injInner 为5 JU。4110/2703 为 restart 启动前滞后零投影，4111/2704 流体重复时间，后续 projection 时间对齐；不会将重复流体时间解释成零粒子采样窗口。稀疏 marker 的 /time 单独按事件检查，未载入百万粒子 payload。

21 个 projection 文件均无 deltas，所以本次真实窗口**不能启用 recorded_full_history**。完整五通道+独立历史的 teacher 配对使用小型确定性 fixture 验证；不是实际数据已补齐历史的声明。只核查上述两个片段，不声称全 run 覆盖，也不沿用旧“phaseA 普遍跨多流体步”的泛化。

## 参数变化与证据缺口

逐项机器清单：`report/native-parameter-routes-phaseA-2026-10-08.json`（代表性 v4004108 的61条旧参数表面，含两个 bc_natural_open，故不假称61个独立参数）。每个窗口支撑记录的完整清单由生成 manifest 保存；各 split 统计及21条时序核验见 `report/native-window-coverage-2026-10-08.json`。

21×61 条参数记录的状态：declared252、unconfirmed_default126、unresolved_unit777、audit_only84、duplicate_legacy_entry42。可声明的参数仍需后续消费者与缩放接入；未解决项不以不明单位数值硬塞入网络。默认值仅调用者明确确认且单位已知才可开放；本审计没有确认任何 fallback。

train 有5个唯一支撑记录：57项常数不可辨识，t_now/tstep变化；validation5条、test4条各58项常数，仅 t_now变化。重复 bc_natural_open 单独登记。物理/场景参数在各 split 内都未提供独立变化证据；没有非恒定物理参数可用于估计共线作用。独立小序列另验证 a=[1,2,3], b=[3,5,7] 被标记为完全仿射共线。两段差异不能证明跨场景泛化。速度/角度等不在旧61维取值表面的条件尚未补充，不从 run 名反推。

重新核对只读 JOREK HEAD=`54c675bcc39c33b082c0c35046e28122dc53b3aa`：`models/preset_parameters.f90:555` 明确 pellet_density 单位10^20 m^-3，`:54` 同样定义 central_density。新路由 overlay 修正这两项；不改旧参数 Interface。旧 ZK_par_max 单位标签未作为可靠 SI 热导声明沿用，继续标未知单位。源码中的默认值与历史表未逐项重新核对，所以 fallback 保持未确认。

## 后续兼容值/梯度输入契约

后续 G/P 切片必须输入完整实际 U[9,4,H,N]、其 NativeUnits/模式及原生 x/signed size/vertex/index/约束几何。H=1 的值布局 `[E,Q,9]`，梯度 `[E,Q,9,2]`，方向为物理 R/Z；版本应绑定原生 B/G/几何 digest、模式及单位。先在物理单位求值和求导，再分别按 field/方向缩放。由同一份完整 U 派生；不维护独立梯度状态，不复用被教师/预测场替换前的梯度。没有新增有限差分、dof0 或双线性代理。

梯度统计尚未拟合，不能复用系数 scale 冒充物理梯度单位。Gauss→节点/DOF 聚合、Torch 可微 B/G、参数编码/缩放、CfC/记忆、物理 BC、非零反馈消费与训练器继续按设计切片实施。此数据准备实现采用显式片段的内存参考路径，尚未证明全 run 的加载吞吐/峰值内存；不默认全粒子回放。TBPTT 截图和课程 RNG/optimizer 续跑由后续训练器实现，不能沿用推理 checkpoint 充当完整训练续跑协议。

## 验证与审计命令

```bash
.torch-cuda-env/bin/python tools/audit_native_windows.py \
  --spec config/native-window-phaseA-audit.json \
  --output-dir /tmp/native-window-audit-2026-10-08
```

输出 manifest.json、normalizer.json、coverage.json、parameter-routes.json；JSON 文件各自原子替换，不宣称四文件事务。输入 spec 与 schema 版本严格区分旧 manifest；显式字段契约见 `docs/schemas/native-causal-window-v1.schema.json`。当前环境没有 jsonschema 包，因此未声称执行独立 Draft2020 验证；实现入口执行所需语义校验，测试读取生成 JSON 并核对布局/长度/资格。

TDD 红→绿：窗口入口缺失；缩放入口缺失（浮点逆变换容差调整）；路由入口缺失；真实读取入口缺失；不该路由的默认项被暴露；审计入口/registry 层级；缩放未绑定单位、未知来源漏拒绝；原生 pellet 单位错误；具体消费者和未确认热导单位；反馈单位/轴错配漏拒绝，均先复现再补实现。早期一次 `python` 命令不存在，改用仓库既有环境后执行；未写入其他路径。

| 命令 | 结果 |
|---|---|
| `.torch-cuda-env/bin/python -m pytest tests/test_native_windows.py -q` | 16 passed / 4.42s |
| `.torch-cuda-env/bin/python -m pytest tests/test_native_windows.py tests/test_native_field_step.py tests/test_native_state.py tests/test_fluid_history.py tests/test_marker_state.py tests/test_native_gauss.py tests/test_marker_feedback.py tests/test_source_module.py tests/test_field_propagator.py tests/test_step_dataset.py tests/test_output_contract.py -q` | **262 passed, 1 skipped / 24.47s** |
| 上述 audit CLI | exit0，21条、5窗口、6 restart 排除，两个真实片段均执行 |
| `.torch-cuda-env/bin/python -m compileall -q src/data/native_windows.py src/data/native_window_io.py src/data/native_parameters.py tools/audit_native_windows.py tests/test_native_windows.py` | exit0 |

原始 HDF5 U/rawD 用 bytes 相等；三 harmonic/全 degree 的仿射逆变换容差 rtol=3e-16、atol=1e-14，仅覆盖 float64 运算舍入，不是预测精度门槛。未来标签改写不改变当前输入；验证数据改写不影响训练统计；split 支撑互斥；缺历史/错时间/未知来源/未支持回放与耦合资格拒绝；失败不修改已有输入。保留已有场步、ReferenceRollout、局部载荷、模型 checkpoint 等回归；这些是组件测试，不是自生成源闭环。
