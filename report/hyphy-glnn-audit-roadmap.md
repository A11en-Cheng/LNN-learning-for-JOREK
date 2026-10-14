# HyPhy-GLNN 全库现状与路线图评估报告

审计日期：2026-09-04。本文只读审计，不修改代码、配置、数据或既有实验产物。

> **2026-10-04 路径注记**：本文引用的 v1 smoke test（`tests/test_dataset_loader.py`、
> `test_perception_pipeline.py`、`test_cognition_pipeline.py`、`test_reconstruction_pipeline.py`、
> `test_training_and_vis.py` 等）已归档到 `tests/legacy/`，不再被 pytest 收集；
> 本文整体为历史报告，引用须标注"（历史）"。

## 1. 执行摘要

- 当前处于 **9 场数据解析已开始、单体 9 场基线尚未跑通** 的迁移阶段；不是可进入 MoE 的稳定基线阶段。
- 已有可信成果：HDF5 解析器能从 Model 502 文件生成 `[N, 9]`，字段重排明确；序列滑窗/collate、图拓扑、边界和部分物理算子已存在。
- 真实数据完整性检查通过 104 个文件，说明这些文件可读取且未发现 NaN/Inf；这不等于训练或物理正确性已验证。
- 最严重问题是 parser 输出 9 场，而 encoder 断言 5 场、decoder 输出 5 场；当前端到端 residual rollout 无法闭合。
- Namelist 参数目前只挂在 `Data.namelist_params`，未编码或注入模型；梯度检查点和梯度 firewall 未实现。
- 默认配置指向 8 通道 BEST 数据，parser 要求至少 9 通道 Model 502；默认训练入口因此存在确定的数据契约风险。
- `JorekPhysicsLoss` 的反归一化已覆盖 9 场，但 PDE forward 仍按旧 5 场切片，物理项不能视为 Model 502 验证。
- 暂不具备进入 MoE 的条件。前三个行动是：统一 9 场契约；建立可运行单体 residual 基线；完成单样本过拟合和短序列 rollout 验收。

## 2. 审计范围与方法

检查范围包括 `src/data`、`src/models`、`src/physics`、`src/training`、`src/evaluation`、`src/diagnostics`、`src/utils`、`tests`、`experiments`、`notebooks`、`output`、`output_p1`，以及 `README.md`、`current_version.md`、`new_version.md`、`config.yaml`、`requirements.txt` 和项目指令。

执行过的命令：

- `git status --short; git branch --show-current`
- `python -m compileall -q src tests`
- HDF5 结构探针：读取 BEST 与 ITER 文件的 `x/values/contain_node/vertex/boundary` shape。
- `python -m pytest tests/test_dataset_loader.py tests/test_perception_pipeline.py tests/test_cognition_pipeline.py tests/test_reconstruction_pipeline.py tests/test_training_and_vis.py -q`
- `python tests/verify_data_integrity.py`
- `python tests/inspect_parsed_jorek_data.py`
- `git diff --stat; git diff -- config.yaml src/models/loss.py tests/train_interpolation_test.py`
- `git log -5 --oneline --decorate`

当前分支为 `main`。工作区已有变更：`config.yaml`、`src/models/loss.py`、`tests/train_interpolation_test.py` 被修改，`new_version.md` 为未跟踪文件；这些不是本次审计修改。`compileall` 发现 `src/data/tempCodeRunnerFile.py` 第 1 行缩进错误。未执行 GPU 长训练、完整 benchmark、跨装置/跨网格实验，也未能证明输出目录中的 checkpoint/log 是由当前代码可复现生成。

## 3. 当前成果总览

| 子系统 | 状态 | 证据 | 已验证行为 | 主要缺口 |
|---|---|---|---|---|
| HDF5 解析 | 部分实现 | [`JorekDataParser.parse_to_graph`](../src/data/jorek_h5_parser.py) | Model 502 样本可生成图；9 通道 shape 已观察 | BEST 为 8 通道 Model 600，默认路径不兼容；模型类型未显式校验 |
| 9 场数据流 | 部分实现 | parser 明确拼接 9 场；encoder/decoder 仍固定 5 场 | ITER/EHL 解析结果为 `[N,9]` | encoder 断言 5；decoder 输出 5；loss/trainer/测试混用两套契约 |
| 归一化与反归一化 | 部分实现 | [`JorekPhysicsLoss.denormalize`](../src/models/loss.py)、`train_interpolation_test.py` | 插值脚本对 `[4,5,7,8]` 使用 log1p/relu，反变换使用 expm1 | sequence pipeline 只缩放 psi；benchmark 仍对 `[2,3]` 做旧索引；统计量协议不统一 |
| Namelist 条件 | 部分实现 | [`extract_namelist_params`](../src/data/jorek_h5_parser.py) | 可从属性/部分字符串数据集提取四个 key，缺失回退 1.0 | 无 condition MLP，无模型注入，无“真实值还是回退值”的有效性标志 |
| 序列数据集 | 已实现 | [`JorekSequenceDataset`](../src/data/dataset.py)、`collate_temporal_graphs` | 滑窗、数字排序、time-major PyG Batch 已由 loader 测试运行到断言处 | 固定节点数假设；未验证变网格；每步重复解析 HDF5，性能未量化 |
| GNN | 部分实现 | [`JorekGNNEncoder`](../src/models/gnn_encoder.py) | GAT+几何 edge_attr 的实现存在 | 输入契约仍为 5；未启用 checkpointing；9 场前向失败 |
| LNN | 部分实现 | [`JorekLNNProcessor`](../src/models/lnn_processor.py) | CfC 缺失时可 GRU fallback；接口支持 state/time_steps | 节点被视为独立长度 1 序列；固定 mesh；真实 9 场端到端未验证 |
| 解码器与 residual prediction | 部分实现 | [`JorekGNNDecoder`](../src/models/gnn_decoder.py)、[`JorekEvaluator`](../src/evaluation/benchmark.py) | decoder 有 5 场输出；benchmark 写有 `current + delta` | `output_dim` 未驱动 head 维度；9+5 相加不可能闭合；训练脚本有绝对输出路径 |
| 物理损失 | 部分实现 | [`JorekLoss`](../src/models/loss.py)、`JorekPhysicsLoss` | Huber/SmoothL1、边界项、部分质量/动量残差存在 | Model 502 的 9 场方程未完整实现；JorekLoss 仍 5 场；物理项训练默认/实际调用不统一 |
| 梯度隔离 | 未实现 | `src`/`tests` 未搜索到 `.detach()` 或定向 firewall | 无可验证行为 | 动量/连续性/温度表征没有 stop-gradient 设计 |
| 梯度检查点 | 未实现 | encoder/decoder 直接调用 GATConv | 无可验证行为 | hidden_dim=64 时不满足项目硬约束；OOM 风险未知但未被控制 |
| 训练流程 | 部分实现 | [`trainer.py`](../src/training/trainer.py)、`tests/train_interpolation_test.py`、`tests/train_sequence_pipeline.py` | 有单步 trainer 和两个训练脚本 | 主 sequence 入口仍 5 场/旧归一化；预训练 checkpoint 格式不一致；无成功日志可追溯 |
| 自动回归评估 | 部分实现 | [`JorekEvaluator`](../src/evaluation/benchmark.py) | hidden state warmup、dt、指标骨架存在 | benchmark 预处理/通道名仍 5 场；当前 decoder shape 会阻断 rollout |
| 数据完整性检查 | 已实现 | [`verify_data_integrity.py`](../tests/verify_data_integrity.py)、`check_mesh_consistency.py` | integrity 命令通过 104 文件，无 NaN/Inf/读取错误 | 检查未证明字段语义、物理量纲或训练可用性 |
| 可视化和报告 | 部分实现 | [`visualization.py`](../src/utils/visualization.py)、`report/`、`output/` | 原生三角拓扑可视化函数和大量产物存在 | 产物来源/配置/commit 未建立 provenance；9 场变量可视化未形成统一协议 |
| 测试与环境复现 | 未验证 | requirements、tests、compileall 结果 | 选定 5 个 smoke test 均运行并失败；数据完整性通过 | Python 3.14 环境、依赖版本、GPU、随机种子和 checkpoint 复现协议不固定 |

## 4. 端到端数据流审计

```text
HDF5 -> Parser -> Dataset -> Normalization -> Encoder
     -> LNN -> Decoder -> Residual Update -> Loss/Evaluation
```

- **HDF5 -> Parser**：`x` 从 `(2,4,1,N)`/兼容形状提取为 `[N,2]`；`values` 的 Model 502 期望至少 9 通道。parser 将原始 `[j,Phi,psi,omega,rho,Ti,vpar,rho_n,Te]` 重排为 `[psi,Phi,Z_j,omega,rho,T_i,v_par,rho_n,T_e]`，输出 `Data.x=[N,9]`。ITER 样本实际为 `x=(2,4,7369)`、`values=(9,4,1,7369)`；BEST 实际为 `values=(8,4,1,9327)`，因此默认 BEST 不满足 parser 的 Model 502 输入。状态：部分实现，解析 shape 已验证，语义/类型隔离未验证。
- **Parser -> Dataset**：`JorekSequenceDataset.get` 按文件名数字排序并 lazy parse；collate 输出长度为 `T` 的 `List[Batch]`，每步节点特征形如 `[B*N,9]`。实测 loader 输出 `[29476,9]`。状态：已实现但固定网格假设未解除。
- **Normalization**：插值脚本使用 9 场统计量，热力学索引为 4、5、7、8；physics loss 的 `denormalize` 同样使用 expm1。另一方面 evaluator 和 sequence pipeline 仍将 rho/Te 当作旧索引 2/3，sequence pipeline 只把 psi 乘 0.1。状态：部分实现，入口之间不一致。
- **Encoder**：设计输入应为 `[TotalNodes,9]`，当前 `JorekGNNEncoder.forward` 明确断言 5，GAT edge_attr 由 `pos[row]-pos[col]` 生成。EHL parser 输出 9 后，该断言直接失败。状态：9 场链路未通过。
- **LNN**：processor 将 `[B*N,latent]` reshape 为 `[B*N,1,latent]`，即每个节点以长度 1 调用 CfC/GRU；`state` 可以跨调用传递，`time_steps` 可传 dt。它不能处理不同节点数 batch。状态：接口部分实现，连续时间物理含义和多节点形状未充分验证。
- **Decoder**：当前双头固定生成 `[TotalNodes,5]`，排列 `[psi,Phi,rho,T,v_par]`，`output_dim` 只出现在构造参数而不改变 head。9 场 residual 需要 `[N,9]`，因此 decoder 是明确断点。状态：5 场实现存在，9 场未实现。
- **Residual update**：benchmark 使用 `next_pred_norm = current_pred_norm + pred_delta_norm`，并传递 hidden state；这符合 residual 形式的代码意图，但由于形状 `[N,9]+[N,5]` 不匹配且训练 sequence 直接将 decoder 输出作为 target，完整 residual/dt 契约未验证。
- **Loss/Evaluation**：`JorekPhysicsLoss` 可按 9 场反归一化，但 PDE forward 取 `rho=[:,2]`、`v=[:,4]` 等旧 5 场位置；`JorekLoss` 的回归权重和诊断固定 5。evaluator 的 `CHANNEL_NAMES` 也是 5 个。状态：只能视为旧 Reduced-MHD 部分实现。

## 5. Model 502 物理一致性审计

| 物理/架构要求 | 当前实现 | 证据 | 状态 | 修复或验证建议 |
|---|---|---|---|---|
| 9 场顺序 | parser 重排正确；后续模块多数为 5 场 | [`parse_to_graph`](../src/data/jorek_h5_parser.py)、encoder、decoder、loss | 部分实现 | 建立单一 `FIELD_NAMES`/索引常量，所有模块 shape 断言和测试共用 |
| thermodynamic log-space | 插值训练和 physics denormalize 覆盖 `[rho,Ti,rho_n,Te]` | [`train_interpolation_test.py`](../tests/train_interpolation_test.py)、[`JorekPhysicsLoss.denormalize`](../src/models/loss.py) | 部分实现 | 统一 Dataset transform；sequence/evaluator 改用同一实现并做 round-trip 测试 |
| linear Z-score | 插值脚本对非热力学场未 log；但 sequence pipeline 非标准 | 同上、[`train_sequence_pipeline.py`](../tests/train_sequence_pipeline.py) | 部分实现 | 统计量只由训练 split 计算；逐字段测试负值保持线性 |
| Namelist conditioning | 提取并挂载四个参数，未进网络 | `Data.namelist_params`；无 condition MLP 调用 | 部分实现 | 定义缺失值策略、MLP 输入 shape、注入点和 ablation |
| residual/dt | benchmark 有 `current + delta` 和 CfC dt；训练/decoder未形成 9 场闭环 | [`benchmark.py`](../src/evaluation/benchmark.py) | 部分实现 | decoder 明确输出 `delta/dt` 或统一在一个地方除 dt；单步数值契约测试 |
| hidden state 连续传递 | evaluator/interpolation 显式复用 state；trainer 每步初始化/不传 state | [`lnn_processor.py`](../src/models/lnn_processor.py)、[`trainer.py`](../src/training/trainer.py) | 部分实现 | 明确 teacher-forcing、detach BPTT 边界；增加 state continuity test |
| PDE residual scaling | 有 `residual*dt/(abs(reference)+eps)` 的质量/动量项 | `JorekPhysicsLoss`、evaluator | 部分实现 | 按 Model 502 九方程逐项定义量纲、reference、dt；用解析/合成场校验 |
| gradient firewall | 未找到 `.detach()`/分支隔离 | `src`/`tests` 搜索无结果 | 未实现 | 先定义梯度图和每项 loss 允许更新的模块，再用 grad hook/范数测试验收 |
| Huber/SmoothL1 | JorekLoss 用 Huber，PhysicsLoss 用 SmoothL1 | [`loss.py`](../src/models/loss.py) | 已实现 | 保留；补通道级 9 场权重和极端值测试 |
| checkpointing | 未导入/调用 `torch.utils.checkpoint` | encoder、decoder | 未实现 | 将 message passing 封装为可检查点 block；CPU 等价梯度测试和显存基准作为门槛 |

## 6. 测试、实验与可复现性

- `python tests/verify_data_integrity.py`：**通过**，104 个 HDF5 文件无 NaN、Inf、读取错误；网格 rank-1 shape 一致性警告未触发。
- `python -m pytest ... -q`（5 个核心 smoke test）：**失败，5 failed，343 warnings，9.81s**。`test_dataset_loader` 期望 5 但收到 9；其余 pipeline 在 encoder 的 `Expected 5 input features ... got 9` 处失败。
- `python -m compileall -q src tests`：**未完全通过**；`src/data/tempCodeRunnerFile.py` 第 1 行 `IndentationError`。
- `inspect_parsed_jorek_data.py`：**可运行**，输出 EHL/ITER 图为 `[N,9]`，并显示 `namelist_params` 四项均为 1.0；这更像回退值，不能证明从 namelist 读取成功。
- 单样本过拟合：**未验证**。仓库有训练脚本和测试意图，但当前 5 场断点阻止将其视为完成。
- 数据加载：真实 HDF5 读取和完整性已验证；Model 502/Model 600 混用和默认 data_dir 不兼容仍未解决。
- 模型前向/反向：当前 9 场 smoke 前向未通过；9 场反向和梯度隔离未验证。
- 自动回归 rollout：**未验证**；benchmark 代码包含 shape 断点。
- 质量守恒/物理指标：代码有质量/动量 PDE 诊断，但没有当前可复现的指标日志或通过阈值。
- checkpoint、日志、指标：`output/` 与 `output_p1/` 存在大量文件（审计时统计 5725 个、约 6.7 GB），但未建立生成 commit、配置、随机种子和 schema 的 provenance，不能据此断言结果属于当前代码。
- 环境风险：`requirements.txt` 使用宽版本范围；当前运行环境为 Python 3.14（由 pytest 路径可见），GPU/驱动状态、精确包版本、随机种子均未形成复现记录。预训练路径在 `config.yaml` 中指向 `checkpoints_good_1750/model_epoch_500.pt`，是否存在及是否与当前 9 场结构匹配未验证。

## 7. 问题与风险清单

### P0

- **9 场 parser 与 5 场模型契约冲突**。证据：parser 输出 `[N,9]`，encoder 断言 5，decoder 固定 5；5 个 smoke test 已失败。影响所有默认训练、前向和 rollout；触发任意 Model 502 文件进入模型。动作：冻结唯一 9 场 schema，端到端统一。验收：9 场单步 forward/backward 和 residual shape 测试通过。
- **默认数据目录不是 Model 502**。证据：`config.yaml` 指向 `data/test_data_for_BEST`，实测 `values` 8 通道；parser 要求至少 9。影响默认启动。验收：配置显式选择 Model 502 数据并启动到首个 batch，或实现明确的 Model 600 adapter。

### P1

- **训练闭环断裂**。sequence pipeline 使用旧 5 场归一化并直接训练绝对输出，interpolation pipeline 才接近 9 场 residual；checkpoint schema 不一致。验收：单一命令完成 train/eval/checkpoint reload。
- **物理 loss 不是完整 Model 502 loss**。`JorekPhysicsLoss` 9 场 denormalize 后仍以旧索引计算 rho/T/v，且只覆盖质量/并行动量子集。验收：九场索引和方程映射表逐项测试，启用物理项的梯度与量纲测试通过。
- **Namelist 未进入模型**。只解析，不编码/注入，无法研究条件泛化。验收：改变条件而固定状态时模型输出/latent 对条件有可检测响应，并有 no-condition ablation。
- **hidden state 语义不统一**。evaluator 复用 state，trainer/部分脚本不一致；LNN 实际每节点长度 1。验收：连续两段 rollout 与一次完整 rollout 的 state/输出一致性在容差内。
- **OOM 防护缺失**。hidden_dim=64 的 GAT encoder/decoder 未使用 checkpointing。验收：目标显存下固定 batch/mesh 的峰值显存低于预算，checkpoint 与非 checkpoint 梯度数值一致。

### P2

- `tempCodeRunnerFile.py` 语法错误污染全库 compileall。
- 测试仍写死 5 场，不能作为当前基线保护网。
- 宽松依赖版本、无锁定环境、无随机种子和数据 split manifest，复现风险高。
- lazy HDF5 每步重复打开文件，`num_workers=4` 的 IO/Windows 行为和性能未基准化。
- 输出产物没有与 checkpoint、config、commit 关联，报告可信度不足。

### P3

- Cross-Attention/大 Latent Bus、源项、复数 3D 频谱和 JOREK runtime 接口均属研究增强，当前应延后。
- 跨网格、跨装置、毫秒级实时和长程数字孪生目标尚无基准数据或验收阈值，不宜写成近期完成承诺。

## 8. `new_version.md` 路线图评估

| 阶段 | 当前完成度 | 前置条件 | 主要风险 | 是否建议进入 | 调整意见 |
|---|---:|---|---|---|---|
| Phase 1：9 场基础 | 约 30%（静态实现，不是可用完成度） | 统一 schema、Model 502 数据、归一化测试、可运行基线 | 当前核心入口全是 5/9 冲突；OOM 和条件注入缺失 | 否，先完成 | 将 1.1/1.2 拆成 schema、数据适配、transform、条件有效性四项；1.3 增加验收门槛 |
| Phase 2：MoE | 约 5%（主要是设计） | Phase 1 全部验收、单体 benchmark、物理指标基线 | 过早引入专家和 Cross-Attention，难区分数据/架构收益 | 否 | 先做单体消融和梯度图，再拆专家；Cross-Attention 延后到简单握手 baseline 后 |
| Phase 3：源项 | 设计完成 | 源项字段真实存在且语义/单位已确认；Forced ODE baseline | 当前 HDF5 parser 尚未有源项 schema；热猝灭指标未定义 | 否 | 先做源项可用性审计和离线 replay，明确源项量纲、时间对齐、稳定性阈值 |
| Phase 4：3D 频谱 | 设计完成 | 2D 长 rollout 稳定、模态数据和复数规范、JOREK 接口协议 | 复杂度、数据规模、复数梯度和接口风险均未估算 | 否 | 拆成模态数据契约、实值等价 baseline、复数算子、接口 sandbox；删除“掌握/替代”式无指标表述 |

任务处置：

- 1.1 **拆分**：先固定字段 schema/Model 502 检测，再实现统一 transform，再做 parser fixture。
- 1.2 **拆分**：namelist extraction、缺失策略、Condition MLP、sequence loader 分开验收；`SEQ_LEN=20` 不应先于单步闭环硬编码。
- 1.3 **保留但重定义**：目标是可运行 9 场 residual baseline、单样本过拟合、checkpoint reload、峰值显存预算，而非仅“升级旧网络”。
- 2.1 **延后并拆分**：先做单体分支/梯度消融，再建三个专家；条件模块必须在单体先验证。
- 2.2 **延后**：先用拼接/加法 latent handshake 建立最低复杂度基线，Cross-Attention 作为可选消融。
- 2.3 **保留并前置一部分**：梯度 firewall 应在单体物理 loss 前完成，benchmark 先于 MoE 消融。
- 3.1 **需要重新定义**：先证明 H5 字段存在、单位、时间对齐和 ground truth；不存在则停止该任务。
- 3.2 **延后**：forced ODE 需在无源单体稳定后进入，Huber 调参不能替代数据/时间对齐验证。
- 3.3 **拆分**：先定义热猝灭、电磁拓扑和守恒指标，再做短扰动 replay，最后才做强源项。
- 4.1 **拆分并延后**：先模态 shape/real-imag 规范和实值 baseline，再复数运算与能量级联。
- 4.2 **延后**：JOREK runtime 替代和逆向工程不是当前阶段可验收目标；需要接口 sandbox、延迟/稳定性/误差预算。

路线图还缺少：数据版本与 split、字段单位、统计量生成协议、单步/单样本/rollout 指标、显存预算、checkpoint schema、失败停止条件、实验 provenance。月份估计在未完成 Phase 1 验收前不可验证；研究目标（跨装置、数字孪生、能量级联）与工程任务混杂，应拆成 gated research hypotheses。

## 9. 修订后的执行路线图

### Task 0.1：冻结 Model 502 数据契约

- **目标**：唯一确认 `[psi, Phi, Z_j, omega, rho, T_i, v_par, rho_n, T_e]` 和 raw-to-field 映射。
- **前置条件**：可读取 ITER/EHL Model 502 fixture。
- **涉及文件或模块**：`src/data/jorek_h5_parser.py`、`src/physics/definitions.md`、新增/更新数据契约测试。
- **输入**：至少一个 9 通道 HDF5 和字段说明。
- **输出**：字段常量、shape/type 断言、schema 文档。
- **实现要求**：拒绝 8 通道文件或显式分流 Model 600；禁止模块私有索引。
- **验证命令**：`python tests/inspect_parsed_jorek_data.py`；新增 `python -m pytest tests/test_schema.py -q`。
- **量化验收标准**：9 个字段逐项映射测试通过；parser 输出 `[N,9]`；错误数据在首个解析点失败。
- **失败时停止条件**：字段语义或 raw index 无法由数据/Model 502 资料确认。
- **风险与回滚方案**：保留旧 parser 只读分支，禁止静默截断通道。
- **完成定义**：所有 downstream 只引用同一 schema。

### Task 0.2：统一归一化、条件和时间元数据

- **目标**：把 log/Z-score、namelist、dt 变成 Dataset 输出的稳定契约。
- **前置条件**：Task 0.1。
- **涉及文件或模块**：`src/data/dataset.py`、parser、训练和 evaluator。
- **输入**：训练 split 原始状态、namelist、`time`。
- **输出**：`x_norm`、`mean/std`、condition tensor、per-step dt。
- **实现要求**：热力学 `[4,5,7,8]` 用 `log1p(relu)`；其余线性；条件缺失有显式 mask/状态；统计量只用 train split。
- **验证命令**：新增 normalization round-trip 和 negative-value tests。
- **量化验收标准**：round-trip 相对误差小于 `1e-5`（非 clamp 区域）；dt 与时间差一致；条件 shape 固定为 4。
- **失败时停止条件**：任一入口仍按 `[2,3]` 处理热力学场。
- **风险与回滚方案**：保存 stats schema version，旧 checkpoint 不自动加载。
- **完成定义**：训练、loss、benchmark 共用同一 transform。

### Task 1.1：跑通 9 场单体 residual baseline

- **目标**：encoder/LNN/decoder 均接收并返回 9 场，输出定义为 `delta_x/dt`。
- **前置条件**：Task 0.1-0.2。
- **涉及文件或模块**：`src/models/gnn_encoder.py`、`lnn_processor.py`、`gnn_decoder.py`、`src/training/trainer.py`。
- **输入**：单个 Model 502 图对和 dt。
- **输出**：`pred_delta_norm [N,9]`、`next_norm [N,9]`。
- **实现要求**：移除私有 5 场假设；显式检查 edge/pos/batch；state 只在时间边界 reset。
- **验证命令**：`python -m pytest tests/test_single_step_9field.py -q`。
- **量化验收标准**：CPU forward/backward 无 NaN；所有 shape 为 9；checkpoint reload 后输出一致到 `1e-6`。
- **失败时停止条件**：出现字段错位、NaN 或 residual shape mismatch。
- **风险与回滚方案**：先以 9 场 monolithic 分支替换旧入口，保留旧 5 场实验只读。
- **完成定义**：单命令可完成一批训练 step 和 reload。

### Task 1.2：实现显存预算与 checkpointing

- **目标**：在大于等于 64 hidden dim 下控制 GAT 激活显存。
- **前置条件**：Task 1.1 的数值基线。
- **涉及文件或模块**：encoder/decoder message-passing blocks。
- **输入**：固定 mesh、batch、hidden_dim=64。
- **输出**：checkpoint/non-checkpoint 模式和显存报告。
- **实现要求**：使用 `torch.utils.checkpoint`，保持训练/推理语义区分。
- **验证命令**：新增 checkpoint gradient equivalence test；GPU benchmark 命令记录峰值显存。
- **量化验收标准**：梯度最大绝对差小于 `1e-5`；在约定 GPU/批量下峰值显存低于预算。
- **失败时停止条件**：数值不等价或仍超预算。
- **风险与回滚方案**：配置开关，默认在基线稳定后启用。
- **完成定义**：报告包含硬件、batch、mesh、峰值显存和耗时。

### Task 1.3：单样本过拟合与短序列 BPTT

- **目标**：证明数据、状态、residual、反向和 loss 形成闭环。
- **前置条件**：Task 1.1、1.2。
- **涉及文件或模块**：`trainer.py`、训练脚本、tests。
- **输入**：固定一个小图/短序列，`SEQ_LEN=3` 起步，再扩到 20。
- **输出**：loss 曲线、checkpoint、seed/config manifest。
- **实现要求**：teacher forcing 与 free-run 分开；hidden state 传递策略写入测试。
- **验证命令**：`python -m pytest tests/test_single_sample_overfit.py -q`。
- **量化验收标准**：固定 seed 下训练 loss 下降至少 99%，无 NaN；reload 后曲线可复现。
- **失败时停止条件**：loss 不降、state 非连续或输出字段错位。
- **风险与回滚方案**：先禁用 physics loss，再逐项启用并记录增量。
- **完成定义**：单样本过拟合和短序列训练均有可重跑产物。

### Task 1.4：建立物理指标与单体 benchmark

- **目标**：在进入 MoE 前建立 persistence baseline、场误差、质量漂移、PDE residual 基线。
- **前置条件**：Task 1.3；完成 9 场 PDE 索引审计。
- **涉及文件或模块**：`src/models/loss.py`、`src/evaluation/benchmark.py`、`run_benchmark.py`。
- **输入**：held-out Model 502 序列、训练 stats、checkpoint。
- **输出**：逐场误差、rollout 曲线、守恒和 residual 指标。
- **实现要求**：9 场名称和物理 reference 一致；报告 checkpoint/config/commit。
- **验证命令**：`python -m pytest tests/test_rollout_9field.py -q`；再执行 benchmark CLI。
- **量化验收标准**：至少 1 个 free-run horizon 相对 persistence 改善；指标无 NaN；质量漂移阈值预先登记。
- **失败时停止条件**：rollout shape 错、指标缺 reference 或只比训练集。
- **风险与回滚方案**：先关闭 PDE 项作为数据基线，逐项回归。
- **完成定义**：单体 baseline 有可复核表格，作为 MoE 对照。

### Task 2.1：梯度图与物理 firewall 消融

- **目标**：在单体网络先证明 `.detach()` 的方向和收益。
- **前置条件**：Task 1.4。
- **涉及文件或模块**：loss、trainer、模型分支。
- **输入**：数据项、PDE 项、边界项及模块参数梯度。
- **输出**：梯度范数矩阵和 ablation 结果。
- **实现要求**：明确温度/动量/连续性允许更新的参数集合。
- **验证命令**：新增 gradient routing tests。
- **量化验收标准**：禁止路径梯度严格为 0；允许路径非零且与数值基线一致；训练指标不劣于登记阈值。
- **失败时停止条件**：无法解释的梯度泄漏或 PDE 项破坏数据基线。
- **风险与回滚方案**：配置化 firewall，默认关闭新分支直至消融完成。
- **完成定义**：有梯度图测试和可比较 ablation 表。

### Task 2.2：MoE 专家与最小握手

- **目标**：在单体基线可复现后拆 EM/Thermo/Momentum 专家。
- **前置条件**：Task 2.1、Task 1.4。
- **涉及文件或模块**：新增 expert modules、共享 condition encoder、trainer。
- **输入**：同一 9 场数据和条件向量。
- **输出**：专家 latent、融合 latent、消融 checkpoint。
- **实现要求**：先 concat/add MLP，Cross-Attention 只能作为后续实验；每个专家输出 shape 和梯度边界固定。
- **验证命令**：专家单测、参数量/显存报告、MoE ablation CLI。
- **量化验收标准**：在相同 seed/split 下相对单体的误差、稳定性、显存均有登记；不能只报告训练 loss。
- **失败时停止条件**：无法优于单体且成本超过预算，暂停复杂握手。
- **风险与回滚方案**：保留 monolithic checkpoint 作为默认 fallback。
- **完成定义**：MoE 结果可由单体基线复现比较。

### Task 3.1：源项可用性和 Forced ODE replay

- **目标**：先确认源项字段、单位、时间对齐，再做受控离线 replay。
- **前置条件**：Task 1.4；真实含源项 HDF5 或明确补充数据。
- **涉及文件或模块**：parser、dataset、processor、physics definitions。
- **输入**：带源项的序列及源项 ground truth。
- **输出**：source schema、forced baseline、极端梯度指标。
- **实现要求**：无源数据不能用零填充冒充；记录单位和时间插值。
- **验证命令**：source schema test、replay test、短扰动 benchmark。
- **量化验收标准**：源项 round-trip/时间对齐通过；强源项下无 NaN，指标相对无源 baseline 可解释。
- **失败时停止条件**：字段不存在、单位不明或时间无法对齐。
- **风险与回滚方案**：只读 source adapter，不改变无源 baseline。
- **完成定义**：源项数据闭环和失败阈值明确。

### Task 4.1：3D 模态数据契约与实值 baseline

- **目标**：在复数网络前固定模态、实虚部、截断和能量指标。
- **前置条件**：2D 长 rollout 稳定；3D 模态数据可用。
- **涉及文件或模块**：新 parser/schema、频谱数据集、benchmark。
- **输入**：含 `n=0..K` 的真实数据。
- **输出**：实值等价 baseline、模态能量报告。
- **实现要求**：先验证 shape 和能量守恒，再引入复数算子。
- **验证命令**：modal parser/unit tests、energy reconstruction test。
- **量化验收标准**：模态重构误差和能量误差阈值预登记；无模态置换。
- **失败时停止条件**：模态语义或复数约定无法确认。
- **风险与回滚方案**：保留 2D 单体/实值模态 baseline。
- **完成定义**：复数扩展具备可测量基准。

## 10. 建议的最小可行里程碑

- **未来 1 周**：交付 9 场 schema、Model 502 fixture、统一 normalization、更新 smoke tests。验证：`python -m pytest tests/test_schema.py tests/test_normalization.py -q`。通过标准：9 场 shape/顺序和 round-trip 全通过；不通过则冻结模型扩展，继续修契约。
- **未来 1 个月**：交付可 reload 的单体 9 场 residual baseline、单样本过拟合、短序列 BPTT、checkpoint 显存报告和 persistence benchmark。验证：`python -m pytest tests/test_single_step_9field.py tests/test_single_sample_overfit.py tests/test_rollout_9field.py -q`。通过标准：forward/backward 无 NaN、单样本 loss 至少下降 99%、free-run 指标和质量阈值可复现；不通过则不得开始 MoE。
- **下一阶段**：交付梯度 firewall 消融、条件注入 ablation 和 MoE 最小融合 baseline。验证：统一 seed/split 的 MoE benchmark 命令。通过标准：相对单体的精度、稳定性、显存、吞吐有完整对照；不通过则回滚到单体并停止 Cross-Attention。

## 11. 审计结论

1. **最可信成果**：Model 502 HDF5 的 9 场 parser、原生图拓扑和序列加载基础；其证据是实际 `[N,9]` 输出、104 文件完整性通过和 loader 已运行到维度断言。
2. **最可能阻断研究进展的问题**：9 场 parser 与 5 场模型/损失/测试/benchmark 的契约分裂，直接阻断默认 forward 和 rollout。
3. **是否具备 MoE 条件**：不具备。单体 9 场 baseline、单样本过拟合、统一指标和显存/梯度验证均未完成。
4. **`new_version.md` 最需修改的三处**：把 Phase 1 从“一个月打通”改为 gated schema/闭环任务；把 MoE/Cross-Attention 延后到单体 benchmark 与 gradient firewall 之后；把源项和 3D 任务改为先验证数据契约、单位、指标再开发。
5. **唯一最高优先级行动**：冻结并贯通 9 场 schema，从 parser 到 decoder/loss/test 建立一个能 forward、backward、reload 的 Model 502 单体 residual baseline。

## 12. 证据索引

- [new_version.md](../new_version.md)：Phase 1 至 Phase 4 原路线图。
- [README.md](../README.md)：旧版 5 场架构与已完成声明。
- [current_version.md](../current_version.md)：当前实现描述及 5 场/9 场迁移说明。
- [config.yaml](../config.yaml)：默认数据目录、9 场配置、训练参数和 checkpoint 路径。
- [jorek_h5_parser.py](../src/data/jorek_h5_parser.py)：HDF5、9 场重排、namelist 和图构建。
- [dataset.py](../src/data/dataset.py)：滑窗数据集与 `collate_temporal_graphs`。
- [gnn_encoder.py](../src/models/gnn_encoder.py)：5 场输入断言和 GAT encoder。
- [lnn_processor.py](../src/models/lnn_processor.py)：CfC/GRU、shape reshape、state/dt 接口。
- [gnn_decoder.py](../src/models/gnn_decoder.py)：固定 5 场双头 decoder。
- [loss.py](../src/models/loss.py)：Huber、SmoothL1、反归一化和部分 PDE residual。
- [trainer.py](../src/training/trainer.py)：单步训练接口。
- [benchmark.py](../src/evaluation/benchmark.py)：5 场 evaluator、rollout 和物理指标骨架。
- [run_benchmark.py](../src/evaluation/run_benchmark.py)：配置、checkpoint 和 benchmark CLI。
- [verify_data_integrity.py](../tests/verify_data_integrity.py)：104 文件完整性检查。
- [test_dataset_loader.py](../tests/test_dataset_loader.py)、[test_perception_pipeline.py](../tests/test_perception_pipeline.py)、[test_cognition_pipeline.py](../tests/test_cognition_pipeline.py)、[test_reconstruction_pipeline.py](../tests/test_reconstruction_pipeline.py)、[test_training_and_vis.py](../tests/test_training_and_vis.py)：当前 5 场 smoke test 证据。
- [train_interpolation_test.py](../tests/train_interpolation_test.py)、[train_sequence_pipeline.py](../tests/train_sequence_pipeline.py)：两套不一致的训练入口。
- [definitions.md](../src/physics/definitions.md)：Model 502 方程与字段说明。
- [report/](.)：已有报告目录和本审计报告。

## 13. 未解决问题

- BEST 8 通道 Model 600 与 Model 502 9 场数据是否需要双 adapter，还是应从默认训练集合移除，需用户确认。
- HDF5 中四个 namelist 参数是否真实存储、当前 1.0 是否为回退值，需提供原始 namelist 或字段规范确认。
- `Z_j`、`omega`、`rho_n` 的精确单位、归一化常数和 Model 502 源码映射仍需物理专家确认。
- 当前 output/checkpoint 是否由本分支、当前配置和当前代码产生，无法仅凭目录确认。
- 真实 GPU 型号、驱动、峰值显存预算和允许的 batch/mesh 未提供，OOM 结论只能标为风险而非实测失败。
- 无源/含源数据的时间步是否来自真实 `t_now`、是否需要使用 per-edge/per-frame dt，尚未完成协议确认。
- 质量、能量、磁通拓扑和热猝灭的验收阈值没有在仓库中定义，需要研究方登记。
- 3D 模态数据、JOREK runtime 接口、跨装置数据和外部实验结果未在本次环境中验证。
- `tempCodeRunnerFile.py` 是否为临时用户文件、是否应移出源码扫描范围，需用户决定；本审计未删除或修改它。
