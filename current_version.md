**一、总体架构（HyPhy‑GLNN 当前实现状态）**

- 目标：在 JOREK 原生非结构化有限元网格上，构建一个 `GNN Encoder → LNN Processor → GNN Decoder` 的时序模型，预测 5 个主变量随时间的演化（ψ, Φ, ρ, T, v_par），并给出破裂/猝灭相关的诊断（破裂概率、物理残差等）。
- 数据流：
  - JOREK `.h5` → 解析为 PyG 图（节点、边、边属性、边界标记、物理常数等）。
  - 时间维度上用滑动窗口构造序列 `[..., Graph_t, Graph_{t+1}, ...]`。
  - 每一帧的节点特征先做物理感知（GNN），再做时序认知（LNN/CfC），最后重构回 5 个状态变量（GNN Decoder）并附带破裂概率。
- 物理约束：
  - 磁场 B 与速度 v 不直接预测，而是由 ψ、Φ 经可微分算子和 Reduced MHD Ansatz 计算。
  - 梯度、散度、Poisson bracket 等在非结构化网格上通过图算子近似。

---

**二、数据与预处理层**

1. **JOREK HDF5 解析（原生网格 → 图）**  
   主要在 jorek_h5_parser.py 的 `JorekDataParser` 中实现：

   - 读取节点坐标 `x`，抽取 (R, Z)，得到节点坐标矩阵 `pos ∈ [Num_nodes, 2]`。
   - 网络“无栅格化”：完全不插值到笛卡尔网格，而是直接使用 JOREK 的 `contain_node` / `vertex` / `neighbours` 构建 `edge_index`：
     - 对每个有限元，把内部节点两两连边形成局部完全图。
     - 如果没有 contain_node，就回退到 vertex 或 neighbours。
   - 读取边界信息：
     - `boundary` 数组被挂到 `Data.boundary` 上，用于之后的内部区/边界区加权和边界损失。
     - 自由边界标志 `freeboundary` 解析成布尔标量。
   - 读取 5 个主变量并组装节点特征 `x ∈ [Num_nodes, 5]`：
     - 索引按 JOREK 的内部顺序重排为 `[psi, Phi, rho, T (用 Ti 代表), v_par]`。
   - 读取时间戳 `t_now` ⇒ `Data.time`。
   - 读取并缓存规范化相关物理常数 `sqrt_mu0_rho0`, `sqrt_mu0_over_rho0`, `F0` 到 `Data.norm_factors`（用于 B 场/速度场等物理算子）。
   - 若存在 `vertex`，拆成两个三角形生成 `triangles`，用于可视化时严格复现 JOREK 拓扑。
   - 计算 `edge_attr`：
     - 每条边的 `(ΔR, ΔZ, distance)`，为后续 GNN 卷积和梯度/散度算子提供几何信息。

2. **序列数据集与时间滑动窗口**  
   在 dataset.py 中的 `JorekSequenceDataset` 与 `collate_temporal_graphs`：

   - 扫描指定目录下的 `jorek*.h5`，通过文件名中的数字排序，保证时间单调。
   - 使用滑动窗口生成样本：长度为 `seq_len` 的子序列，`len = N_files - seq_len + 1`。
   - `__getitem__` 返回的是一个时间有序的 `List[Data]`：`[Graph_t, Graph_{t+1}, ..., Graph_{t+seq_len-1}]`，不打乱时间。
   - 在 `get` 中进一步补充：
     - 确保每步的 `freeboundary` 以统一标量张量形式存在。
     - 从 HDF5 中优先读取 `psi_axis`, `psi_bnd`（若缺失，则退化为场的 min/max），挂到每个 Data 作为磁通归一化参考（用于 core/SOL 区域判定、边界惩罚等）。
   - 自定义 `collate_temporal_graphs(batch)`：
     - 输入：`batch` 是若干“样本”，每个样本是 `List[Data]`（长度为 seq_len）。
     - 输出：长度为 `seq_len` 的列表，每个元素是一个 PyG `Batch`，代表“同一时间步上所有 batch 样本的拼接图”。
     - 这样保留了时间轴结构，又兼容 PyG 的批次拼接方式。

3. **数据检查与可视化工具**  
   - check_data_stats.py：  
     - 直接用 `JorekDataParser` 解析一个 HDF5，打印 5 个通道的 min/max/mean/std，排查 NaN、量纲异常。
   - check_mesh_consistency.py：  
     - 扫描目录内所有 `.h5`，检查网格拓扑、节点数、values 维度是否在时间上保持一致。
   - inspect_parsed_jorek_data.py：  
     - 批量解析若干 HDF5，详细打印 Data 对象的各个字段，便于调试 parser 是否挂齐了所有信息。

---

**三、模型层：GNN‑LNN‑GNN**

1. **GNN Encoder（Perception）**  
   gnn_encoder.py 的 `JorekGNNEncoder`：

   - 输入：`x ∈ [Total_nodes, 5]`（严格断言输入通道为 5），`edge_index`，`pos`。
   - 通过 `edge_attr = pos[row] - pos[col]` (ΔR, ΔZ) 引入几何信息到 GATConv（`edge_dim=2`），打破纯拓扑对称性。
   - 结构：若干层 `GATConv + LayerNorm + SiLU`，最后一层输出 `latent_x ∈ [Total_nodes, LATENT_DIM]`，作为每个节点的空间“物理 embedding”。

2. **LNN Processor（Cognition）**  
   lnn_processor.py 的 `JorekLNNProcessor`：

   - 优先使用 `ncps.torch.CfC` 作为 liquid neural network，如果环境缺少库则降级为 `GRU`，保证管线可跑。
   - 解决 PyG 与 RNN 在张量形状上的冲突：
     - PyG: `[Total_nodes, latent_dim]`，`batch_index ∈ [Total_nodes]`。
     - LNN 期望: `[Batch_size, Seq_len, latent_dim]`。
     - 当前实现假设**相同网格拓扑**，把 `Total_nodes` 重新整理成 `[Batch_size, Num_nodes, latent_dim]`，每个节点相当于时序长度为 1 的序列，再在时间维度上迭代使用（完整时间展开在训练/诊断脚本中通过 for 循环实现）。
   - 维护 `hidden_state`：  
     - 接收可选 `state` 作为上一时间的隐状态输出，下一个时间步沿用，用于实现“无遗忘”的连续时间建模。

3. **GNN Decoder（Reconstruction + Disruption Head）**  
   gnn_decoder.py 的 `JorekGNNDecoder`：

   - 输入：`latent_x ∈ [Total_nodes, LATENT_DIM]`，以及 `edge_index`, `batch_index`, `pos`。
   - 共享基座 + 双头结构：
     - Shared：一层 GATConv 提取高阶空间特征。
     - Head 1（Smooth fields）：若干 GATConv 预测 `[psi, Phi, v_par]`，主要针对相对平滑的场量。
     - Head 2（Thermodynamic fields）：更深更宽的 GATConv + MLP 预测 `[rho, T]`，专注台基陡峭梯度。
   - 输出：
     - `state_vars ∈ [Total_nodes, 5]`，按 `[psi, Phi, rho, T, v_par]` 排列。
     - `p_disrupt ∈ [Batch_size, 1]`：图级破裂概率 logits，由 `global_mean_pool(x_shared)` 后接 MLP 得到。
   - 正定性策略：
     - 代码中明确记录：因为训练在 Z‑score 空间，网络输出允许为负值；真正的物理正定性（ρ, T > 0）放在**反归一化之后**保证，而不在 decoder 里硬套 `Softplus`，避免干扰归一化残差学习。

4. **物理约束层（Hard Constraints）**  
   - hard_constraints.py 的 `ReducedMHDConstraints`：
     - 内部用 `GraphSpatialGradient` 对 ψ、Φ 做图上的有限差分梯度估计。
     - 依据 Reduced MHD 公式构造：
       - B：`B = (F0/R) e_phi + (1/R) ∇ψ × e_phi`，输出 `[B_R, B_Z, B_phi]`。
       - v：`v = -R ∇Φ × e_phi + v_par B`，输出 `[v_R, v_Z, v_phi]`。
   - mhd.py、cocos8.py：
     - 提供更底层的 `get_magnetic_field`, `get_velocity_field`, `poisson_bracket`, `reconstruct_velocity` 等函数，配合梯度算子使用。

5. **梯度/散度算子（在非结构化网格上）**  
   grad_ops.py：

   - `compute_gradient`：基于边上的 `(ΔR, ΔZ, distance)` 和节点上的 `f_j - f_i`，做距离加权的有限差分，并按节点 scatter_add 回去，近似 ∇f。
   - `compute_divergence`：对向量场 F 同样利用邻边差值和几何权重，近似 div F。
   - 都严格工作在图/有限元拓扑上，未做任何栅格化。

---

**四、损失函数与物理一致性**

1. **综合破裂预测损失 `JorekLoss`**  
   loss.py 中 `JorekLoss`：

   - 组成：
     - 回归：使用 `HuberLoss` (`delta` 可调)，对 5 个变量逐节点计算，支持节点加权和通道加权。
     - 分类：`BCEWithLogitsLoss` 用于破裂概率 logits。
     - 边界项：根据 `boundary` 和 `freeboundary` 调整内部/边界掩膜；对边界 ψ 通常给更大权重（通过 `lambda_bc`）以锚定磁面的物理位置。
     - 预留 PDE 残差权重 `lambda_phys`，方便未来进一步耦合物理项。
   - 通道权重：
     - `channel_weights` 以 buffer 形式注册，确保在 `model.to(device)` 时跟随移动；解决 ψ 与 Φ 等量级差异可能导致的小变量被忽略的问题。

2. **物理插值/残差损失 `JorekPhysicsLoss`**  
   loss.py 的 `JorekPhysicsLoss`：

   - 面向例如 train_interpolation_test.py 这种“给定两帧做物理插值/残差学习”的任务。
   - 归一化与反归一化：
     - 假设输入 `pred` 和 `target` 已在训练脚本中做了 `log1p(ReLU(rho/Te)) + Z‑score`。
     - 内部 `denormalize` 先做 Z‑score 反变换，再对 `rho, Te` 做 `expm1` 并 clamp 至正值，得到真正物理量。
   - PDE 残差：
     - 质量守恒：利用 `compute_gradient`, `compute_divergence`, `get_magnetic_field`, `reconstruct_velocity` 构造离散化的质量连续方程残差 `res_mass`，并通过除以 `rho` 再乘 `dt` 做成**无量纲相对误差**。
     - 并行动量：类似地构造 `res_momentum`，用 `rho * v_ref` 归一化，避免刚性区的量纲爆炸。
   - 边界项：
     - 使用 `boundary` 掩膜、`psi_axis/psi_bnd`、`freeboundary` 信息，对边界 ψ 和等离子体其它变量施加强边界惩罚，保证重建出的场在边界处贴合 JOREK 结果。

---

**五、训练循环与物理诊断**

1. **单步训练管理器 `JorekTrainer`**  
   trainer.py：

   - 维护 encoder / processor / decoder / physics_layer + AdamW 优化器。
   - `train_step` 与 `eval_step`：
     - 前向：`x → encoder → latent → processor → decoder → pred_states, pred_p_disrupt`。
     - 节点加权：通过 `compute_spatial_weights`（在 loss 模块中）对磁轴、台基等区域赋不同权重（例如 axis_weight, pedestal_weight）。
     - 损失：交给 `criterion`（一般是 `JorekLoss`）处理。
     - 反向：`total_loss.backward()` + 三个子网络的梯度裁剪 `clip_grad_norm_`，缓解破裂阶段的梯度爆炸。
   - 当前训练范式：
     - `trainer` 本身是“单时间步预测 t → t+1”的接口。
     - 更长序列的时间展开（多步 teacher forcing / 自回归）在外层训练脚本或诊断脚本中通过 for 循环来实现。

2. **可靠性与物理保真度验证脚本**  
   verify_reliability.py：

   - 使用 `JorekSequenceDataset` 构造一个长序列（warmup + rollout）。
   - 全局归一化（关键细节）：
     - 在整个序列上拼接所有 `x`，对 `rho`、`Te` 先做 `log1p(ReLU(.))`，然后计算整体 `global_mean` 和 `global_std`，完全遵守“按通道全局 Z‑score”的规范。
     - 推理时，若 checkpoint 中保存了训练期的 mean/std，则优先使用 checkpoint 的统计量，保证推理与训练一致。
   - LNN hidden state “预热”（遵守“不 amnesia”原则）：
     - WARMUP 阶段用真实的归一化帧顺序送入 encoder+processor，让 CfC/GRU 的 `hidden_state` 收集历史信息。
     - 在最后一个 warmup 步，用 decoder 输出的 Δx 预测得到 `x_{t+1}` 作为自回归初值。
   - 自回归 rollout：
     - 在归一化空间中演化：`x_{t+1} = x_t + Δx_t`。
     - 每步把当前预测 `current_pred_norm` 作为下一步输入（teacher forcing 关闭）。
     - 同时把每一步的预测反归一化回物理空间（undo Z‑score + undo log1p for rho/Te）与 ground truth 对比。
   - 误差与物理剖面分析：
     - 计算每个时间步上 5 个变量的“相对 MAE / 全局 std”，绘制 Autoregressive 误差累计图。
     - 取最后一帧，选 Z≈0 的中平面节点，按 R 排序，绘制 ρ 和 T 的 1D 剖面，对比台基梯度（预测 vs 真值）。

3. **可视化工具**  
   visualization.py 的 `plot_jorek_field`：

   - 严格遵守“不栅格化”：直接用 (R, Z) 和原生三角拓扑 `triangles` 构造 `matplotlib.tri.Triangulation`。
   - 根据变量类型选择 colormap（例如 Φ / v_par 用对称的 `RdBu_r`）。
   - 保持 R/Z 轴比例一致，适配等离子体截面的几何展示。

---

**六、测试与分段管线验证**

代码在 tests 目录下对整条 GNN‑LNN‑GNN 管线做了逐段和整体的 smoke test：

- test_perception_pipeline.py：  
  - 单独测试 GNN Encoder：HDF5 → Data → Encoder → latent，检查形状 `[N, 32]` 等。
- test_cognition_pipeline.py：  
  - 构造 batch（两个图）测试 Encoder+LNN 的联通性以及 batch 索引对 Graph→Sequence 的转换逻辑。
- test_reconstruction_pipeline.py：  
  - 测完整 `Encoder → Processor → Decoder → ReducedMHDConstraints`，验证：
    - 5 维状态向量输出正常。
    - 图级破裂概率输出正常。
    - 利用 `F0` 通过物理约束层成功重建 B、v 场。
- test_training_and_vis.py：  
  - 用 t 和 t+1 两帧模拟训练，跑若干 epoch，验证 `JorekTrainer + JorekLoss` 能正常收敛一小段。
  - 同时测试可视化函数能正确画出 ψ 场。
- test_data_visualization.py：  
  - 对 5 个变量逐个渲染快照，检查三角拓扑、色标是否合理。
- test_grad_ops.py：  
  - 从真实 JOREK 帧中读取 ψ，测试 `compute_gradient` 和随后的物理场计算，验证梯度算子在真实网格上的表现。
- test_dataset_loader.py：  
  - 检查序列加载和 `collate_temporal_graphs` 打包逻辑是否正确保留时间维度。

---

**七、目前阶段的能力与尚未完全完成的部分**

- 已经基本实现的：
  - 完整的 JOREK HDF5 → PyG Graph 解析（含拓扑、边界、物理常数、时间戳）。
  - 满足“滑动窗口 + 时间顺序 + 不打乱”的序列数据集与自定义 collate。
  - GNN‑LNN‑GNN 三段式模型结构，以及 Reduced MHD 物理约束 / 微分算子。
  - 针对破裂预测的混合损失（Huber 回归 + BCE 分类 + 边界项）。
  - 针对物理插值/残差任务的 PDE 物理损失（带无量纲化和 log‑space 反变换）。
  - 全局归一化（含 rho/Te 的 log1p 处理）在诊断脚本和物理损失中已贯通。
  - LNN hidden_state 的“warm‑up”式推理流程已经在 verify_reliability 中落实。
  - 成体系的测试和可视化脚本保证每一子模块可以单独验证。

- 仍然需要/可能在后续完善的：
  - 主训练脚本中对**完整时间序列**的多步训练（比如 teacher forcing over seq_len）目前主要通过测试与诊断示例展示逻辑，正式的序列训练 pipeline（如 train_sequence_pipeline.py）还需要继续打磨和统一到一个标准接口。
  - `JorekTrainer` 目前主要驱动单步损失（t → t+1），物理残差型损失 `JorekPhysicsLoss` 主要在单独脚本中调用，后续可以统一到一个更通用的 Trainer/Task 抽象中。
  - 部分训练相关文件（例如 src/training/loss.py）在当前版本不存在或未启用，说明历史上有过重构，现阶段以 loss.py 为主。

