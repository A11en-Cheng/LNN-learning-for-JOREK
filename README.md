# 提案：基于混合物理图与液体神经网络 (LNN-learning-for-JOREK) 的 JOREK 聚变破裂预测基准

| 维度 | 旧提案 (V2): CNN+LNN (图像插值) | 新提案 (V3): LNN-learning-for-JOREK (GNN原生网格) | 核心改进点 |
| --- | --- | --- | --- |
| **输入数据处理** | **强制栅格化**：将非结构化网格插值到 $64\times64$ 笛卡尔网格，丢失高阶曲率与梯度信息。 | **图原生学习 (Graph-Native)**：直接读取 JOREK 节点 (`t_values`) 与拓扑连接 (`t_contain_node`) 构建图。 | **物理保真度**：完美保留等离子体边界、X 点结构及通量面 (Flux-aligned) 几何特征。 |
| **空间特征提取** | **二维卷积 (CNN)**：依赖平移不变性，无法处理复杂的边界和极点。 | **图神经网络 (GNN)**：通过消息传递机制 (Message Passing) 在相邻节点间传播物理量。 | **几何自适应**：天然适应球形托卡马克 (ST) 等复杂或多连通域拓扑。 |
| **网格与装置迁移** | **分辨率锁死**：模型绑定于固定的输入尺寸，遇到自适应加密 (AMR) 或新装置即失效。 | **分辨率无关 (Resolution Independent)**：学习节点间的交互法则，支持 Zero-shot 网格迁移。 | **泛化性极强**：在低分辨率训练的模型可直接用于高分辨率推理，且支持跨装置迁移。 |
| **动力学核心** | **LNN 预测全局 Latent**：处理插值图像压缩后的隐变量演化。 | **LNN 预测图节点 Latent**：结合 GNN，推演每个节点特征的连续时间 ODE。 | **精准度**：能够追踪局部区域（如磁岛核心）在非均匀时间步长下的指数级突变。 |
| **物理约束解码** | **有限差分**：在规整网格上进行简单的图像差分求磁场 $\mathbf{B}$。 | **离散图微分算子**：在原生图上应用硬约束（预测 $\psi$，图上求旋度）与软约束（能量/质量守恒）。 | **严格物理一致**：满足 $\nabla \cdot \mathbf{B}=0$，并保证长时程动力学不发散。 |

---

## 1. 愿景与核心目的 (Vision & Objective)

* **核心愿景**：构建一个**几何自适应 (Geometry-Adaptive)**、**物理守恒 (Physics-Conserving)** 且 **时间连续 (Continuous-Time)** 的托卡马克等离子体演化预测模型，彻底打破深度学习在处理非结构化科学数据时的插值损耗。
* **直接目的**：
  1. 建立基于 JOREK 原生有限元数据的标准化 Benchmark，验证 GNN+LNN 架构处理复杂磁流体动力学 (MHD) 的有效性。
  2. 实现跨网格分辨率的迁移学习，证明模型学习到了物理法则而非特定网格的数值拟合。
* **长远目标**：实现一套“即插即用”的破裂预测与数字孪生代理模型，能够在不同托卡马克装置（如 EXL-50U, HL-2M, ITER）间实现高精度知识迁移。

## 2. 核心技术架构 (Technical Architecture)

本提案采用 **GNN-LNN-GNN (感知-认知-重构)** 三段式混合物理架构。

### 模块 A：感知层 - 图拓扑特征提取 (Perception: Graph Encoder)
* **技术路线**：废弃 CNN，使用 **Graph Neural Networks (GCN/GAT)**。
* **输入构建**：
  * **节点特征 ($V$)**：直接读取 HDF5 中的物理坐标 $(R, Z)$ 与状态变量 $(\rho, T, \psi, \dots)$。
  * **边特征 ($E$)**：解析 JOREK 单元连接矩阵 (`contain_node` 或 `neighbours`)，构建物理连通图。区分极向 (Poloidal) 与径向 (Radial) 边缘以体现等离子体输运的各向异性。
* **作用**：提取 X 点、磁岛结构等局部几何特征，将节点维度压缩为 Latent Representation。

### 模块 B：认知层 - 连续时间动力学 (Cognition: Continuous Dynamics)
* **技术路线**：采用 **Liquid Neural Networks (CfC - Closed-form Continuous models)**。
* **输入与输出**：接收 GNN 提取的图结构 Latent 向量，以及真实的物理时间步长 $\Delta t$。
* **优势**：完美契合 JOREK 的自适应时间步长（$dt$ 在破裂前夕会急剧变小），在连续时间域求解微分方程，无需重采样序列。

### 模块 C：重构层 - 物理硬/软约束解码 (Reconstruction: Physics Decoder)
* **硬约束 (Hard Constraints)**：
  * **磁场无散**：模型输出节点上的磁通 $\psi$，通过图结构上的离散微分算子计算 $\mathbf{B} = \nabla \times (\psi \hat{\phi})$，强制 $\nabla \cdot \mathbf{B} \equiv 0$。
  * **正定性**：在输出密度 $\rho$ 和温度 $T$ 的最后一层应用 `Softplus` 激活函数，根除负值导致的非物理崩溃。
* **软约束 (Soft Constraints/Loss)**：在损失函数中引入整体能量守恒惩罚 ($Loss_{energy}$) 与质量连续性惩罚 ($Loss_{mass}$)。

## 3. 实施途径与技术路线 (Implementation Pathway)

### 阶段一：短期目标 —— 数据图管线与可行性验证 (Month 1-2)
1. **数据解析器**：编写 Python 脚本 (`jorek_parser.py`)，利用 `h5py` 直接读取 JOREK 重启文件中的 `t_values`、`t_x` 和 `t_contain_node`。
2. **图构建与加载**：基于 PyTorch Geometric (PyG) 将非结构化数据封装为 `Data` 对象，建立包含时序的 `GraphDataset`。
3. **基准跑通**：构建初步的 `GCN -> CfC-LNN -> GCN` 骨架，验证数据流张量维度匹配，完成初步收敛。

### 阶段二：中期目标 —— 动力学引入与物理约束 (Month 3-5)
1. **定制图微积分**：开发在非结构化网格上的梯度计算层 (`grad_ops.py`)，实现磁场 $\mathbf{B}$ 和速度场 $\mathbf{v}$ (经流函数 $\Phi$) 的推导。
2. **物理正则化训练**：引入自定义物理 Loss 函数（状态重构误差 + 全局能量守恒误差），观察并解决多目标优化可能带来的梯度冲突（Gradient Pathology）。
3. **演化验证**：在热猝灭 (Thermal Quench) 序列上测试，对比 LNN 预测的动力学轨迹与 JOREK 真实轨迹，验证时间连续性优势。

### 阶段三：长期目标 —— 装置迁移与长程控制 (Month 6+)
1. **跨网格泛化测试 (Zero-shot)**：在 JOREK 粗网格 (e.g., 5k nodes) 上完成训练，直接输入细网格 (e.g., 20k nodes) 数据进行推理验证。
2. **多装置测试**：评估模型从特定装置（如球形托卡马克 EXL-50U）向其他磁位形迁移的适应性。
3. **闭环代理模型**：将模型封装为极速 (毫秒级) 数字孪生接口，评估其辅助实时放电控制的潜力。

---

## 4. 附录：核心伪代码架构 (Implementation Details)

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from ncps.torch import CfC

class HyPhyGLNN(nn.Module):
    def __init__(self, in_node_features, latent_dim):
        super().__init__()
        # 1. 感知层 (The Eye): GNN Encoder
        # 输入: 原生网格节点特征 (R, Z, rho, T, psi)
        # 作用: 沿着 edge_index 传播物理信息
        self.gnn_encoder = nn.ModuleList([
            GCNConv(in_node_features, 64),
            GCNConv(64, latent_dim)
        ])
        
        # 2. 动力学层 (The Brain): LNN ODE Solver
        # 作用: 预测每个节点(或全图)特征在任意 dt 下的演化
        self.dynamics = CfC(input_size=latent_dim, hidden_size=128, proj_size=latent_dim)
        
        # 3. 解码层 (The Body): GNN Decoder
        self.gnn_decoder = nn.ModuleList([
            GCNConv(latent_dim, 64),
            GCNConv(64, 3)  # 输出: d_psi, d_rho, d_T
        ])
        
        self.softplus = nn.Softplus() # 物理正定性约束

    def physics_hard_constraint(self, psi_nodes, edge_index, coords):
        """
        在图上计算离散微分: B = curl(psi)
        """
        # 利用 PyG 的聚合函数和节点坐标计算差分，返回满足 div(B)=0 的磁场
        B_field = compute_curl_on_graph(psi_nodes, edge_index, coords)
        return B_field

    def forward(self, graph_data_t, time_spans):
        x, edge_index, coords = graph_data_t.x, graph_data_t.edge_index, graph_data_t.coords
        
        # Step A: GNN 空间特征提取
        for conv in self.gnn_encoder:
            x = torch.relu(conv(x, edge_index))
            
        # Step B: LNN 动力学演化 (输入包含非均匀时间 dt)
        # x_evolved 预测下一时刻的节点特征
        x_evolved, _ = self.dynamics(x.unsqueeze(0), times=time_spans)
        x_evolved = x_evolved.squeeze(0)
        
        # Step C: GNN 还原至物理量
        out = x_evolved
        for conv in self.gnn_decoder[:-1]:
            out = torch.relu(conv(out, edge_index))
        out = self.gnn_decoder[-1](out, edge_index)
        
        # Step D: 分解变量并施加物理限制
        psi_pred = out[:, 0]
        # 强制热力学变量为正
        rho_pred = self.softplus(out[:, 1]) 
        T_pred = self.softplus(out[:, 2])
        
        # Step E: 由势函数严格计算磁场
        B_pred = self.physics_hard_constraint(psi_pred, edge_index, coords)
        
        return B_pred, psi_pred, rho_pred, T_pred

```

