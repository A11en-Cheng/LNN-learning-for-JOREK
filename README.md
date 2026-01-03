

| 维度 | 旧提案 (V1): LNN Baseline | 新提案 (V2): Physics-Hard-Constrained HyPhy-LNN | 核心改进点 |
| --- | --- | --- | --- |
| **输入数据处理** | **纯数值压缩**：使用 PCA/Autoencoder 将高维场压缩为抽象的 Latent Vector。 | **对象中心 (Object-Centric)**：引入特征提取器 (类 YOLO/U-Net)，显式提取**物理对象**（磁岛宽度、X点位置）+ 隐式残差。 | **可解释性与泛化性**：网络不再是“看像素”，而是“看物理结构”，对新工况更鲁棒。 |
| **物理约束** | **软约束 (Soft)**：依靠 Loss 函数 (MSE) 逼近，或无约束。可能出现 。 | **硬约束 (Hard)**：网络预测磁矢势  或磁通 ，通过数学定义的微分层计算 。 | **物理一致性**：在数学层面强制满足 ，消除非物理的“磁单极子”幻觉。 |
| **动力学核心** | **LNN 单核**：LNN 负责学习所有 Latent 变量的演化。 | **混合动力学**：LNN 学习关键物理对象的演化（主导模态）+ 细节演化。 | **效率**：将算力集中在决定破裂的关键物理量上。 |
| **输出与解码** | **直接重构**：解码器直接输出磁场 。 | **势函数重构**：输出 ，经由不可学习的微分层转化为物理场。 | **拓扑保证**：确保磁拓扑结构的连续性，便于后续热流追踪。 |
| **OOD 应对** | 依赖数据覆盖率，遇到未见过的 SPI 策略可能失效。 | 依赖**特征解耦**：即使注入策略改变，只要物理特征（如磁岛）被识别，动力学规律依然适用。 | **鲁棒性**：将“感知”与“推理”解耦。 |

---

# 提案：基于混合物理特征与液体神经网络 (HyPhy-LNN) 的 JOREK 聚变破裂预测基准

## 1. 愿景与核心目的 (Vision & Objective)

* **核心愿景**：构建一个**物理一致 (Physics-Consistent)** 且**结构化 (Structured)** 的等离子体演化预测模型，解决传统深度学习模型不满足守恒律、难以解释以及在分布外 (OOD) 数据上泛化能力差的问题。
* **直接目的**：
1. 建立一个标准化的 Benchmark，验证“对象中心学习”结合“连续时间神经网络 (LNN)”在 MHD 模拟数据上的有效性。
2. 通过**硬约束架构**，从数学层面根除磁场散度不为零 () 的非物理伪影。


* **长远目标**：实现对托卡马克破裂前夕磁拓扑结构的毫秒级精准推演，进而实现基于物理场的实时热流沉积预测与控制。

## 2. 核心技术架构 (Technical Architecture)

本提案采用 **“感知-认知-重构” (Perception-Cognition-Reconstruction)** 的三段式架构，将物理先验知识深度嵌入神经网络设计中。

### 模块 A：感知层 - 物理对象特征提取 (Perception: Object-Centric Extraction)

* **痛点解决**：传统 CNN/Autoencoder 提取的潜变量缺乏明确物理意义，导致模型难以应对未见过的扰动（OOD 问题）。
* **具体实现**：引入 **Physics-Informed U-Net** 或 **Attention Mechanism**。
* **显式特征 (Explicit Physics)**：利用语义分割思想，识别并提取关键相干结构参数：
* **磁岛 (Magnetic Islands)**：提取宽度 、相位 、位置 。
* **随机化区域 (Ergodic Layer)**：提取随机化边界半径。


* **隐式特征 (Implicit Residuals)**：利用卷积层编码背景流场的细微变化，作为显式特征的补充。



### 模块 B：认知层 - 连续时间动力学 (Cognition: Continuous Dynamics)

* **痛点解决**：JOREK 的自适应时间步长 ( 变化范围可达  倍) 导致传统 RNN/LSTM 需重采样，丢失高频信息。
* **具体实现**：采用 **Liquid Neural Networks (Specifically CfC - Closed-form Continuous models)**。
* LNN 将感知层输出的特征向量  视为输入，学习其对应的常微分方程 (ODE)：


* **优势**：在连续时间域上求解，能够自适应地处理破裂前夕的指数级增长 (Thermal Quench)，并准确捕捉因果关系。



### 模块 C：重构层 - 物理硬约束解码 (Reconstruction: Hard Constraints)

* **痛点解决**：直接预测磁场  往往导致 ，破坏磁力线拓扑，使得后续的热流分析失效。
* **具体实现**：**势函数预测 + 自动微分**。
* 模型**不直接输出**磁场或速度场。
* 模型输出 **磁通函数 (Poloidal Flux, )** 和 **流函数 (Stream Function, )**。
* 通过一个**不可学习的微分层 (Fixed Differentiable Layer)** 计算物理量：



* **结果**：输出场在机器精度下天然满足  和  (在特定近似下)。



## 3. 实施途径与技术路线 (Implementation Pathway)

### 阶段一：数据管线构建 (Data Pipeline)

1. **JOREK 数据源**：生成涵盖平稳放电、密度极限破裂、不同 SPI 注入量/位置的模拟数据集。
2. **降维预处理**：
* 将 JOREK 的有限元数据插值到高分辨率极坐标网格 。
* 标注/提取关键物理特征（如利用庞加莱图辅助标记磁岛位置），用于训练感知层的特征提取器。



### 阶段二：模型构建与训练 (Modeling)

1. **编码器训练**：
* 使用 **U-Net** 结构，训练目标是既能重构输入场，又能准确分割出磁岛区域（Multi-task Learning）。


2. **动力学训练**：
* 冻结编码器，将提取的时间序列特征输入 **CfC-LNN**。
* Loss 函数设计：包含状态预测误差 (MSE) + 物理守恒误差（如能量守恒惩罚）。


3. **端到端微调 (End-to-End Fine-tuning)**：
* 连接所有模块，加入微分层，进行整体梯度下降。



### 阶段三：基准测试与评估 (Evaluation Benchmark)

为了验证可靠性，我们设立三个递进的测试任务：

* **Task 1: 拓扑一致性测试 (Topological Consistency)**
* **指标**： 的残差值（应接近 0）；X-point 和 O-point 位置的预测精度。
* **目的**：验证硬约束架构的有效性。


* **Task 2: 动力学分岔预测 (Bifurcation Prediction)**
* **指标**：Time-to-TQ (热猝灭发生时间) 的误差；磁岛生长率  的拟合度。
* **目的**：验证 LNN 对混沌边界指数增长的捕捉能力。


* **Task 3: 泛化能力测试 (OOD Generalization)**
* **场景**：输入训练集中未包含的 SPI 注入位置或混合注入策略。
* **指标**：模型是否能基于物理特征推演出合理的演化路径，而非产生非物理的伪影。



## 4. 预期成果与延伸价值 (Impact)

* **科学价值**：首个将 **对象中心学习 (Object-Centric)** 与 **LNN** 结合应用于 MHD 破裂预测的研究，为 AI for Science 提供一种“白盒化”的新范式。
* **工程价值**：
* 提供满足  的高质量替代模型，可直接用于后续的**热流沉积 (Heat Flux)** 计算（Field Line Tracing）。
* 计算速度相比 JOREK 提升 3-4 个数量级，使实时放电模拟成为可能。



---

## 5. 附录：具体实现的技术细节 (Implementation Details)

为了回应你关于“实现途径”的询问，以下是核心代码逻辑的伪代码描述：

```python
import torch
import torch.nn as nn
from ncps.torch import CfC  # Neural Circuit Policies for LNN

class HyPhyLNN(nn.Module):
    def __init__(self, input_shape, n_latent):
        super().__init__()
        # 1. 感知层 (The Eye): Physics-Aware Encoder
        # 输入: JOREK Snapshot (Density, Temperature, Flux)
        # 任务: 提取显式特征(磁岛宽度等) + 隐式Latent
        self.perception = PhysicsInformedEncoder(input_shape, n_latent)
        
        # 2. 动力学层 (The Brain): LNN ODE Solver
        # 任务: 预测下一时刻的特征状态
        self.dynamics = CfC(input_size=n_latent, hidden_size=128, proj_size=n_latent)
        
        # 3. 解码层 (The Body): Potential Function Decoder
        # 任务: 将Latent解码为 Poloidal Flux (psi)
        self.decoder = SpatialDecoder(n_latent, output_shape=(H, W))

    def physics_hard_constraint(self, psi):
        """
        物理硬约束层 (不可学习)
        通过离散微分算子，由 psi 计算 B
        """
        # 使用 PyTorch 的 gradient 或 卷积核计算梯度
        dpsi_dR = torch.gradient(psi, dim=1) 
        dpsi_dZ = torch.gradient(psi, dim=2)
        
        # B_R = -(1/R) * dpsi_dZ
        # B_Z = +(1/R) * dpsi_dR
        # 这样计算出的 B 场天然满足 div(B) = 0
        B_field = compute_B_from_psi(dpsi_dR, dpsi_dZ)
        return B_field

    def forward(self, x_series, time_spans):
        # x_series: [Batch, Time, Channels, Height, Width]
        
        # Step A: 逐帧感知特征
        batch, time, c, h, w = x_series.shape
        x_flat = x_series.view(batch * time, c, h, w)
        features = self.perception(x_flat) # -> [Batch, Time, Latent]
        
        # Step B: LNN 动力学演化
        # prediction_latent 是预测的未来特征
        prediction_latent, _ = self.dynamics(features, times=time_spans)
        
        # Step C: 解码为势函数
        psi_pred = self.decoder(prediction_latent)
        
        # Step D: 物理约束计算物理场
        B_pred = self.physics_hard_constraint(psi_pred)
        
        return B_pred, psi_pred

```


# 项目架构
```text
HyPhy-LNN/
├── data/                   # 数据存放 (不要上传大文件到 git)
│   ├── raw/                # 原始 JOREK 输出文件 (.h5, .dat)
│   └── processed/          # 预处理后的 PyTorch Tensor (.pt)
├── src/                    # 核心源代码
│   ├── __init__.py
│   ├── models/             # 模型定义
│   │   ├── __init__.py
│   │   ├── encoder.py      # 模块A: 感知层 (U-Net / CNN)
│   │   ├── lnn_core.py     # 模块B: 动力学层 (CfC/LNN 封装)
│   │   └── decoder.py      # 模块C: 物理硬约束与解码 (Psi -> B)
│   ├── physics/            # 物理规则与 JOREK 接口
│   │   ├── __init__.py
│   │   ├── jorek_loader.py # 解析 JOREK 数据
│   │   └── mhd_utils.py    # 磁面坐标变换、物理常数、微分算子
│   └── utils/              # 工具函数
│       ├── __init__.py
│       ├── visualization.py# 画磁拓扑图、庞加莱图
│       └── metrics.py      # 自定义评估指标 (e.g., div(B) check)
├── experiments/            # 实验脚本
│   ├── train_topology.py   # Level 1 训练脚本
│   └── train_heatflux.py   # Level 2 训练脚本
├── notebooks/              # Jupyter Notebooks (用于探索性分析)
│   └── 01_data_preview.ipynb
├── tests/                  # 单元测试 (保证各模块独立工作正常)
├── requirements.txt        # 依赖清单
├── verify_env.py           # 环境验证脚本
├── minimal_lnn_test.py     # LNN 最小算例
└── README.md               # 项目说明
```


Note that pytorch should be installed via official instructions from https://pytorch.org/get-started/locally/ to ensure CUDA compatibility.

After pytorch is installed, run `pip install -r requirements.txt` to install other dependencies, then Run `python verify_env.py` to check your environment setup.

