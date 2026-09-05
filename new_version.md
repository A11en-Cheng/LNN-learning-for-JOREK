# HyPhy-GLNN MoE: Project Roadmap

本路线图旨在规划 HyPhy-GLNN 项目从“单体 5 维基准”向“9 维条件注入专家网络 (MoE)”，再到“3D 傅里叶神经算子”的演进路径。核心目标是构建一个能够处理复杂热猝灭 (Thermal Quench)、适配外部物理源项 (SPI/MGI)，并最终作为 JOREK 代理求解器核心的聚变等离子体基础模型。

## Phase 1: 9-Field Foundation & Baseline (Month 1)
**目标**：打通 9 维数据流，确立单体网络 (Monolithic) 基准，排除底层数据与内存隐患。

*   **[ ] 任务 1.1：数据管线升级**
    *   重构 HDF5 解析器，精准提取 9 大物理场（$\psi, \Phi, Z_j, \omega, \rho, T_i, v_{\parallel}, \rho_{imp}, T_e$）。
    *   严格落实热力学变量的 `log1p` 正定保护映射与电磁变量的 Z-score 缩放。
*   **[ ] 任务 1.2：Metadata 与条件提取**
    *   解析并提取 JOREK Namelist 中的耗散与弛豫参数（$\eta, visco, ZK_{\parallel}, ZK_{\perp}$）。
    *   构建 20 步长 (SEQ_LEN=20) 的马尔可夫短序列数据切片加载器。
*   **[ ] 任务 1.3：基准对照组确立 (Monolithic V1.5)**
    *   升级旧版本单体网络至 9 维输入。
    *   **关键防御**：全面实装 `torch.utils.checkpoint` (梯度检查点) 以防止 OOM。
    *   完成单样本过拟合测试 (Single-Batch Overfit)，验证网络容量底线。

## Phase 2: Mixture of Physical Experts Architecture (Month 2 - 3)
**目标**：完成核心算子分裂，解决多物理场梯度干涉，提升复杂非线性演化的稳定性。

*   **[ ] 任务 2.1：专家感知层开发**
    *   构建三大独立 GNN Encoder：电磁专家 (EM)、热力学专家 (Thermo)、动量专家 (Momentum)。
    *   开发 Namelist 条件注入模块 (Conditioning MLP)，并将全局参数融入各专家特征空间。
*   **[ ] 任务 2.2：潜空间握手与中央引擎**
    *   实现 Cross-Attention/MLP 潜空间握手层 (Latent Handshake)，确保物理因果律在分裂架构中的瞬时交互。
    *   搭建大容量 Latent Bus，接入 LNN 进行连续时间 ODE 推演。
*   **[ ] 任务 2.3：梯度解耦与验证**
    *   解码器应用 `.detach()` 建立物理防火墙，实现 PDE Loss 的定向反向传播。
    *   在无源 2D ($n=0$) 演化数据上，进行 MoE 与 Monolithic 基准的全面消融实验 (Ablation Study) 与多维 Benchmark 评估。

## Phase 3: Explicit Forcing & Source Term Integration (Month 4 - 5)
**目标**：突破无源限制，引入 SPI/MGI 极高刚性源项，适配 EXL-50U / EHL-2 等装置的破裂缓解真实工况。

*   **[ ] 任务 3.1：源项数据通道解析**
    *   从 H5 文件中提取杂质沉积率、电离与辐射冷却率等显式源项 ($S_{\rho}, S_E$)。
*   **[ ] 任务 3.2：控制驱动的神经 ODE (Forced ODE)**
    *   将源项作为独立的外部控制信号 (Control Inputs) 注入 LNN 中央推演总线。
    *   调整 Huber/SmoothL1 Loss 权重，应对热猝灭瞬间极端的时空梯度爆炸。
*   **[ ] 任务 3.3：自洽闭环验证**
    *   验证模型在剧烈外力扰动下，热力学崩溃向电磁拓扑重联传导的物理自洽性。

## Phase 4: The 3D Spectral-Graph Frontier (Month 6+)
**目标**：从 2D 截面迈向全景 3D，掌握非线性模式耦合，逼近 JOREK 异构求解器核心形态。

*   **[ ] 任务 4.1：图-谱混合架构设计 (Graph-Spectral LNN)**
    *   扩展输入维度以支持截断的傅里叶环向模态 ($n=0, 1, 2, 3...$)。
    *   在特征传递中引入复数张量运算，支持不同模态间的能量级联 (Energy Cascade)。
*   **[ ] 任务 4.2：算子级替代与 JOREK 接口集成**
    *   开发双向数据接口，实现在 JOREK 运行时代替部分极耗时的隐式求解模块。
    *   最终目标：实现基于梯度的逆向工程（神经代理自动弛豫）与长程数字孪生推演。