import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class GraphSpatialGradient(MessagePassing):
    """
    在非结构化网格上计算空间梯度的可微算子。
    使用基于距离加权的最小二乘近似 (对角化简化版) 来计算 \nabla f = (\partial f / \partial R, \partial f / \partial Z)。
    """
    def __init__(self):
        super().__init__(aggr='add')

    def forward(self, x, pos, edge_index):
        """
        Args:
            x: 标量场 [Num_nodes, 1]
            pos: 节点坐标 (R, Z) [Num_nodes, 2]
            edge_index: 图连接矩阵 [2, Num_edges]
        Returns:
            grad: 梯度向量 [Num_nodes, 2]
        """
        row, col = edge_index
        
        # 计算边上的差值
        dp = pos[col] - pos[row] # [Num_edges, 2] (dR, dZ)
        dx = x[col] - x[row]     # [Num_edges, 1] (df)
        
        # 权重：距离平方的倒数 (防止除零)
        dist_sq = (dp ** 2).sum(dim=-1, keepdim=True) + 1e-6 # [Num_edges, 1]
        w = 1.0 / dist_sq
        
        # 消息：w * df * dp
        msg = w * dx * dp # [Num_edges, 2]
        
        # 聚合消息 (分子)
        grad_sum = self.propagate(edge_index, msg=msg) # [Num_nodes, 2]
        
        # 归一化因子 (分母)：w * dp^2
        norm_msg = w * (dp ** 2) # [Num_edges, 2]
        norm_sum = self.propagate(edge_index, msg=norm_msg) + 1e-6 # [Num_nodes, 2]
        
        # 计算最终梯度
        grad = grad_sum / norm_sum # [Num_nodes, 2]
        return grad

    def message(self, msg):
        return msg

class ReducedMHDConstraints(nn.Module):
    """
    物理约束层 (Hard Constraints):
    严格遵守 JOREK Reduced MHD Ansatz。
    接收网络预测的 5 个核心变量，通过可微的图空间梯度计算真实的磁场 B 和速度场 v。
    """
    def __init__(self):
        super().__init__()
        self.grad_op = GraphSpatialGradient()
        
    def forward(self, state_vars, pos, edge_index, F0):
        """
        Args:
            state_vars: 预测的状态变量 [Num_nodes, 5] (psi, Phi, rho, T, v_parallel)
            pos: 节点坐标 (R, Z) [Num_nodes, 2]
            edge_index: 图连接矩阵 [2, Num_edges]
            F0: 真空环向磁场常数 (R0 * B0)
        Returns:
            B_vec: 磁场矢量 (B_R, B_Z, B_phi) [Num_nodes, 3]
            v_vec: 速度矢量 (v_R, v_Z, v_phi) [Num_nodes, 3]
        """
        # 1. 提取核心变量
        psi = state_vars[:, 0:1]
        Phi = state_vars[:, 1:2]
        v_par = state_vars[:, 4:5]
        
        R = pos[:, 0:1] # 大半径 R
        
        # 2. 计算空间梯度
        grad_psi = self.grad_op(psi, pos, edge_index) # [Num_nodes, 2] -> (dpsi/dR, dpsi/dZ)
        grad_Phi = self.grad_op(Phi, pos, edge_index) # [Num_nodes, 2] -> (dPhi/dR, dPhi/dZ)
        
        dpsi_dR = grad_psi[:, 0:1]
        dpsi_dZ = grad_psi[:, 1:2]
        
        dPhi_dR = grad_Phi[:, 0:1]
        dPhi_dZ = grad_Phi[:, 1:2]
        
        # 3. 计算磁场 B (Reduced MHD Ansatz)
        # B = (F0/R) * e_phi + (1/R) * \nabla psi x e_phi
        # 在 (R, Z, phi) 柱坐标系下: \nabla psi x e_phi = (dpsi/dZ) * e_R - (dpsi/dR) * e_Z
        B_R = (1.0 / R) * dpsi_dZ
        B_Z = -(1.0 / R) * dpsi_dR
        B_phi = F0 / R
        B_vec = torch.cat([B_R, B_Z, B_phi], dim=1) # [Num_nodes, 3]
        
        # 4. 计算速度场 v (Reduced MHD Ansatz)
        # v = -R * \nabla Phi x e_phi + v_parallel * B
        # -R * \nabla Phi x e_phi = -R * (dPhi/dZ * e_R - dPhi/dR * e_Z)
        v_R = -R * dPhi_dZ + v_par * B_R
        v_Z = R * dPhi_dR + v_par * B_Z
        v_phi = v_par * B_phi
        v_vec = torch.cat([v_R, v_Z, v_phi], dim=1) # [Num_nodes, 3]
        
        return B_vec, v_vec
