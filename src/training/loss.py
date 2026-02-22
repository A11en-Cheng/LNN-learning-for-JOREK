import torch
import torch.nn as nn
import torch.nn.functional as F

# Import physics operators and utilities for Physics-Informed Loss
from src.physics.grad_ops import compute_gradient, compute_divergence
from src.physics.mhd import get_magnetic_field, get_velocity_field
from src.physics.cocos8 import reconstruct_velocity

class JorekLoss(nn.Module):
    """
    Physics-Informed Loss function for HyPhy-GLNN.
    Combines data-driven loss (Huber/Log-Cosh) with physics constraints (Mass Continuity, Parallel Momentum).
    """
    def __init__(self, lambda_data=1.0, lambda_phys=0.01, stats=None, dt=1.0):
        super().__init__()
        self.lambda_data = lambda_data
        self.lambda_phys = lambda_phys
        self.dt = dt
        # stats should be a dict with 'mean' and 'std' tensors of shape [5]
        self.stats = stats
        
    def denormalize(self, x):
        """
        Denormalize the features from Z-score back to JOREK dimensionless units.
        """
        if self.stats is None:
            return x
        mean = self.stats['mean'].to(x.device)
        std = self.stats['std'].to(x.device)
        return x * std + mean

    def forward(self, pred, target, graph_data, curr_state=None):
        """
        Compute the total loss.
        
        Args:
            pred: Predicted next node features [num_nodes, 5]
            target: Target next node features [num_nodes, 5]
            graph_data: PyG Data object containing edge_index, edge_attr, pos, etc.
            curr_state: Current node features [num_nodes, 5]. If None, uses graph_data.x
            
        Returns:
            total_loss: Scalar loss value
        """
        # 1. Data-driven loss (Huber Loss for stiffness)
        # Using smooth_l1_loss which is equivalent to Huber loss
        data_loss = F.smooth_l1_loss(pred, target, reduction='mean')
        
        if self.lambda_phys <= 0.0:
            return data_loss
            
        # 2. Physics-informed loss
        if curr_state is None:
            curr_state = graph_data.x
            
        # Denormalize current and next step predictions
        curr_denorm = self.denormalize(curr_state)
        next_denorm = self.denormalize(pred)
        
        # Extract variables (0: psi, 1: Phi, 2: rho, 3: T, 4: v_par)
        rho_curr = curr_denorm[:, 2:3]
        v_par_curr = curr_denorm[:, 4:5]
        
        psi_next = next_denorm[:, 0:1]
        Phi_next = next_denorm[:, 1:2]
        rho_next = next_denorm[:, 2:3]
        T_next = next_denorm[:, 3:4]
        v_par_next = next_denorm[:, 4:5]
        
        R = graph_data.pos[:, 0:1]
        
        # Get normalization factors if available, else default to 1.0
        F0 = 1.0
        if hasattr(graph_data, 'norm_factors') and graph_data.norm_factors is not None:
            if isinstance(graph_data.norm_factors, dict) and 'F0' in graph_data.norm_factors:
                F0 = graph_data.norm_factors['F0']
            elif hasattr(graph_data.norm_factors, 'F0'):
                F0 = graph_data.norm_factors.F0
                
        # Compute Gradients on denormalized quantities
        # grad_rho: [num_nodes, 2]
        grad_rho = compute_gradient(rho_next, graph_data.edge_index, graph_data.edge_attr).squeeze(1)
        grad_T = compute_gradient(T_next, graph_data.edge_index, graph_data.edge_attr).squeeze(1)
        grad_v_par = compute_gradient(v_par_next, graph_data.edge_index, graph_data.edge_attr).squeeze(1)
        grad_Phi = compute_gradient(Phi_next, graph_data.edge_index, graph_data.edge_attr).squeeze(1)
        grad_psi = compute_gradient(psi_next, graph_data.edge_index, graph_data.edge_attr).squeeze(1)
        
        # Reconstruct B field and velocity field
        B_field = get_magnetic_field(grad_psi, R, F0)  # [num_nodes, 3]
        v_field = reconstruct_velocity(grad_Phi, v_par_next, B_field, R)  # [num_nodes, 3]
        
        v_R = v_field[:, 0:1]
        v_Z = v_field[:, 1:2]
        
        # --- Mass Continuity Residual ---
        # div(rho * v) in cylindrical coordinates (assuming axisymmetry d/dphi = 0)
        # div(F) = d(F_R)/dR + d(F_Z)/dZ + F_R / R
        rho_v_R = rho_next * v_R
        rho_v_Z = rho_next * v_Z
        rho_v_pol = torch.cat([rho_v_R, rho_v_Z], dim=-1)
        
        div_cart_rho_v = compute_divergence(rho_v_pol, graph_data.edge_index, graph_data.edge_attr).unsqueeze(-1)
        div_rho_v = div_cart_rho_v + rho_v_R / (R + 1e-8)
        
        res_mass = (rho_next - rho_curr) / self.dt + div_rho_v
        
        # --- Parallel Momentum Residual ---
        # v . grad(v_par) = v_R * d(v_par)/dR + v_Z * d(v_par)/dZ
        v_dot_grad_v_par = v_R * grad_v_par[:, 0:1] + v_Z * grad_v_par[:, 1:2]
        
        # grad_parallel(rho * T_e) = (B . grad(rho * T_e)) / |B|
        rho_T = rho_next * T_next
        grad_rho_T = compute_gradient(rho_T, graph_data.edge_index, graph_data.edge_attr).squeeze(1)
        
        B_R = B_field[:, 0:1]
        B_Z = B_field[:, 1:2]
        B_mag = torch.norm(B_field, p=2, dim=-1, keepdim=True) + 1e-8
        
        B_dot_grad_rho_T = B_R * grad_rho_T[:, 0:1] + B_Z * grad_rho_T[:, 1:2]
        grad_par_rho_T = B_dot_grad_rho_T / B_mag
        
        res_momentum = rho_next * ((v_par_next - v_par_curr) / self.dt + v_dot_grad_v_par) + grad_par_rho_T
        
        # Physics Loss
        phys_loss = torch.mean(res_mass**2) + torch.mean(res_momentum**2)
        
        total_loss = self.lambda_data * data_loss + self.lambda_phys * phys_loss
        
        return total_loss
