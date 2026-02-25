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
    Combines data-driven loss (Huber/Log-Cosh) with physics constraints (Mass Continuity, Parallel Momentum)
    and exact boundary anchoring from JOREK metadata.
    """
    def __init__(self, lambda_data=1.0, lambda_phys=0.01, lambda_bc=10.0, stats=None, dt=1.0):
        super().__init__()
        self.lambda_data = lambda_data
        self.lambda_phys = lambda_phys
        self.lambda_bc = lambda_bc
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

    def forward(self, pred, target, graph_data, curr_state=None, boundary=None, freeboundary=None):
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
        device = pred.device

        # Determine boundary metadata
        boundary_tensor = None
        if boundary is None and hasattr(graph_data, 'boundary'):
            boundary = graph_data.boundary
        if boundary is not None:
            if isinstance(boundary, torch.Tensor):
                boundary_tensor = boundary.to(device).view(-1)
            else:
                boundary_tensor = torch.tensor(boundary, dtype=torch.long, device=device).view(-1)
        internal_mask = None
        boundary_mask = None
        if boundary_tensor is not None:
            internal_mask = (boundary_tensor == 0)
            boundary_mask = (boundary_tensor > 0)

        # Determine freeboundary flag
        if freeboundary is None and hasattr(graph_data, 'freeboundary'):
            freeboundary = graph_data.freeboundary
        if isinstance(freeboundary, torch.Tensor):
            freeboundary_flag = bool(int(freeboundary.view(-1)[0].item()))
        elif freeboundary is None:
            freeboundary_flag = True
        else:
            freeboundary_flag = bool(freeboundary)

        # 1. Data-driven loss (Huber Loss for stiffness)
        data_loss = F.smooth_l1_loss(pred, target, reduction='mean')
        
        if self.lambda_phys <= 0.0:
            loss_bc = torch.tensor(0.0, device=device)
            if boundary_mask is not None and boundary_mask.any():
                pred_bnd = pred[boundary_mask]
                target_bnd = target[boundary_mask]
                if pred_bnd.shape[0] > 0:
                    loss_bc_plasma = F.mse_loss(pred_bnd[:, 1:], target_bnd[:, 1:])
                    psi_penalty = 5.0 if freeboundary_flag else 50.0
                    loss_bc_psi = F.mse_loss(pred_bnd[:, 0], target_bnd[:, 0]) * psi_penalty
                    loss_bc = loss_bc_plasma + loss_bc_psi
            return data_loss + self.lambda_bc * loss_bc
            
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
                
        # If F0 is a tensor (e.g., from a batch), we need to expand it to match the number of nodes
        if isinstance(F0, torch.Tensor):
            # graph_data.batch maps each node to its graph index in the batch
            if hasattr(graph_data, 'batch') and graph_data.batch is not None:
                F0 = F0[graph_data.batch].view(-1, 1)
            else:
                # If no batch info, assume all nodes belong to the same graph
                F0 = F0[0].item() if F0.numel() > 0 else 1.0
                
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
        
        res_mass_raw = (rho_next - rho_curr) / self.dt + div_rho_v
        res_mass = res_mass_raw / (torch.abs(rho_curr) + 1e-3)
        
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
        
        res_momentum_raw = rho_next * ((v_par_next - v_par_curr) / self.dt + v_dot_grad_v_par) + grad_par_rho_T
        res_momentum = res_momentum_raw / (torch.abs(rho_curr * v_par_curr) + 1e-3)

        # Physics Loss only on internal nodes
        zero = torch.tensor(0.0, device=device)
        if internal_mask is not None and internal_mask.any():
            mass_term = torch.mean(res_mass[internal_mask] ** 2)
            mom_term = torch.mean(res_momentum[internal_mask] ** 2)
        else:
            mass_term = torch.mean(res_mass ** 2)
            mom_term = torch.mean(res_momentum ** 2)
        phys_loss = mass_term + mom_term

        # Boundary condition anchoring
        loss_bc = zero
        if boundary_mask is not None and boundary_mask.any():
            pred_bnd = pred[boundary_mask]
            target_bnd = target[boundary_mask]
            if pred_bnd.shape[0] > 0:
                loss_bc_plasma = F.mse_loss(pred_bnd[:, 1:], target_bnd[:, 1:])
                psi_penalty = 5.0 if freeboundary_flag else 50.0
                loss_bc_psi = F.mse_loss(pred_bnd[:, 0], target_bnd[:, 0]) * psi_penalty
                loss_bc = loss_bc_plasma + loss_bc_psi

        total_loss = self.lambda_data * data_loss + self.lambda_phys * phys_loss + self.lambda_bc * loss_bc
        
        return total_loss
