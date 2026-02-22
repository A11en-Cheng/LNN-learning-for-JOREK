import torch

def poisson_bracket(A: torch.Tensor, B: torch.Tensor, grad_A: torch.Tensor, grad_B: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """
    Compute the Poisson bracket [A, B] in cylindrical coordinates (R, Z).
    [A, B] = (1/R) * (d_A/dR * d_B/dZ - d_A/dZ * d_B/dR)
    
    Args:
        A: [num_nodes, 1]
        B: [num_nodes, 1]
        grad_A: [num_nodes, 2] (d_A/dR, d_A/dZ)
        grad_B: [num_nodes, 2] (d_B/dR, d_B/dZ)
        R: [num_nodes, 1]
        
    Returns:
        bracket: [num_nodes, 1]
    """
    dA_dR = grad_A[:, 0:1]
    dA_dZ = grad_A[:, 1:2]
    
    dB_dR = grad_B[:, 0:1]
    dB_dZ = grad_B[:, 1:2]
    
    eps = 1e-8
    R_safe = R + eps
    
    bracket = (dA_dR * dB_dZ - dA_dZ * dB_dR) / R_safe
    return bracket

def reconstruct_velocity(grad_Phi: torch.Tensor, v_par: torch.Tensor, B_field: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct the velocity field v from the electric potential Phi and parallel velocity v_par.
    v_pol = [-R * d_Phi/dZ, R * d_Phi/dR]
    v = v_pol + v_par * B / |B|
    
    Args:
        grad_Phi: [num_nodes, 2] (d_Phi/dR, d_Phi/dZ)
        v_par: [num_nodes, 1]
        B_field: [num_nodes, 3] (B_R, B_Z, B_phi)
        R: [num_nodes, 1]
        
    Returns:
        v_field: [num_nodes, 3] (v_R, v_Z, v_phi)
    """
    dPhi_dR = grad_Phi[:, 0:1]
    dPhi_dZ = grad_Phi[:, 1:2]
    
    v_pol_R = -R * dPhi_dZ
    v_pol_Z = R * dPhi_dR
    
    # B_field is [num_nodes, 3]
    B_mag = torch.norm(B_field, p=2, dim=-1, keepdim=True) + 1e-8
    B_hat = B_field / B_mag
    
    v_R = v_pol_R + v_par * B_hat[:, 0:1]
    v_Z = v_pol_Z + v_par * B_hat[:, 1:2]
    v_phi = v_par * B_hat[:, 2:3]
    
    v_field = torch.cat([v_R, v_Z, v_phi], dim=-1)
    return v_field
