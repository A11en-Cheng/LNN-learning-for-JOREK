import torch

def get_magnetic_field(psi_gradient: torch.Tensor, R: torch.Tensor, F0: float) -> torch.Tensor:
    """
    Compute the magnetic field vector B from the poloidal flux psi.
    B = (F0/R) * phi_hat + (1/R) * grad(psi) x phi_hat
    
    In cylindrical coordinates (R, Z, phi):
    grad(psi) x phi_hat = (d_psi/dR * R_hat + d_psi/dZ * Z_hat) x phi_hat
                        = -d_psi/dR * Z_hat + d_psi/dZ * R_hat
                        
    So:
    B_R = (1/R) * d_psi/dZ
    B_Z = -(1/R) * d_psi/dR
    B_phi = F0 / R
    
    Args:
        psi_gradient: [num_nodes, 2] where columns are (d_psi/dR, d_psi/dZ)
        R: [num_nodes, 1] R coordinates of the nodes
        F0: float, normalization constant
        
    Returns:
        B_field: [num_nodes, 3] where columns are (B_R, B_Z, B_phi)
    """
    dpsi_dR = psi_gradient[:, 0:1]
    dpsi_dZ = psi_gradient[:, 1:2]
    
    # Avoid division by zero
    eps = 1e-8
    R_safe = R + eps
    
    B_R = dpsi_dZ / R_safe
    B_Z = -dpsi_dR / R_safe
    
    if isinstance(F0, torch.Tensor):
        # If F0 is already expanded to match R_safe shape, just divide
        # Ensure F0 has the same shape as R_safe [num_nodes, 1]
        if F0.shape != R_safe.shape:
            F0 = F0.view(-1, 1)
        B_phi = F0 / R_safe
    else:
        B_phi = torch.full_like(R_safe, float(F0)) / R_safe
    
    B_field = torch.cat([B_R, B_Z, B_phi], dim=-1)
    return B_field

def get_velocity_field(phi_gradient: torch.Tensor, B_field: torch.Tensor, v_par: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """
    Compute the velocity field vector v from the electric potential Phi and parallel velocity v_par.
    v = -R * grad(Phi) x phi_hat + v_par * B
    
    In cylindrical coordinates (R, Z, phi):
    grad(Phi) x phi_hat = (d_Phi/dR * R_hat + d_Phi/dZ * Z_hat) x phi_hat
                        = -d_Phi/dR * Z_hat + d_Phi/dZ * R_hat
                        
    So:
    v_R = -R * d_Phi/dZ + v_par * B_R
    v_Z = R * d_Phi/dR + v_par * B_Z
    v_phi = v_par * B_phi
    
    Args:
        phi_gradient: [num_nodes, 2] where columns are (d_Phi/dR, d_Phi/dZ)
        B_field: [num_nodes, 3] where columns are (B_R, B_Z, B_phi)
        v_par: [num_nodes, 1] parallel velocity
        R: [num_nodes, 1] R coordinates of the nodes
        
    Returns:
        v_field: [num_nodes, 3] where columns are (v_R, v_Z, v_phi)
    """
    dPhi_dR = phi_gradient[:, 0:1]
    dPhi_dZ = phi_gradient[:, 1:2]
    
    B_R = B_field[:, 0:1]
    B_Z = B_field[:, 1:2]
    B_phi = B_field[:, 2:3]
    
    v_R = -R * dPhi_dZ + v_par * B_R
    v_Z = R * dPhi_dR + v_par * B_Z
    v_phi = v_par * B_phi
    
    v_field = torch.cat([v_R, v_Z, v_phi], dim=-1)
    return v_field
