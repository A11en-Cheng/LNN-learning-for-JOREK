import torch
from torch_geometric.utils import scatter

def compute_gradient(features: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
    """
    Compute the spatial gradient of node features on an unstructured mesh using Distance-Weighted Finite Difference.
    
    Args:
        features: Node features [num_nodes, num_features] or [num_nodes]
        edge_index: Graph connectivity [2, num_edges]
        edge_attr: Edge attributes [num_edges, 3] where columns are (delta_R, delta_Z, distance)
                   delta_R = R_j - R_i, delta_Z = Z_j - Z_i
                   
    Returns:
        gradients: [num_nodes, num_features, 2] where the last dimension is (d/dR, d/dZ).
                   If input is [num_nodes], output is [num_nodes, 2].
    """
    if features.dim() == 1:
        features = features.unsqueeze(-1)
        squeeze_output = True
    else:
        squeeze_output = False
        
    num_nodes = features.size(0)
    row, col = edge_index  # row is source (i), col is target (j)
    
    # Extract edge attributes
    delta_pos = edge_attr[:, :2]  # [num_edges, 2] (delta_R, delta_Z)
    distance = edge_attr[:, 2:3]  # [num_edges, 1]
    
    # Avoid division by zero
    eps = 1e-8
    dist_sq = distance ** 2 + eps
    weight = 1.0 / (distance + eps)  # w_ij = 1 / d_ij
    
    # Feature difference: f_j - f_i
    f_i = features[row]  # [num_edges, num_features]
    f_j = features[col]  # [num_edges, num_features]
    delta_f = f_j - f_i  # [num_edges, num_features]
    
    # Compute the term for each edge: w_ij * (f_j - f_i) / d_ij^2 * r_ij
    # delta_f is [num_edges, num_features]
    # delta_pos is [num_edges, 2]
    # We want [num_edges, num_features, 2]
    term = weight.unsqueeze(-1) * (delta_f.unsqueeze(-1) / dist_sq.unsqueeze(-1)) * delta_pos.unsqueeze(1)
    
    # Sum over neighbors (target nodes j for each source node i)
    # We scatter_add into the 'row' index (node i)
    grad_sum = scatter(term, row, dim=0, dim_size=num_nodes, reduce='sum')  # [num_nodes, num_features, 2]
    
    # Sum of weights for normalization
    weight_sum = scatter(weight, row, dim=0, dim_size=num_nodes, reduce='sum')  # [num_nodes, 1]
    weight_sum = weight_sum.unsqueeze(-1) + eps  # [num_nodes, 1, 1]
    
    # Final gradient
    grad = grad_sum / weight_sum
    
    if squeeze_output:
        grad = grad.squeeze(1)  # [num_nodes, 2]
        
    return grad

def compute_divergence(vector_field: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
    """
    Compute the divergence of a vector field on an unstructured mesh.
    
    Args:
        vector_field: Node vector features [num_nodes, 2] (F_R, F_Z)
        edge_index: Graph connectivity [2, num_edges]
        edge_attr: Edge attributes [num_edges, 3] (delta_R, delta_Z, distance)
        
    Returns:
        divergence: [num_nodes]
    """
    num_nodes = vector_field.size(0)
    row, col = edge_index
    
    delta_pos = edge_attr[:, :2]  # [num_edges, 2]
    distance = edge_attr[:, 2:3]  # [num_edges, 1]
    
    eps = 1e-8
    dist_sq = distance ** 2 + eps
    weight = 1.0 / (distance + eps)
    
    F_i = vector_field[row]  # [num_edges, 2]
    F_j = vector_field[col]  # [num_edges, 2]
    delta_F = F_j - F_i      # [num_edges, 2]
    
    # Dot product of delta_F and delta_pos: (F_R_j - F_R_i)*delta_R + (F_Z_j - F_Z_i)*delta_Z
    # delta_F: [num_edges, 2], delta_pos: [num_edges, 2]
    dot_product = (delta_F * delta_pos).sum(dim=-1, keepdim=True)  # [num_edges, 1]
    
    # Term: w_ij * dot_product / d_ij^2
    term = weight * dot_product / dist_sq  # [num_edges, 1]
    
    # Sum over neighbors
    div_sum = scatter(term, row, dim=0, dim_size=num_nodes, reduce='sum')  # [num_nodes, 1]
    
    # Sum of weights
    weight_sum = scatter(weight, row, dim=0, dim_size=num_nodes, reduce='sum')  # [num_nodes, 1]
    weight_sum = weight_sum + eps
    
    div = div_sum / weight_sum
    
    return div.squeeze(-1)  # [num_nodes]
