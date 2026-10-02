import torch
from torch_geometric.utils import scatter


def _solve2x2(normal, rhs):
    """Deterministic closed-form 2x2 solve.

    ``torch.linalg.solve`` dispatches to cuSOLVER on CUDA; keeping this
    closed-form inverse makes the graph operators bitwise-identical on CPU and
    GPU, removing a device-dependent variable from the physics residuals.
    ``normal`` is [N, 2, 2] and ``rhs`` is [N, 2, F].
    """
    a = normal[:, 0, 0]
    b = normal[:, 0, 1]
    c = normal[:, 1, 1]
    det = (a * c - b * b).unsqueeze(-1).unsqueeze(-1)
    inverse = torch.stack(
        [torch.stack([c, -b], dim=-1), torch.stack([-b, a], dim=-1)], dim=-2
    ) / det
    return torch.matmul(inverse, rhs)


def _normal_equations(row, delta_pos, distance, weight):
    """Inverse-distance-weighted least-squares normal equations per node.

    The previous scalar ``sum(weight)`` normalisation underestimated gradients
    by a mesh-dependent factor (~2x on the 4125-node Delaunay mesh, with large
    node-to-node noise).  Solving the 2x2 normal system instead makes the
    gradient exact for linear fields on any stencil and stabilises the
    Laplacian computed via ``div(grad(...))``.
    """
    inv_dist_sq = 1.0 / (distance.squeeze(1).clamp_min(1e-8) ** 2)
    inv_dist_sq = inv_dist_sq.unsqueeze(-1).unsqueeze(-1)
    weight = weight.unsqueeze(-1)
    normal = (
        weight
        * delta_pos.unsqueeze(-1)
        * delta_pos.unsqueeze(-2)
        * inv_dist_sq
    )
    num_nodes = int(row.max()) + 1
    normal = scatter(normal, row, dim=0, dim_size=num_nodes, reduce="sum")
    eye = torch.eye(2, device=normal.device, dtype=normal.dtype).unsqueeze(0)
    return normal + 1e-8 * eye, inv_dist_sq


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
    
    normal, inv_dist_sq = _normal_equations(row, delta_pos, distance, weight)

    # rhs = sum_e w_e (f_j - f_i) delta_pos_e / d_e^2, then grad = normal^-1 rhs.
    rhs = (
        weight.unsqueeze(-1)
        * delta_f.unsqueeze(-1)
        * delta_pos.unsqueeze(1)
        * inv_dist_sq
    )
    rhs = rhs.permute(0, 2, 1)  # [edges, 2, features]
    rhs = scatter(rhs, row, dim=0, dim_size=num_nodes, reduce="sum")
    grad = _solve2x2(normal, rhs).permute(0, 2, 1)  # [nodes, features, 2]

    if squeeze_output:
        grad = grad.squeeze(1)  # [num_nodes, 2]

    return grad


def build_gradient_operator(
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    n_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute the (linear) graph least-squares gradient as a sparse operator.

    ``compute_gradient`` is linear in the field, so the whole operator can be
    baked into

        ``rhs = A @ f``          (A is sparse ``[2 * n_nodes, n_nodes]``)
        ``grad = normal_inv @ rhs``  (per-node 2x2 mixing, ``[n_nodes, 2, 2]``)

    so that gradients of *model-generated* fields cost one sparse matmul
    (``torch.sparse.mm``) plus a tiny einsum.  This is what makes the
    decode -> gradient -> re-encode feedback affordable inside training.
    """
    row, col = edge_index
    delta_pos = edge_attr[:, :2]
    distance = edge_attr[:, 2:3]
    weight = 1.0 / (distance + 1e-8)
    normal, inv_dist_sq = _normal_equations(row, delta_pos, distance, weight)
    coefficient = (weight * inv_dist_sq.squeeze(-1)).squeeze(-1)      # [E]

    rows = torch.stack([2 * row, 2 * row + 1], dim=0).reshape(-1)     # [2E]
    # component-major ordering must match ``rows``
    values = torch.stack(
        [coefficient * delta_pos[:, 0], coefficient * delta_pos[:, 1]], dim=0
    ).reshape(-1)
    indices = torch.stack(
        [
            torch.cat([rows, rows]),
            torch.cat([col.repeat(2), row.repeat(2)]),
        ]
    )
    a_mat = torch.sparse_coo_tensor(
        indices, torch.cat([values, -values]), (2 * n_nodes, n_nodes)
    ).coalesce()

    a = normal[:, 0, 0]
    b = normal[:, 0, 1]
    c = normal[:, 1, 1]
    det = (a * c - b * b).clamp_min(1e-30)
    normal_inv = torch.stack(
        [torch.stack([c, -b], dim=-1), torch.stack([-b, a], dim=-1)], dim=-2
    ) / det[:, None, None]
    return a_mat, normal_inv


def apply_gradient_operator(
    operator: tuple[torch.Tensor, torch.Tensor], features: torch.Tensor
) -> torch.Tensor:
    """Apply :func:`build_gradient_operator` to nodal features."""
    a_mat, normal_inv = operator
    squeeze = features.dim() == 1
    field = features.unsqueeze(-1) if squeeze else features
    n_nodes = field.shape[0]
    rhs = torch.sparse.mm(a_mat, field).view(n_nodes, 2, field.shape[1])
    grad = torch.einsum("nab,nbf->naf", normal_inv, rhs).permute(0, 2, 1)
    return grad.squeeze(1) if squeeze else grad

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
    
    normal, inv_dist_sq = _normal_equations(row, delta_pos, distance, weight)

    # Assemble the gradient-Jacobian normal system and read its trace.
    rhs = (
        weight.unsqueeze(-1)
        * delta_F.unsqueeze(-1)
        * delta_pos.unsqueeze(-2)
        * inv_dist_sq
    )
    rhs = scatter(rhs, row, dim=0, dim_size=num_nodes, reduce="sum")
    jacobian = _solve2x2(normal, rhs)  # [nodes, 2, 2]
    return jacobian[:, 0, 0] + jacobian[:, 1, 1]  # [num_nodes]
