import sys
import os
import torch
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import JorekSequenceDataset
from src.physics.grad_ops import compute_gradient
from src.physics.mhd import get_magnetic_field

def test_grad_ops():
    print("==================================================")
    print("🚀 Starting Gradient Sanity Checker")
    print("==================================================")
    
    DATA_DIR = "data/test_data_for_pipeline_EHL-2"
    
    # 1. Load a single Ground Truth frame
    print("[1/4] Loading a single Ground Truth frame...")
    dataset = JorekSequenceDataset(root_dir=DATA_DIR, seq_len=1)
    if len(dataset) == 0:
        print("Error: Dataset is empty.")
        return
        
    # Get the first frame
    data_seq = dataset[-1]
    graph_data = data_seq[0]
    
    print(f"   Loaded graph with {graph_data.num_nodes} nodes and {graph_data.num_edges} edges.")
    
    # 2. Extract variables
    # Variables: 0: psi, 1: Phi, 2: rho, 3: T, 4: v_par
    # Data is already denormalized (raw from HDF5)
    psi = graph_data.x[:, 0:1]
    R = graph_data.pos[:, 0:1]
    Z = graph_data.pos[:, 1:2]
    
    # Get F0
    F0 = 1.0
    if hasattr(graph_data, 'norm_factors') and graph_data.norm_factors is not None:
        if isinstance(graph_data.norm_factors, dict) and 'F0' in graph_data.norm_factors:
            F0 = graph_data.norm_factors['F0']
        elif hasattr(graph_data.norm_factors, 'F0'):
            F0 = graph_data.norm_factors.F0
            
    if isinstance(F0, torch.Tensor):
        F0 = F0[0].item() if F0.numel() > 0 else 1.0
        
    print(f"   Extracted Psi. F0 = {F0}")
    
    # 3. Compute spatial gradients
    print("[2/4] Computing spatial gradients of Psi...")
    grad_psi = compute_gradient(psi, graph_data.edge_index, graph_data.edge_attr).squeeze(1)
    
    dpsi_dR = grad_psi[:, 0].numpy()
    dpsi_dZ = grad_psi[:, 1].numpy()
    
    # 4. Calculate B field
    print("[3/4] Calculating B field...")
    B_field = get_magnetic_field(grad_psi, R, F0)
    
    B_R = B_field[:, 0].numpy()
    B_Z = B_field[:, 1].numpy()
    B_phi = B_field[:, 2].numpy()
    
    # 5. Plotting
    print("[4/4] Plotting results...")
    os.makedirs("output/verification", exist_ok=True)
    
    R_np = R.squeeze().numpy()
    Z_np = Z.squeeze().numpy()
    psi_np = psi.squeeze().numpy()
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot Psi
    sc0 = axs[0, 0].scatter(R_np, Z_np, c=psi_np, cmap='viridis', s=1)
    axs[0, 0].set_title('Poloidal Flux ($\\psi$)')
    axs[0, 0].set_xlabel('R')
    axs[0, 0].set_ylabel('Z')
    plt.colorbar(sc0, ax=axs[0, 0])
    
    # Plot dpsi/dR
    sc1 = axs[0, 1].scatter(R_np, Z_np, c=dpsi_dR, cmap='coolwarm', s=1)
    axs[0, 1].set_title('$\\partial \\psi / \\partial R$')
    axs[0, 1].set_xlabel('R')
    axs[0, 1].set_ylabel('Z')
    plt.colorbar(sc1, ax=axs[0, 1])
    
    # Plot dpsi/dZ
    sc2 = axs[1, 0].scatter(R_np, Z_np, c=dpsi_dZ, cmap='coolwarm', s=1)
    axs[1, 0].set_title('$\\partial \\psi / \\partial Z$')
    axs[1, 0].set_xlabel('R')
    axs[1, 0].set_ylabel('Z')
    plt.colorbar(sc2, ax=axs[1, 0])
    
    # Plot B field vectors (subsampled for clarity)
    # Subsample
    stride = max(1, len(R_np) // 1000)
    axs[1, 1].quiver(R_np[::stride], Z_np[::stride], B_R[::stride], B_Z[::stride], 
                     color='black', scale=None, alpha=0.7)
    axs[1, 1].set_title('Poloidal Magnetic Field ($B_R, B_Z$)')
    axs[1, 1].set_xlabel('R')
    axs[1, 1].set_ylabel('Z')
    
    plt.tight_layout()
    save_path = "output/verification/grad_ops_sanity_check.png"
    plt.savefig(save_path, dpi=300)
    print(f"   Saved plot to {save_path}")
    
if __name__ == "__main__":
    test_grad_ops()
