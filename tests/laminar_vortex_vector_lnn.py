"""
Laminar-Vortex Flow LNN Training - Vector Field Prediction
===========================================================
This script predicts 2D velocity vector field (u_x, u_y) instead of scalar magnitude.

Enhanced features:
- Predicts full velocity components (u_x, u_y)
- Vector field visualization with predictions
- Directional error analysis
- Advanced metrics for vector field accuracy
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches
from ncps.torch import CfC
from ncps.wirings import AutoNCP
import random

# =============================================================================
# PART 1: 2D Laminar-Vortex Flow Physics (Same as before)
# =============================================================================

class LaminarVortexFlow:
    """2D laminar flow with vortex pair evolution."""
    
    def __init__(self, Lx=2*np.pi, Ly=2*np.pi, nx=32, ny=32, 
                 U0=1.0, nu=0.01, vortex_strength=1.0):
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.U0 = U0
        self.nu = nu
        self.vortex_strength = vortex_strength
        
        # Create grid
        self.x = np.linspace(0, Lx, nx, endpoint=False)
        self.y = np.linspace(-Ly/2, Ly/2, ny, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        
        # Initial vortex positions
        self.vortex1_x0 = Lx / 4
        self.vortex1_y0 = Ly / 4
        self.vortex2_x0 = 3 * Lx / 4
        self.vortex2_y0 = -Ly / 4
        
        self.vortex_core_radius = 0.2
        
    def laminar_base_flow(self, y):
        """Base laminar shear layer."""
        return self.U0 * np.tanh(y / 0.3)
    
    def vortex_pair_field(self, t):
        """Vortex pair velocity field with advection and diffusion."""
        decay_factor = np.exp(-self.nu * t / self.vortex_core_radius**2)
        current_strength = self.vortex_strength * decay_factor
        
        # Vortex self-advection
        advection_velocity = self.U0
        vortex1_x = self.vortex1_x0 + advection_velocity * t
        vortex2_x = self.vortex2_x0 + advection_velocity * t
        
        vortex1_x = vortex1_x % self.Lx
        vortex2_x = vortex2_x % self.Lx
        
        # Diffusing Gaussian cores
        core_radius = self.vortex_core_radius * np.sqrt(1 + 4 * self.nu * t / self.vortex_core_radius**2)
        
        # Vortex 1 (counter-clockwise, +z direction)
        r1_sq = (self.X - vortex1_x)**2 + (self.Y - self.vortex1_y0)**2
        u_x_v1 = -current_strength * (self.Y - self.vortex1_y0) / (r1_sq + core_radius**2)
        u_y_v1 = current_strength * (self.X - vortex1_x) / (r1_sq + core_radius**2)
        
        # Vortex 2 (clockwise, -z direction)
        r2_sq = (self.X - vortex2_x)**2 + (self.Y - self.vortex2_y0)**2
        u_x_v2 = current_strength * (self.Y - self.vortex2_y0) / (r2_sq + core_radius**2)
        u_y_v2 = -current_strength * (self.X - vortex2_x) / (r2_sq + core_radius**2)
        
        # Total vortex field
        u_x_vortex = u_x_v1 + u_x_v2
        u_y_vortex = u_y_v1 + u_y_v2
        
        # Laminar shear contribution
        u_x_laminar = self.laminar_base_flow(self.Y)
        
        u_x = u_x_laminar + u_x_vortex
        u_y = u_y_vortex
        
        return u_x, u_y
    
    def velocity_at_time(self, t):
        """Get velocity field at time t."""
        return self.vortex_pair_field(t)


# =============================================================================
# PART 2: Data Generation - Now Predicting Vector Components
# =============================================================================

def generate_laminar_vortex_vector_dataset(flow, num_snapshots=100, t_max=10.0, 
                                           spatial_samples=8, batch_size=1):
    """
    Generate training dataset with vector field predictions.
    
    Args:
        flow: LaminarVortexFlow instance
        num_snapshots: Number of time snapshots
        t_max: Maximum simulation time
        spatial_samples: Grid points per side for sampling (8x8=64 points)
        batch_size: Number of sequences in a batch
    """
    
    seq_len = num_snapshots
    
    # Irregular time sampling
    dt = torch.rand(batch_size, seq_len) * 0.15 + 0.02
    times = torch.cumsum(dt, dim=1)
    times = torch.clamp(times, max=t_max)
    
    print(f"Batch size: {batch_size}")
    print(f"Time span (batch0): {times[0, 0]:.4f} to {times[0, -1]:.4f}")
    print(f"Mean dt: {dt.mean():.4f}, Range: [{dt.min():.4f}, {dt.max():.4f}]")
    
    # Pre-compute reference snapshots
    t_refs = np.linspace(0, t_max, num_snapshots)
    u_x_refs = []
    u_y_refs = []
    
    for t in t_refs:
        u_x, u_y = flow.velocity_at_time(t)
        u_x_refs.append(u_x)
        u_y_refs.append(u_y)
    
    # Spatial sampling points
    stride = max(1, flow.nx // spatial_samples)
    sample_indices = np.arange(0, flow.nx, stride)[:spatial_samples]
    
    print(f"\nSpatial sampling: {spatial_samples}x{spatial_samples} = {spatial_samples**2} points")
    
    # Generate data sequences
    data_sequences_list = []
    target_sequences_list = []
    
    for b in range(batch_size):
        data_list = []
        target_list = []
        for i, t in enumerate(times[b].numpy()):
            # Find nearest reference snapshot
            idx = np.argmin(np.abs(t_refs - t))
            u_x_ref = u_x_refs[idx]
            u_y_ref = u_y_refs[idx]
            
            # Extract sampled velocity components
            u_x_sampled = u_x_ref[np.ix_(sample_indices, sample_indices)].flatten()
            u_y_sampled = u_y_ref[np.ix_(sample_indices, sample_indices)].flatten()
            
            # Interleave u_x and u_y: [u_x0, u_y0, u_x1, u_y1, ...]
            # Alternative: concatenate [u_x0, u_x1, ..., u_y0, u_y1, ...]
            # We use interleave for better correlation
            features = np.zeros(2 * spatial_samples**2)
            features[0::2] = u_x_sampled
            features[1::2] = u_y_sampled
            
            data_list.append(features)
            target_list.append(features)  # Target is the same (input features are already the ground truth)
        data_sequences_list.append(torch.FloatTensor(np.array(data_list[:-1])))
        target_sequences_list.append(torch.FloatTensor(np.array(target_list[1:])))
    
    data_sequences = torch.stack(data_sequences_list, dim=0)  # [batch, seq_len-1, features]
    target_sequences = torch.stack(target_sequences_list, dim=0)  # [batch, seq_len-1, features]
    
    # Add noise to input
    noise_level = 0.03
    data_sequences = data_sequences + noise_level * torch.randn_like(data_sequences)
    
    print(f"\nData shapes:")
    print(f"  Inputs: {data_sequences.shape}")
    print(f"  Targets: {target_sequences.shape}")
    print(f"  Feature dimensions: 2*{spatial_samples}² = {2*spatial_samples**2}")
    
    return times, dt, data_sequences, target_sequences, t_refs, u_x_refs, u_y_refs, sample_indices


# =============================================================================
# PART 3: Vector Field LNN Model
# =============================================================================

class VectorFieldLNN(nn.Module):
    """LNN model for vector field prediction."""
    
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lnn = CfC(
            input_size=input_size,
            units=hidden_size,
            proj_size=output_size,
            return_sequences=True
        )
    
    def forward(self, x, times=None):
        """
        Args:
            x: [batch, seq_len, input_size]
            times: Optional time information
        
        Returns:
            output: [batch, seq_len, output_size]
        """
        output, _ = self.lnn(x)
        return output


# =============================================================================
# PART 4: Vector Field Error Metrics
# =============================================================================

def compute_vector_metrics(pred_components, true_components, spatial_samples):
    """
    Compute vector field specific metrics.
    
    Args:
        pred_components: Predictions [seq_len, 2*samples²]
        true_components: Ground truth [seq_len, 2*samples²]
        spatial_samples: Number of spatial sample points per side
    
    Returns:
        Dictionary with various metrics
    """
    seq_len = pred_components.shape[0]
    n_points = spatial_samples ** 2
    
    # Reshape to separate u_x and u_y
    pred_ux = pred_components[:, 0::2].reshape(seq_len, spatial_samples, spatial_samples)
    pred_uy = pred_components[:, 1::2].reshape(seq_len, spatial_samples, spatial_samples)
    true_ux = true_components[:, 0::2].reshape(seq_len, spatial_samples, spatial_samples)
    true_uy = true_components[:, 1::2].reshape(seq_len, spatial_samples, spatial_samples)
    
    # Component-wise errors
    error_ux = np.abs(pred_ux - true_ux)
    error_uy = np.abs(pred_uy - true_uy)
    
    # Magnitude error
    pred_mag = np.sqrt(pred_ux**2 + pred_uy**2)
    true_mag = np.sqrt(true_ux**2 + true_uy**2)
    error_mag = np.abs(pred_mag - true_mag)
    
    # Directional error (angle between vectors)
    cos_angle = (pred_ux * true_ux + pred_uy * true_uy) / (pred_mag * true_mag + 1e-8)
    cos_angle = np.clip(cos_angle, -1, 1)
    angle_error = np.arccos(cos_angle) * 180 / np.pi  # degrees
    
    metrics = {
        'mae_ux': error_ux.mean(),
        'mae_uy': error_uy.mean(),
        'mae_magnitude': error_mag.mean(),
        'mae_angle': angle_error.mean(),
        'rmse_ux': np.sqrt((error_ux**2).mean()),
        'rmse_uy': np.sqrt((error_uy**2).mean()),
        'rmse_magnitude': np.sqrt((error_mag**2).mean()),
        'max_error_ux': error_ux.max(),
        'max_error_uy': error_uy.max(),
        'max_error_magnitude': error_mag.max(),
    }
    
    return metrics, pred_ux, pred_uy, true_ux, true_uy, error_ux, error_uy


# =============================================================================
# PART 5: Training Loop
# =============================================================================

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_vector_field_lnn(batch_size: int = 1):
    """Main training function for vector field prediction."""
    
    print("=" * 70)
    print("2D Laminar-Vortex Flow LNN - Vector Field Prediction")
    print("=" * 70)
    
    # Create flow
    print("\n[1] Initializing 2D Laminar-Vortex Flow...")
    set_seed(42)
    flow = LaminarVortexFlow(
        Lx=2*np.pi, Ly=2*np.pi, nx=32, ny=32,
        U0=1.0, nu=0.01, vortex_strength=2.0
    )
    
    # Generate dataset
    print("\n[2] Generating vector field training dataset...")
    spatial_samples = 16  # 16x16 = 256 spatial points
    times, dt, data_sequences, target_sequences, t_refs, u_x_refs, u_y_refs, sample_indices = \
        generate_laminar_vortex_vector_dataset(flow, num_snapshots=150, t_max=12.0, 
                                               spatial_samples=spatial_samples, batch_size=16)
    
    # Model setup
    print("\n[3] Setting up Vector Field LNN model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")
    
    input_size = data_sequences.shape[-1]  # 2*spatial_samples²
    hidden_size = 128
    output_size = target_sequences.shape[-1]  # 2*spatial_samples²
    
    model = VectorFieldLNN(input_size, hidden_size, output_size)
    model.to(device)
    
    data_sequences = data_sequences.to(device)
    target_sequences = target_sequences.to(device)
    
    print(f"   Input size: {input_size}")
    print(f"   Hidden size: {hidden_size}")
    print(f"   Output size: {output_size}")
    
    # Training
    print("\n[4] Starting training (10000 epochs)...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                        optimizer, mode='min', factor=0.5, patience=500
                                    )
    
    criterion = nn.MSELoss()
    
    losses = []
    best_loss = float('inf')
    no_improve = 0
    early_stop_patience = 2000
    
    for epoch in range(10000):
        optimizer.zero_grad()
        
        predictions = model(data_sequences, times)
        loss = criterion(predictions, target_sequences)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        losses.append(loss.item())
        
        if loss.item() < best_loss - 1e-6:
            best_loss = loss.item()
            no_improve = 0
        else:
            no_improve += 1
        
        if (epoch + 1) % 100 == 0:
            print(f"   Epoch {epoch+1:4d}/10000 | Loss: {loss.item():.6e} | Best: {best_loss:.6e}")
            
        scheduler.step(loss.item())
        
        if no_improve >= early_stop_patience:
            print(f"   Early stopping at epoch {epoch+1} (no improvement for {early_stop_patience} epochs).")
            break
    
    print("\n[5] Training completed!")
    print(f"   Final loss: {losses[-1]:.6e}")
    print(f"   Best loss: {best_loss:.6e}")
    
    # =============================================================================
    # PART 6: Evaluation and Visualization
    # =============================================================================
    
    print("\n[6] Computing vector field metrics and generating plots...")
    
    with torch.no_grad():
        predictions = model(data_sequences, times)
    
    pred_np = predictions.cpu().numpy()[0, :, :]
    target_np = target_sequences.cpu().numpy()[0, :, :]
    times_np = times[0, :-1].cpu().numpy()  # Exclude last time point
    
    # Compute metrics
    metrics, pred_ux, pred_uy, true_ux, true_uy, error_ux, error_uy = \
        compute_vector_metrics(pred_np, target_np, spatial_samples)
    
    # Plot 1: Training Loss
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    epochs = np.arange(1, len(losses) + 1)
    ax.semilogy(epochs, losses, linewidth=1.5, color='#1f77b4')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('Vector Field Prediction - Training Loss Evolution', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lnn_vector_loss_evolution.png', dpi=150, bbox_inches='tight')
    print("   ✓ Loss plot saved: lnn_vector_loss_evolution.png")
    plt.close()
    
    # Plot 2: Component-wise Errors
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=150)
    
    # u_x error time series
    error_ux_mean = error_ux.mean(axis=(1, 2))
    error_uy_mean = error_uy.mean(axis=(1, 2))
    
    ax = axes[0, 0]
    ax.plot(times_np, error_ux_mean, 'r-', linewidth=1.5, label='u_x')
    ax.fill_between(times_np, 0, error_ux_mean, color='red', alpha=0.2)
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('u_x Component Error')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # u_y error
    ax = axes[0, 1]
    ax.plot(times_np, error_uy_mean, 'g-', linewidth=1.5, label='u_y')
    ax.fill_between(times_np, 0, error_uy_mean, color='green', alpha=0.2)
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('u_y Component Error')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Both components
    ax = axes[1, 0]
    ax.plot(times_np, error_ux_mean, 'r-', linewidth=1.5, label='u_x')
    ax.plot(times_np, error_uy_mean, 'g-', linewidth=1.5, label='u_y')
    ax.set_xlabel('Time (t)')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_title('Component-wise Error Comparison')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Scatter: True vs Pred
    ax = axes[1, 1]
    pred_all = np.concatenate([pred_ux.flatten(), pred_uy.flatten()])
    true_all = np.concatenate([true_ux.flatten(), true_uy.flatten()])
    ax.scatter(true_all, pred_all, alpha=0.3, s=10, color='#9467bd')
    min_val = min(true_all.min(), pred_all.min())
    max_val = max(true_all.max(), pred_all.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2)
    ax.set_xlabel('Ground Truth')
    ax.set_ylabel('Prediction')
    ax.set_title('Velocity Components Accuracy')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Vector Field Component Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('lnn_vector_component_analysis.png', dpi=150, bbox_inches='tight')
    print("   ✓ Component analysis saved: lnn_vector_component_analysis.png")
    plt.close()
    
    # Plot 3: Vector Field Snapshots - Predictions vs Ground Truth
    print("   Generating vector field visualizations...")
    time_indices = [0, len(times_np)//4, len(times_np)//2, 3*len(times_np)//4, len(times_np)-1]
    
    fig = plt.figure(figsize=(18, 10), dpi=150)
    
    for plot_idx, t_idx in enumerate(time_indices):
        if t_idx >= len(times_np):
            t_idx = len(times_np) - 1
        
        # Ground truth
        true_u_x = true_ux[t_idx]
        true_u_y = true_uy[t_idx]
        true_mag = np.sqrt(true_u_x**2 + true_u_y**2)
        
        # Prediction
        pred_u_x = pred_ux[t_idx]
        pred_u_y = pred_uy[t_idx]
        pred_mag = np.sqrt(pred_u_x**2 + pred_u_y**2)
        
        # Ground truth
        ax = plt.subplot(2, len(time_indices), plot_idx + 1)
        im = ax.contourf(true_mag, levels=15, cmap='viridis')
        stride = 1
        ax.quiver(true_u_x[::stride, ::stride], true_u_y[::stride, ::stride],
                 alpha=0.6, scale=3, scale_units='inches')
        ax.set_title(f'Ground Truth (t={times_np[t_idx]:.2f})', fontsize=10, fontweight='bold')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='|V|')
        
        # Prediction
        ax = plt.subplot(2, len(time_indices), len(time_indices) + plot_idx + 1)
        im = ax.contourf(pred_mag, levels=15, cmap='viridis')
        ax.quiver(pred_u_x[::stride, ::stride], pred_u_y[::stride, ::stride],
                 alpha=0.6, scale=3, scale_units='inches')
        ax.set_title(f'LNN Prediction (t={times_np[t_idx]:.2f})', fontsize=10, fontweight='bold')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='|V|')
    
    plt.suptitle('Vector Field Predictions vs Ground Truth', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('lnn_vector_field_comparison.png', dpi=150, bbox_inches='tight')
    print("   ✓ Vector field comparison saved: lnn_vector_field_comparison.png")
    plt.close()
    
    # Plot 4: Error Field Visualizations
    fig = plt.figure(figsize=(18, 10), dpi=150)
    
    for plot_idx, t_idx in enumerate(time_indices):
        if t_idx >= len(times_np):
            t_idx = len(times_np) - 1
        
        error_mag = np.sqrt(error_ux[t_idx]**2 + error_uy[t_idx]**2)
        
        # u_x error
        ax = plt.subplot(3, len(time_indices), plot_idx + 1)
        im = ax.contourf(error_ux[t_idx], levels=15, cmap='Reds')
        ax.set_title(f'Error u_x (t={times_np[t_idx]:.2f})', fontsize=9)
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax)
        
        # u_y error
        ax = plt.subplot(3, len(time_indices), len(time_indices) + plot_idx + 1)
        im = ax.contourf(error_uy[t_idx], levels=15, cmap='Greens')
        ax.set_title(f'Error u_y (t={times_np[t_idx]:.2f})', fontsize=9)
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax)
        
        # Magnitude error
        ax = plt.subplot(3, len(time_indices), 2*len(time_indices) + plot_idx + 1)
        im = ax.contourf(error_mag, levels=15, cmap='RdYlBu_r')
        ax.set_title(f'Error |V| (t={times_np[t_idx]:.2f})', fontsize=9)
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax)
    
    plt.suptitle('Vector Field Prediction Errors', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('lnn_vector_field_errors.png', dpi=150, bbox_inches='tight')
    print("   ✓ Error field visualization saved: lnn_vector_field_errors.png")
    plt.close()
    
    # Plot 5: Metrics Dashboard
    fig = plt.figure(figsize=(14, 10), dpi=150)
    
    ax1 = plt.subplot(2, 3, 1)
    ax1.semilogy(epochs, losses, linewidth=1.5, color='#1f77b4')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (log scale)')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(2, 3, 2)
    components = ['MAE u_x', 'MAE u_y', 'MAE |V|', 'RMSE u_x', 'RMSE u_y']
    values = [metrics['mae_ux'], metrics['mae_uy'], metrics['mae_magnitude'],
              metrics['rmse_ux'], metrics['rmse_uy']]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars = ax2.bar(range(len(components)), values, color=colors)
    ax2.set_xticks(range(len(components)))
    ax2.set_xticklabels(components, rotation=45, ha='right')
    ax2.set_ylabel('Error Value')
    ax2.set_title('Component-wise Error Metrics')
    ax2.grid(True, alpha=0.3, axis='y')
    
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(times_np, error_ux_mean, 'r-', linewidth=1.5, label='u_x', alpha=0.8)
    ax3.plot(times_np, error_uy_mean, 'g-', linewidth=1.5, label='u_y', alpha=0.8)
    mag_error_mean = np.sqrt(error_ux_mean**2 + error_uy_mean**2)
    ax3.plot(times_np, mag_error_mean, 'b--', linewidth=1.5, label='|V|', alpha=0.8)
    ax3.set_xlabel('Time (t)')
    ax3.set_ylabel('Mean Error')
    ax3.set_title('Error Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4 = plt.subplot(2, 3, 4)
    metrics_list = ['mae_ux', 'mae_uy', 'mae_magnitude', 'mae_angle']
    metrics_values = [metrics['mae_ux'], metrics['mae_uy'], metrics['mae_magnitude'], metrics['mae_angle']]
    # Normalize for visualization
    max_val = max(metrics_values[:3])
    metrics_norm = [metrics_values[0]/max_val, metrics_values[1]/max_val, 
                    metrics_values[2]/max_val, metrics_values[3]/30]  # Angle in degrees
    ax4.barh(metrics_list, metrics_norm, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    ax4.set_xlabel('Normalized Error')
    ax4.set_title('Overall Metrics Summary')
    ax4.grid(True, alpha=0.3, axis='x')
    
    ax5 = plt.subplot(2, 3, 5)
    ax5.scatter(true_all, pred_all, alpha=0.2, s=5, color='#9467bd')
    min_val = min(true_all.min(), pred_all.min())
    max_val = max(true_all.max(), pred_all.max())
    ax5.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect')
    ax5.set_xlabel('Ground Truth Velocity')
    ax5.set_ylabel('Predicted Velocity')
    ax5.set_title('Prediction Accuracy')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    ax6 = plt.subplot(2, 3, 6)
    angle_error_flat = []
    cos_angle = (pred_ux * true_ux + pred_uy * true_uy) / (np.sqrt(pred_ux**2 + pred_uy**2) * np.sqrt(true_ux**2 + true_uy**2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1, 1)
    angle_error_vals = np.arccos(cos_angle) * 180 / np.pi
    ax6.hist(angle_error_vals.flatten(), bins=50, color='#d62728', alpha=0.7, edgecolor='black')
    ax6.axvline(metrics['mae_angle'], color='black', linestyle='--', linewidth=2, label=f"Mean: {metrics['mae_angle']:.2f}°")
    ax6.set_xlabel('Angular Error (degrees)')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Directional Error Distribution')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Vector Field Prediction - Comprehensive Metrics Dashboard', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('lnn_vector_metrics_dashboard.png', dpi=150, bbox_inches='tight')
    print("   ✓ Metrics dashboard saved: lnn_vector_metrics_dashboard.png")
    plt.close()
    
    # Summary
    print("\n[7] Vector Field Prediction Summary:")
    print(f"   Initial Loss: {losses[0]:.6e}")
    print(f"   Final Loss: {losses[-1]:.6e}")
    print(f"   Best Loss: {best_loss:.6e}")
    print(f"   Loss Reduction: {losses[0]/losses[-1]:.2f}x")
    print(f"\n   Component Errors (MAE):")
    print(f"   - u_x: {metrics['mae_ux']:.6e}")
    print(f"   - u_y: {metrics['mae_uy']:.6e}")
    print(f"   - |V|: {metrics['mae_magnitude']:.6e}")
    print(f"\n   Component Errors (RMSE):")
    print(f"   - u_x: {metrics['rmse_ux']:.6e}")
    print(f"   - u_y: {metrics['rmse_uy']:.6e}")
    print(f"   - |V|: {metrics['rmse_magnitude']:.6e}")
    print(f"\n   Directional Error: {metrics['mae_angle']:.4f}°")
    print(f"\n   Maximum Errors:")
    print(f"   - u_x: {metrics['max_error_ux']:.6e}")
    print(f"   - u_y: {metrics['max_error_uy']:.6e}")
    print(f"   - |V|: {metrics['max_error_magnitude']:.6e}")
    
    print("\n" + "=" * 70)
    print("Vector field training completed! Check output PNG files for results.")
    print("=" * 70)


if __name__ == "__main__":
    train_vector_field_lnn()
