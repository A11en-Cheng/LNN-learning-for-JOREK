"""
Laminar-Vortex Flow LNN Training
================================
This script demonstrates LNN training on 2D laminar-vortex flow dynamics
with irregular time sampling (simulating adaptive time-stepping).

Features:
- 2D laminar flow with vortex pair evolution
- Time-varying velocity field with diffusion
- Irregular temporal sampling
- LNN model training with variable dt
- Visualization of training results and predictions
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from ncps.torch import CfC
from ncps.wirings import AutoNCP

# =============================================================================
# PART 1: 2D Laminar-Vortex Flow Physics
# =============================================================================

class LaminarVortexFlow:
    """
    2D laminar flow with vortex pair evolution.
    
    Base flow: Laminar shear layer u_x = U_0 * tanh(y/delta)
    Perturbation: Vortex pair that advects and diffuses over time
    """
    
    def __init__(self, Lx=2*np.pi, Ly=2*np.pi, nx=32, ny=32, 
                 U0=1.0, nu=0.01, vortex_strength=1.0):
        """
        Initialize the flow domain and parameters.
        
        Args:
            Lx, Ly: Domain size
            nx, ny: Grid resolution
            U0: Base velocity amplitude
            nu: Kinematic viscosity (diffusion coefficient)
            vortex_strength: Initial vortex pair strength
        """
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
        
        # Initial vortex positions (pair configuration)
        self.vortex1_x0 = Lx / 4
        self.vortex1_y0 = Ly / 4
        self.vortex2_x0 = 3 * Lx / 4
        self.vortex2_y0 = -Ly / 4
        
        self.vortex_core_radius = 0.2
        
    def laminar_base_flow(self, y):
        """Base laminar shear layer."""
        return self.U0 * np.tanh(y / 0.3)
    
    def vortex_pair_field(self, t):
        """
        Vortex pair velocity field with advection and diffusion.
        
        Returns:
            u_x, u_y: Velocity components on the grid at time t
        """
        # Decay of vortex strength due to viscous diffusion
        decay_factor = np.exp(-self.nu * t / self.vortex_core_radius**2)
        current_strength = self.vortex_strength * decay_factor
        
        # Vortex self-advection (pair moves together)
        # Simplified: vortex pair advects at mean velocity
        advection_velocity = self.U0
        vortex1_x = self.vortex1_x0 + advection_velocity * t
        vortex2_x = self.vortex2_x0 + advection_velocity * t
        
        # Wrap around periodic boundary
        vortex1_x = vortex1_x % self.Lx
        vortex2_x = vortex2_x % self.Lx
        
        # Compute vortex induced velocities (using Gaussian cores)
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
        
        # Laminar shear contribution (only in x-direction)
        u_x_laminar = self.laminar_base_flow(self.Y)
        
        u_x = u_x_laminar + u_x_vortex
        u_y = u_y_vortex
        
        return u_x, u_y
    
    def velocity_at_time(self, t):
        """Get velocity field at time t."""
        return self.vortex_pair_field(t)
    
    def pressure_from_velocity(self, u_x, u_y):
        """Simple pressure field from velocity (for monitoring)."""
        # This is a simplified approach - just for visualization
        dudx = np.gradient(u_x, axis=0)
        dudy = np.gradient(u_x, axis=1)
        dvdx = np.gradient(u_y, axis=0)
        dvdy = np.gradient(u_y, axis=1)
        return dudx + dvdy  # Divergence (should be ~0 for incompressible flow)


# =============================================================================
# PART 2: Data Generation with Irregular Time Sampling
# =============================================================================

def generate_laminar_vortex_dataset(flow, num_snapshots=100, t_max=10.0, 
                                   input_size=8):
    """
    Generate training dataset with irregular time sampling.
    
    Args:
        flow: LaminarVortexFlow instance
        num_snapshots: Number of time snapshots to generate
        t_max: Maximum simulation time
        input_size: Reduced dimension for feature extraction
    
    Returns:
        times: Actual time values [1, num_snapshots]
        dt: Time intervals [1, num_snapshots]
        data_sequences: Input features [batch=1, seq_len, features]
        target_sequences: Ground truth velocity [batch=1, seq_len, output_features]
    """
    
    # Generate irregular time points
    batch_size = 1
    seq_len = num_snapshots
    
    # Random time steps (irregular sampling)
    dt = torch.rand(batch_size, seq_len) * 0.15 + 0.02  # dt in [0.02, 0.17]
    times = torch.cumsum(dt, dim=1)
    times = torch.clamp(times, max=t_max)  # Cap at t_max
    
    print(f"Time span: {times[0, 0]:.4f} to {times[0, -1]:.4f}")
    print(f"Mean dt: {dt.mean():.4f}, Range: [{dt.min():.4f}, {dt.max():.4f}]")
    
    # Pre-compute reference snapshots
    t_refs = np.linspace(0, t_max, num_snapshots)
    u_x_refs = []
    u_y_refs = []
    
    for t in t_refs:
        u_x, u_y = flow.velocity_at_time(t)
        u_x_refs.append(u_x)
        u_y_refs.append(u_y)
    
    # Feature extraction: reduce spatial dimensions using POD-like approach
    # Simple approach: use low-order Fourier modes or spatial sampling
    def extract_features(u_x, u_y, n_features=input_size):
        """Extract features from velocity field."""
        # Method 1: Sample at grid points and flatten
        stride = max(1, flow.nx // int(np.sqrt(n_features//2)))
        u_x_sampled = u_x[::stride, ::stride].flatten()
        u_y_sampled = u_y[::stride, ::stride].flatten()
        features = np.concatenate([u_x_sampled, u_y_sampled])
        # Pad or truncate to desired size
        if len(features) > n_features:
            features = features[:n_features]
        else:
            features = np.pad(features, (0, n_features - len(features)))
        return features
    
    # Generate data sequences
    data_list = []
    target_list = []
    
    for i, t in enumerate(times[0].numpy()):
        # Find nearest reference snapshot
        idx = np.argmin(np.abs(t_refs - t))
        u_x_ref = u_x_refs[idx]
        u_y_ref = u_y_refs[idx]
        
        # Extract features
        features = extract_features(u_x_ref, u_y_ref, input_size)
        data_list.append(features)
        
        # Target: velocity magnitude (or velocity field representation)
        velocity_mag = np.sqrt(u_x_ref**2 + u_y_ref**2)
        target = np.mean(velocity_mag)  # Simple scalar target
        target_list.append(target)
    
    data_sequences = torch.FloatTensor(np.array(data_list)).unsqueeze(0)  # [1, seq_len, features]
    target_sequences = torch.FloatTensor(np.array(target_list)).unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
    
    # Add noise to simulate measurement uncertainty
    noise_level = 0.05
    data_sequences = data_sequences + noise_level * torch.randn_like(data_sequences)
    
    print(f"\nData shapes:")
    print(f"  Inputs: {data_sequences.shape}")
    print(f"  Targets: {target_sequences.shape}")
    
    return times, dt, data_sequences, target_sequences, t_refs, u_x_refs, u_y_refs


# =============================================================================
# PART 3: LNN Model (modified from minimal_lnn_test.py)
# =============================================================================

class LaminarVortexLNN(nn.Module):
    """
    LNN model for laminar-vortex flow prediction.
    """
    def __init__(self, input_size, hidden_size, output_size=1):
        super().__init__()
        self.lnn = CfC(
            input_size=input_size,
            units=hidden_size,
            proj_size=output_size,
            return_sequences=True
        )
    
    def forward(self, x, times=None):
        """
        Forward pass.
        
        Args:
            x: Input features [batch, seq_len, input_size]
            times: Optional time information (not directly used by CfC)
        
        Returns:
            output: Predictions [batch, seq_len, output_size]
        """
        output, _ = self.lnn(x)
        return output


# =============================================================================
# PART 4: Training Loop
# =============================================================================

def train_laminar_vortex_lnn():
    """Main training function."""
    
    print("=" * 70)
    print("2D Laminar-Vortex Flow LNN Training")
    print("=" * 70)
    
    # Create flow
    print("\n[1] Initializing 2D Laminar-Vortex Flow...")
    flow = LaminarVortexFlow(
        Lx=2*np.pi, Ly=2*np.pi, nx=32, ny=32,
        U0=1.0, nu=0.01, vortex_strength=1.0
    )
    
    # Generate dataset
    print("\n[2] Generating training dataset with irregular time sampling...")
    times, dt, data_sequences, target_sequences, t_refs, u_x_refs, u_y_refs = \
        generate_laminar_vortex_dataset(flow, num_snapshots=150, t_max=8.0, input_size=16)
    
    # Model setup
    print("\n[3] Setting up LNN model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using device: {device}")
    
    input_size = data_sequences.shape[-1]
    hidden_size = 32
    output_size = target_sequences.shape[-1]
    
    model = LaminarVortexLNN(input_size, hidden_size, output_size)
    model.to(device)
    
    data_sequences = data_sequences.to(device)
    target_sequences = target_sequences.to(device)
    
    print(f"   Input size: {input_size}")
    print(f"   Hidden size: {hidden_size}")
    print(f"   Output size: {output_size}")
    
    # Training
    print("\n[4] Starting training (3000 epochs)...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.MSELoss()
    
    losses = []
    best_loss = float('inf')
    
    for epoch in range(3000):
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(data_sequences, times)
        
        # Loss computation
        loss = criterion(predictions, target_sequences)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        # Track best model
        if loss.item() < best_loss:
            best_loss = loss.item()
        
        # Progress output
        if (epoch + 1) % 100 == 0:
            print(f"   Epoch {epoch+1:4d}/3000 | Loss: {loss.item():.6e} | Best: {best_loss:.6e}")
    
    print("\n[5] Training completed!")
    print(f"   Final loss: {losses[-1]:.6e}")
    print(f"   Best loss: {best_loss:.6e}")
    
    # =============================================================================
    # PART 5: Evaluation and Visualization
    # =============================================================================
    
    print("\n[6] Generating prediction and evaluation plots...")
    
    with torch.no_grad():
        predictions = model(data_sequences, times)
    
    pred_np = predictions.cpu().numpy()[0, :, 0]
    target_np = target_sequences.cpu().numpy()[0, :, 0]
    times_np = times.cpu().numpy()[0, :]
    dt_np = dt.cpu().numpy()[0, :]
    
    # Plot 1: Training Loss Evolution
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    epochs = np.arange(1, len(losses) + 1)
    ax.semilogy(epochs, losses, linewidth=1.5, color='#1f77b4')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('Training Loss Evolution - Laminar-Vortex Flow LNN', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lnn_loss_evolution.png', dpi=150, bbox_inches='tight')
    print("   ✓ Loss plot saved: lnn_loss_evolution.png")
    plt.close()
    
    # Plot 2: Prediction vs Ground Truth
    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    ax.plot(times_np, target_np, 'k--', linewidth=2, label='Ground Truth (Physics)', alpha=0.7)
    ax.plot(times_np, pred_np, 'r-', linewidth=1.5, label='LNN Prediction', alpha=0.8)
    
    # Add error band
    error = np.abs(pred_np - target_np)
    ax.fill_between(times_np, pred_np - error, pred_np + error, 
                     color='red', alpha=0.1, label='Prediction ± Error')
    
    ax.set_xlabel('Time (t)', fontsize=12)
    ax.set_ylabel('Velocity Magnitude', fontsize=12)
    ax.set_title('LNN Prediction vs Ground Truth - Velocity Evolution', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lnn_prediction_comparison.png', dpi=150, bbox_inches='tight')
    print("   ✓ Prediction plot saved: lnn_prediction_comparison.png")
    plt.close()
    
    # Plot 3: Training Metrics Dashboard
    fig = plt.figure(figsize=(14, 10), dpi=150)
    
    # Loss in log scale
    ax1 = plt.subplot(2, 3, 1)
    ax1.semilogy(epochs, losses, linewidth=1.5, color='#1f77b4')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (log scale)')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    
    # Loss in linear scale (last 500 epochs)
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(epochs[-500:], losses[-500:], linewidth=1, color='#1f77b4')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Loss (Last 500 Epochs)')
    ax2.grid(True, alpha=0.3)
    
    # Prediction error
    ax3 = plt.subplot(2, 3, 3)
    error = np.abs(pred_np - target_np)
    ax3.plot(times_np, error, color='#ff7f0e', linewidth=1.5)
    ax3.fill_between(times_np, 0, error, color='#ff7f0e', alpha=0.2)
    ax3.set_xlabel('Time (t)')
    ax3.set_ylabel('Absolute Error')
    ax3.set_title('Prediction Error vs Time')
    ax3.grid(True, alpha=0.3)
    
    # Time step distribution
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(dt_np, bins=30, color='#2ca02c', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Time Step (dt)')
    ax4.set_ylabel('Frequency')
    ax4.set_title(f'Irregular Time Sampling\n(Mean dt: {dt_np.mean():.4f})')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Prediction scatter plot
    ax5 = plt.subplot(2, 3, 5)
    ax5.scatter(target_np, pred_np, alpha=0.5, s=20, color='#9467bd')
    min_val = min(target_np.min(), pred_np.min())
    max_val = max(target_np.max(), pred_np.max())
    ax5.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Prediction')
    ax5.set_xlabel('Ground Truth')
    ax5.set_ylabel('Prediction')
    ax5.set_title('Prediction Quality')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Mean absolute error evolution
    ax6 = plt.subplot(2, 3, 6)
    window_size = 10
    mae_rolling = np.convolve(error, np.ones(window_size)/window_size, mode='valid')
    ax6.plot(times_np[:len(mae_rolling)], mae_rolling, color='#d62728', linewidth=1.5)
    ax6.set_xlabel('Time (t)')
    ax6.set_ylabel('Rolling Mean Absolute Error')
    ax6.set_title(f'MAE (window={window_size})')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Laminar-Vortex Flow LNN Training Metrics Dashboard', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('lnn_metrics_dashboard.png', dpi=150, bbox_inches='tight')
    print("   ✓ Metrics dashboard saved: lnn_metrics_dashboard.png")
    plt.close()
    
    # Plot 4: Flow Field Snapshots at Different Times
    print("   Generating flow field visualizations...")
    fig = plt.figure(figsize=(16, 10), dpi=150)
    
    time_indices = [0, len(t_refs)//4, len(t_refs)//2, 3*len(t_refs)//4, len(t_refs)-1]
    
    for plot_idx, t_idx in enumerate(time_indices, 1):
        if t_idx >= len(u_x_refs):
            t_idx = len(u_x_refs) - 1
        
        u_x = u_x_refs[t_idx]
        u_y = u_y_refs[t_idx]
        
        # Velocity magnitude
        velocity_mag = np.sqrt(u_x**2 + u_y**2)
        
        ax = plt.subplot(2, 3, plot_idx)
        im = ax.contourf(flow.X, flow.Y, velocity_mag, levels=20, cmap='viridis')
        
        # Quiver plot for velocity direction
        stride = 2
        ax.quiver(flow.X[::stride, ::stride], flow.Y[::stride, ::stride],
                 u_x[::stride, ::stride], u_y[::stride, ::stride],
                 alpha=0.6, scale=15)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f't = {t_refs[t_idx]:.2f}')
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax, label='|V|')
    
    plt.suptitle('Laminar-Vortex Flow Field Evolution', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('lnn_flow_field_snapshots.png', dpi=150, bbox_inches='tight')
    print("   ✓ Flow field snapshots saved: lnn_flow_field_snapshots.png")
    plt.close()
    
    # Summary statistics
    print("\n[7] Training Summary:")
    print(f"   Initial Loss: {losses[0]:.6e}")
    print(f"   Final Loss: {losses[-1]:.6e}")
    print(f"   Best Loss: {best_loss:.6e}")
    print(f"   Loss Reduction: {losses[0]/losses[-1]:.2f}x")
    print(f"   Mean Absolute Error: {error.mean():.6e}")
    print(f"   Max Absolute Error: {error.max():.6e}")
    print(f"   R² Score: {1 - np.var(error)/np.var(target_np):.4f}")
    
    print("\n" + "=" * 70)
    print("Training completed! Check output PNG files for results.")
    print("=" * 70)


if __name__ == "__main__":
    train_laminar_vortex_lnn()
