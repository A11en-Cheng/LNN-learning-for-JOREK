import sys
import os
import torch
from torch_geometric.data import Batch

# 确保可以导入 src 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.jorek_h5_parser import JorekDataParser
from src.models.gnn_encoder import JorekGNNEncoder
from src.models.lnn_processor import JorekLNNProcessor
from src.models.gnn_decoder import JorekGNNDecoder
from src.physics.hard_constraints import ReducedMHDConstraints
from src.models.loss import JorekLoss
from src.training.trainer import JorekTrainer
from src.utils.visualization import plot_jorek_field

def test_training_and_visualization():
    print("==================================================")
    print("1. Loading HDF5 Data (Simulating t and t+1)...")
    # 假设 jorek000000.h5 是 t 时刻，jorek000010.h5 是 t+1 时刻
    parser_t0 = JorekDataParser("data/test_data_for_pipeline_EHL-2/jorek000000.h5")
    parser_t1 = JorekDataParser("data/test_data_for_pipeline_EHL-2/jorek000010.h5")
    
    graph_t0 = parser_t0.parse_to_graph()
    graph_t1 = parser_t1.parse_to_graph()
    
    # 为了测试，我们创建一个 Batch Size = 1 的数据
    batch_data = Batch.from_data_list([graph_t0])
    
    # 目标状态是 t+1 时刻的特征
    target_states = graph_t1.x
    
    # 模拟一个破裂标签 (假设这个样本没有破裂)
    target_p_disrupt = torch.tensor([[0.0]], dtype=torch.float32)
    
    print("\n2. Initializing Models, Loss, and Trainer...")
    encoder = JorekGNNEncoder(input_dim=5, hidden_dim=64, latent_dim=32, num_layers=3)
    processor = JorekLNNProcessor(latent_dim=32, hidden_size=64, num_layers=1)
    decoder = JorekGNNDecoder(latent_dim=32, hidden_dim=64, output_dim=5, num_layers=2)
    physics_layer = ReducedMHDConstraints()
    
    criterion = JorekLoss(lambda_reg=1.0, lambda_cls=0.1)
    trainer = JorekTrainer(encoder, processor, decoder, physics_layer, criterion, lr=1e-3, clip_value=1.0)
    
    print("\n3. Running Training Loop (5 Epochs)...")
    for epoch in range(5):
        total_loss, reg_loss, cls_loss = trainer.train_step(batch_data, target_states, target_p_disrupt)
        print(f"   Epoch {epoch+1}/5 | Total Loss: {total_loss:.4f} | Reg Loss: {reg_loss:.4f} | Cls Loss: {cls_loss:.4f}")
        
    print("\n4. Testing Visualization (No Rasterization)...")
    # 提取 t0 时刻的极向磁通 psi (索引 0)
    psi_t0 = graph_t0.x[:, 0]
    time_t0 = graph_t0.time.item()
    
    # 绘制并保存图像
    output_dir = "output/visualizations"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "psi_t0.png")
    
    plot_jorek_field(
        pos=graph_t0.pos, 
        values=psi_t0, 
        triangles=getattr(graph_t0, 'triangles', None), # CRITICAL: Use native JOREK topology
        title=f"Poloidal Flux ($\\psi$) at t={time_t0:.3f}", 
        save_path=save_path
    )
    
    print("\n✅ Training Loop & Visualization test passed successfully!")
    print("==================================================")

if __name__ == "__main__":
    test_training_and_visualization()
