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

def test_reconstruction_pipeline():
    print("==================================================")
    print("1. Loading HDF5 Data (Simulating a Batch)...")
    parser = JorekDataParser("data/test_data_for_pipeline_EHL-2/jorek000000.h5")
    graph_data_1 = parser.parse_to_graph()
    graph_data_2 = parser.parse_to_graph() # 模拟第二个时间步或样本
    
    # 使用 PyG 的 Batch 将多个图拼接在一起
    batch_data = Batch.from_data_list([graph_data_1, graph_data_2])
    print(f"   Batched Graph Data: {batch_data}")
    
    print("\n2. Initializing Full Pipeline (Encoder -> LNN -> Decoder -> Physics)...")
    encoder = JorekGNNEncoder(input_dim=9, hidden_dim=64, latent_dim=32, num_layers=3)
    processor = JorekLNNProcessor(latent_dim=32, hidden_size=64, num_layers=1)
    decoder = JorekGNNDecoder(latent_dim=32, hidden_dim=64, output_dim=9, num_layers=2)
    physics_layer = ReducedMHDConstraints()
    
    print("\n3. Forward Pass (Full Pipeline)...")
    # 3.1 Perception (GNN)
    latent_features = encoder(batch_data.x, batch_data.edge_index, batch_data.pos)
    
    # 3.2 Cognition (LNN)
    next_latent_features, _ = processor(latent_features, batch_data.batch)
    
    # 3.3 Reconstruction (GNN Decoder)
    # 预测 Model 502 的 9 场残差和破裂概率
    pred_state_vars, p_disrupt = decoder(next_latent_features, batch_data.edge_index, batch_data.batch, batch_data.pos)
    print(f"   Predicted State Vars shape: {pred_state_vars.shape} [Total_nodes, 9]")
    print(f"   Predicted P_disruption shape: {p_disrupt.shape} [Batch_size, 1]")
    
    # 3.4 Physics Constraints (Reduced MHD Ansatz)
    # 提取 F0 常数 (PyG Batch 会将字典中的标量合并为列表)
    F0 = batch_data.norm_factors['F0'][0]
    
    # 计算真实的磁场 B 和速度场 v
    B_vec, v_vec = physics_layer(pred_state_vars, batch_data.pos, batch_data.edge_index, F0)
    print(f"   Computed B Field shape: {B_vec.shape} [Total_nodes, 3]")
    print(f"   Computed v Field shape: {v_vec.shape} [Total_nodes, 3]")
    
    # 验证输出
    assert pred_state_vars.shape[1] == 9, "State vars must have 9 dimensions!"
    assert p_disrupt.shape[0] == 2, "P_disruption must have Batch_size dimensions!"
    assert B_vec.shape[1] == 3 and v_vec.shape[1] == 3, "B and v must be 3D vectors!"
    
    print("\n✅ Reconstruction & Physics Pipeline test passed successfully!")
    print("==================================================")

if __name__ == "__main__":
    test_reconstruction_pipeline()
