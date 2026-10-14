import sys
import os
import torch
import pytest

# 确保可以导入 src 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.jorek_h5_parser import JorekDataParser
from src.models.gnn_encoder import JorekGNNEncoder

def test_perception_pipeline():
    if not os.path.isdir("data/EHL-2/scan_1d22_ang12_massloss_neon"):
        pytest.skip("real EHL-2 data not present")
    print("==================================================")
    print("1. Loading HDF5 Data (No Rasterization)...")
    parser = JorekDataParser("data/EHL-2/scan_1d22_ang12_massloss_neon/jorek002710.h5")
    graph_data = parser.parse_to_graph()
    print(f"   Graph Data: {graph_data}")
    
    print("\n2. Initializing GNN Encoder (Perception)...")
    encoder = JorekGNNEncoder(input_dim=9, hidden_dim=64, latent_dim=32, num_layers=3)
    print(encoder)
    
    print("\n3. Forward Pass (Extracting Topological Features)...")
    # 传入节点特征和边索引
    latent_features = encoder(graph_data.x, graph_data.edge_index, graph_data.pos)
    
    print(f"\n   Input shape:  {graph_data.x.shape} [Total_nodes, Input_Feat]")
    print(f"   Output shape: {latent_features.shape} [Total_nodes, Latent_Feat]")
    
    # 验证输出
    assert latent_features.shape[0] == graph_data.x.shape[0], "Node count mismatch!"
    assert latent_features.shape[1] == 32, "Latent dimension mismatch!"
    print("\n✅ Pipeline test passed successfully!")
    print("==================================================")

if __name__ == "__main__":
    test_perception_pipeline()
