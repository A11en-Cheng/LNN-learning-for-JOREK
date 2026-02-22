import sys
import os
import torch
from torch_geometric.data import Batch

# 确保可以导入 src 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.jorek_h5_parser import JorekDataParser
from src.models.gnn_encoder import JorekGNNEncoder
from src.models.lnn_processor import JorekLNNProcessor

def test_cognition_pipeline():
    print("==================================================")
    print("1. Loading HDF5 Data (Simulating a Batch)...")
    parser = JorekDataParser("data/test_data_for_pipeline_EHL-2/jorek000000.h5")
    graph_data_1 = parser.parse_to_graph()
    graph_data_2 = parser.parse_to_graph() # 模拟第二个时间步或样本
    
    # 使用 PyG 的 Batch 将多个图拼接在一起
    batch_data = Batch.from_data_list([graph_data_1, graph_data_2])
    print(f"   Batched Graph Data: {batch_data}")
    
    print("\n2. Initializing GNN Encoder & LNN Processor...")
    encoder = JorekGNNEncoder(input_dim=5, hidden_dim=64, latent_dim=32, num_layers=3)
    processor = JorekLNNProcessor(latent_dim=32, hidden_size=64, num_layers=1)
    
    print("\n3. Forward Pass (Perception -> Cognition)...")
    # 3.1 Perception (GNN)
    latent_features = encoder(batch_data.x, batch_data.edge_index, batch_data.pos)
    print(f"   GNN Output shape: {latent_features.shape} [Total_nodes, Latent_Feat]")
    
    # 3.2 Cognition (LNN)
    # 传入 batch 索引以进行 Graph-to-Sequence 转换
    # LNN Forward Pass now returns (output, hidden_state)
    next_latent_features, _ = processor(latent_features, batch_data.batch)
    print(f"   LNN Output shape: {next_latent_features.shape} [Total_nodes, Latent_Feat]")
    
    # 验证输出
    assert next_latent_features.shape == latent_features.shape, "Shape mismatch after LNN processing!"
    print("\n✅ Cognition Pipeline test passed successfully!")
    print("==================================================")

if __name__ == "__main__":
    test_cognition_pipeline()
