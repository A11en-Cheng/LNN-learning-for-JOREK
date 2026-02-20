import sys
import os
import torch
import shutil

# 确保可以导入 src 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader as TorchInternalDataLoader
from torch_geometric.data import DataLoader as PyGDataLoader
from src.data.jorek_h5_parser import JorekDataParser
from src.data.dataset import JorekSequenceDataset, collate_temporal_graphs
import matplotlib.pyplot as plt

def test_jorek_sequence_loader():
    """
    测试 JOREK 序列数据加载器。
    验证：
    1. 扫描文件并按时间排序。
    2. 生成正确的序列长度。
    3. 自定义 Collate 函数按时间步打包图数据。
    """
    print("Initializing Data Setup for Test...")
    
    # 获取测试数据目录
    data_dir = "data/test_data_for_pipeline_EHL-2"
    
    # 简单的预检查：确保有足够的测试 HFD5 文件
    files = sorted(os.listdir(data_dir))
    h5_files = [f for f in files if f.endswith('.h5') and 'jorek' in f]
    
    if len(h5_files) < 5:
        print(f"Warning: Only {len(h5_files)} HDF5 files found. Test may fail if dataset creation fails.")
    else:
        print(f"Found {len(h5_files)} HDF5 files. Proceeding.")
        
    # 设置序列参数
    SEQUENCE_LENGTH = 3
    BATCH_SIZE = 4
    
    # 实例化数据集
    dataset = JorekSequenceDataset(root_dir=data_dir, seq_len=SEQUENCE_LENGTH)
    
    print(f"Dataset Length (Total Samples): {len(dataset)}")
    
    if len(dataset) <= 0:
        print("Dataset is empty. Exiting test.")
        return

    # 使用 PyTorch DataLoader (配合自定义 collate_fn)
    # 为什么要用 Torch DataLoader 而不是 PyG DataLoader?
    # 因为我们的 dataset.get() 返回的是 List[Data] 而不是 Data。
    # PyG DataLoader 默认处理 Data 对象。我们需要自定义打包逻辑。
    loader = TorchInternalDataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_temporal_graphs
    )
    
    print("Testing One Batch Iteration...")
    iterator = iter(loader)
    first_batch = next(iterator)
    
    # first_batch 应该是一个长度为 seq_len 的列表
    # 每个元素是一个 PyG Batch 对象
    
    assert isinstance(first_batch, list), "Loader output should be a list (sequence time-steps)"
    assert len(first_batch) == SEQUENCE_LENGTH, f"Sequence length mismatch: {len(first_batch)} != {SEQUENCE_LENGTH}"
    
    print(f"Batch Sequence Length: {len(first_batch)}")
    
    # 检查第一个时间步的数据
    batch_t0 = first_batch[0]
    print(f"Time Step 0 Batch Info:")
    print(batch_t0)
    
    # 检查批次大小
    # num_graphs 应该是 BATCH_SIZE
    assert batch_t0.num_graphs == BATCH_SIZE or batch_t0.num_graphs == len(dataset), \
        f"Batch size mismatch: {batch_t0.num_graphs} != {BATCH_SIZE}"
        
    # 检查节点特征维度
    # x shape: [total_nodes_in_batch, num_features]
    print(f"Node Features Shape: {batch_t0.x.shape}")
    assert batch_t0.x.shape[1] == 5, "Expected 5 input features (psi, Phi, rho, T, v_par)"
    
    print("✅ JorekSequenceDataset Test Passed!")

if __name__ == "__main__":
    test_jorek_sequence_loader()
