import os
import glob
import re
import h5py
import torch
import numpy as np
from torch_geometric.data import Dataset
from typing import List, Optional, Callable
from .jorek_h5_parser import JorekDataParser

class JorekSequenceDataset(Dataset):
    """
    JOREK 时间序列数据集加载器。
    
    功能：
    1. 扫描指定目录下的 JOREK HDF5 文件。
    2. 按文件名中的数字索引排序。
    3. 生成滑动窗口序列 (Slide Window Sequences)。
    4. 按需加载 (Lazy Loading) 并支持处理后的数据缓存。
    
    返回：
    getitem(idx) 返回一个列表 [Data_t0, Data_t1, ..., Data_tn]，长度为 seq_len。
    """
    def __init__(self, 
                 root_dir: str, 
                 seq_len: int = 1, 
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 file_pattern: str = "jorek*.h5",
                 freeboundary: bool = True,
                 require_source: bool = False):
        """
        :param root_dir: 包含 .h5 文件的根目录
        :param seq_len: 每个样本的时间序列长度
        :param transform: 动态变换函数
        :param pre_transform: 预处理函数 (用于 process 阶段)
        :param file_pattern: 匹配文件名的模式
        """
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.file_pattern = file_pattern
        self.sorted_files = self._scan_and_sort_files()
        self.freeboundary = bool(freeboundary)
        self.require_source = bool(require_source)
        
        # 验证文件数量
        if len(self.sorted_files) < seq_len:
            raise ValueError(f"Found only {len(self.sorted_files)} files, but seq_len is {seq_len}. Need at least {seq_len} files.")
            
        super().__init__(root_dir, transform, pre_transform)

    @property
    def raw_file_names(self) -> List[str]:
        return [os.path.basename(f) for f in self.sorted_files]

    @property
    def processed_file_names(self) -> List[str]:
        # 我们使用一个简单的索引机制：data_{index}.pt
        # 注意：这里的数据量可能很大，简单的 disk-cache 可能会占用大量空间。
        # 如果空间有限，建议跳过 process 阶段，直接在 get 中读取 h5。
        # 为了演示和性能平衡，这里我们暂时不强制预处理所有数据到 .pt，
        # 而是采用 get 时读取 HDF5 的策略 (Memory/IO Trade-off)。
        return [] 

    def _scan_and_sort_files(self) -> List[str]:
        """扫描并按数字顺序排序文件"""
        search_path = os.path.join(self.root_dir, self.file_pattern)
        files = glob.glob(search_path)
        
        # 提取文件名中的数字进行排序 (例如 jorek000100.h5 -> 100)
        def extract_number(filepath):
            match = re.search(r'(\d+)', os.path.basename(filepath))
            return int(match.group(1)) if match else 0
            
        return sorted(files, key=extract_number)

    def len(self):
        # 样本数量 = 总文件数 - 序列长度 + 1
        # 例如：5个文件 [0,1,2,3,4], seq_len=3
        # 序列: [0,1,2], [1,2,3], [2,3,4] -> 3个样本
        return len(self.sorted_files) - self.seq_len + 1

    def get(self, idx):
        """
        获取一个时间序列样本。
        
        :param idx: 序列的起始索引
        :return: List[Data] 包含 seq_len 个图数据对象
        """
        sequence_files = self.sorted_files[idx : idx + self.seq_len]
        
        data_sequence = []
        for filepath in sequence_files:
            # 实时解析 HDF5 (如果 IO 成为瓶颈，可以在这里添加内存缓存 LRU Cache)
            parser = JorekDataParser(filepath, require_source=self.require_source)
            data = parser.parse_to_graph()
            if not hasattr(data, 'boundary'):
                raise AttributeError(f"Graph parsed from {os.path.basename(filepath)} is missing boundary annotations.")

            # 处理 freeboundary 标记，确保以标量张量形式存在
            if hasattr(data, 'freeboundary') and data.freeboundary is not None:
                fb_value = data.freeboundary
                if isinstance(fb_value, torch.Tensor):
                    fb_scalar = int(fb_value.view(-1)[0].item())
                else:
                    fb_scalar = int(fb_value)
            else:
                fb_scalar = int(self.freeboundary)
            data.freeboundary = torch.tensor(fb_scalar, dtype=torch.int8)

            # === 加载每个时间步的磁通归一化常数: psi_axis 与 psi_bnd ===
            # 优先从 HDF5 标量读取；若缺失则退化为使用场最小/最大值近似。
            psi_axis_val = None
            psi_bnd_val = None
            try:
                with h5py.File(filepath, 'r') as f:
                    if 'psi_axis' in f:
                        raw_axis = f['psi_axis'][()]
                        if isinstance(raw_axis, np.ndarray):
                            raw_axis = raw_axis.reshape(-1)[0]
                        psi_axis_val = float(raw_axis)
                    if 'psi_bnd' in f:
                        raw_bnd = f['psi_bnd'][()]
                        if isinstance(raw_bnd, np.ndarray):
                            raw_bnd = raw_bnd.reshape(-1)[0]
                        psi_bnd_val = float(raw_bnd)
            except OSError:
                # 如果 HDF5 读取异常，则在后面用场本身近似
                psi_axis_val = None
                psi_bnd_val = None

            # 如果文件中未提供，则用当前图上的 psi 场最小/最大值做保底近似
            if data.x is None or data.x.shape[1] < 1:
                raise ValueError("Parsed JOREK graph is missing psi channel in x.")
            psi_field = data.x[:, 0]
            if psi_axis_val is None:
                psi_axis_val = float(psi_field.min().item())
            if psi_bnd_val is None:
                psi_bnd_val = float(psi_field.max().item())

            # 将归一化常数挂到 Data 对象上，供损失函数与物理算子使用
            data.psi_axis = torch.tensor(psi_axis_val, dtype=torch.float32)
            data.psi_bnd = torch.tensor(psi_bnd_val, dtype=torch.float32)

            data_sequence.append(data)
            
        return data_sequence

def collate_temporal_graphs(batch_list):
    """
    自定义 Collate 函数，用于 DataLoader。
    处理 List[List[Data]] -> List[Batch] 或 适合 LNN 的结构。
    
    Args:
        batch_list: DataLoader 产生的一个 batch，包含 batch_size 个样本。
                    每个样本是一个 List[Data] (长度为 seq_len)。
                    结构: [ [t0_s0, t1_s0], [t0_s1, t1_s1], ... ]
    
    Returns:
        spatial_batches: 一个长度为 seq_len 的列表。
                         列表中的每个元素是一个 PyG Batch 对象，
                         包含该时刻所有样本的图组合在一起。
                         
    Example:
        LNN Trainer 可以在每个时间步 t，取 spatial_batches[t] 输入到 GNN -> LNN Cell。
    """
    from torch_geometric.data import Batch
    
    seq_len = len(batch_list[0])
    batch_size = len(batch_list)
    
    spatial_batches = []
    
    # 转置: Time-major 
    # 从 [Sample, Time] -> [Time, Sample]
    for t in range(seq_len):
        graphs_at_t = [sample[t] for sample in batch_list]
        
        # 将该时刻的所有样本图合并为一个大图 (Standard PyG Batching)
        # 这样 GNN 可以并行处理所有样本的空间特征
        batch_at_t = Batch.from_data_list(graphs_at_t)
        spatial_batches.append(batch_at_t)
        
    return spatial_batches
