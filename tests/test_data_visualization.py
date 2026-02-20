import sys
import os
import torch
import matplotlib.pyplot as plt

# 确保可以导入 src 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.jorek_h5_parser import JorekDataParser
from src.utils.visualization import plot_jorek_field

def visualize_h5_snapshot(h5_filepath, output_dir="output/visualizations"):
    """
    读取单个 HDF5 文件并可视化其包含的所有 5 个核心物理场。
    用于验证数据解析器是否正确提取了向量场和标量场。
    """
    print(f"Loading data from: {h5_filepath}")
    parser = JorekDataParser(h5_filepath)
    graph = parser.parse_to_graph()
    
    time_val = graph.time.item()
    print(f"Snapshot Time: t = {time_val:.4f}")
    print(f"Number of nodes: {graph.x.shape[0]}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 核心变量的名称和索引映射 (根据 jorek_h5_parser.py 中的定义)
    # Parser 中构建 x 的顺序: x = torch.cat([psi, Phi, rho, T, v_par], dim=1)
    # 对应的索引: 0: psi, 1: Phi, 2: rho, 3: T, 4: v_par
    variables = {
        0: ("Poloidal Flux ($\\psi$)", "psi"),
        1: ("Stream Function ($\\Phi$)", "Phi"),
        2: ("Density ($\\rho$)", "rho"),
        3: ("Temperature ($T$)", "T"),
        4: ("Parallel Velocity ($v_{\\parallel}$)", "v_par")
    }
    
    # 提取文件名作为保存前缀
    base_name = os.path.basename(h5_filepath).replace('.h5', '')
    
    for idx, (title, short_name) in variables.items():
        print(f"Rendering {short_name}...")
        values = graph.x[:, idx]
        
        save_path = os.path.join(output_dir, f"{base_name}_{short_name}.png")
        
        # 为不同的物理量选择合适的 colormap
        cmap = 'jet'
        if short_name in ['Phi', 'v_par']:
            cmap = 'RdBu_r' # 速度/流函数通常有正有负，使用发散型 colormap
            
            # 由于 Phi 值非常小，可能需要更敏感的显示范围
            # 但 matplotlib 会自动缩放，这里暂时保持自动
            
        plot_jorek_field(
            pos=graph.pos,
            values=values,
            triangles=getattr(graph, 'triangles', None), # 传入原生拓扑以避免凹边界填充错误
            title=f"{title} at t={time_val:.4f}",
            save_path=save_path,
            cmap=cmap
        )
        
    print(f"All visualizations saved to {output_dir}")

if __name__ == "__main__":
    # 测试可视化 jorek000010.h5
    test_file = "data/test_data_for_pipeline_EHL-2/jorek000010.h5"
    if os.path.exists(test_file):
        visualize_h5_snapshot(test_file)
    else:
        print(f"Error: Test file {test_file} not found.")
