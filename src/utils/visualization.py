import matplotlib.pyplot as plt
import matplotlib.tri as tri
import torch
import os

def plot_jorek_field(pos, values, triangles=None, title="JOREK Field", save_path=None, cmap='jet'):
    """
    可视化 JOREK 非结构化网格上的标量场。
    严格遵守 'No Rasterization' 规则，直接使用节点坐标进行三角剖分渲染。
    
    Args:
        pos: 节点坐标 (R, Z) [Num_nodes, 2]
        values: 标量场值 [Num_nodes]
        triangles: 预定义的三角形拓扑 [Num_triangles, 3] (可选，用于精确渲染凹边界)
        title: 图表标题
        save_path: 保存路径 (如果为 None，则直接显示)
        cmap: 颜色映射
    """
    # 确保数据在 CPU 上并转换为 numpy 数组
    if isinstance(pos, torch.Tensor):
        pos = pos.detach().cpu().numpy()
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    if isinstance(triangles, torch.Tensor):
        triangles = triangles.detach().cpu().numpy()
        
    # 提取 R 和 Z 坐标
    R = pos[:, 0]
    Z = pos[:, 1]
    
    # 确保 values 是一维数组
    if values.ndim > 1:
        values = values.squeeze()
        
    # 创建三角剖分
    if triangles is not None:
        # 使用 JOREK 原生的拓扑结构，避免在凹边界外生成错误的三角形
        triang = tri.Triangulation(R, Z, triangles)
    else:
        # 回退方案：Delaunay 三角剖分 (可能会在凹边界外生成错误的三角形)
        print("Warning: No explicit triangles provided. Using Delaunay triangulation which may fill concave regions.")
        triang = tri.Triangulation(R, Z)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(8, 10))
    
    # 绘制等高线填充图
    # levels=50 提供平滑的颜色过渡
    contour = ax.tricontourf(triang, values, levels=50, cmap=cmap)
    
    # 添加颜色条
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('Value')
    
    # 设置标签和标题
    ax.set_xlabel('R [m]')
    ax.set_ylabel('Z [m]')
    ax.set_title(title)
    
    # 保持 R 和 Z 的比例一致 (等离子体截面通常需要等比例显示)
    ax.axis('equal')
    
    # 保存或显示
    if save_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
        plt.close(fig)
    else:
        plt.show()
