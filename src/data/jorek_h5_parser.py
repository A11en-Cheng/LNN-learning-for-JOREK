import h5py
import torch
import numpy as np
from torch_geometric.data import Data
import os
import re
from .model502_schema import (
    BACKGROUND_SOURCE_DATASET_ALIASES,
    BACKGROUND_SOURCE_FIELD_NAMES,
    BACKGROUND_SOURCE_VECTOR_DATASET_ALIASES,
    FIELD_NAMES,
    MARKER_PROJECTION_FIELD_NAMES,
    NUM_BACKGROUND_SOURCE_FIELDS,
    NUM_MARKER_PROJECTION_FIELDS,
    reorder_h5_values,
    validate_background_source_tensor,
    validate_projection_tensor,
)
from .parameter_channel import build_condition_vector


STATE_FRAME_RE = re.compile(r"^jorek(?P<index>\d+)\.h5$")
PROJECTION_FRAME_RE = re.compile(r"^projections(?P<index>\d+)\.h5$")

class JorekDataParser:
    """
    解析 JOREK HDF5 输出文件，提取原生非结构化网格数据。
    严格遵守 'No Rasterization' 规则，直接构建 PyG 图数据。
    """
    def __init__(self, h5_filepath, require_source=False):
        self.filepath = h5_filepath
        self.require_source = bool(require_source)
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Cannot find HDF5 file: {self.filepath}")

    def _build_edge_index(self, contain_node):
        """
        根据 JOREK 的单元-节点连接矩阵 (contain_node 或 vertex) 构建图的 edge_index。
        将同一个有限元内的节点两两连接（全连接子图）。
        """
        edges = set()
        num_elements = contain_node.shape[0]
        
        for i in range(num_elements):
            nodes_in_elem = contain_node[i, :]
            # 过滤掉无效节点（通常用 0 或 -1 填充）
            valid_nodes = [n - 1 for n in nodes_in_elem if n > 0] # JOREK 索引通常从 1 开始
            
            # 在同一个单元内的节点两两相连
            for n1 in valid_nodes:
                for n2 in valid_nodes:
                    if n1 != n2:
                        edges.add((n1, n2))
                        
        if not edges:
            # 如果没有边，返回空的 edge_index
            return torch.empty((2, 0), dtype=torch.long)
            
        edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
        return edge_index

    def _build_edge_index_from_neighbours(self, neighbours):
        """
        根据 JOREK 的单元邻居矩阵 (neighbours) 构建图的 edge_index。
        """
        edges = set()
        num_elements = neighbours.shape[0]
        
        for i in range(num_elements):
            # i 是当前单元的索引 (0-based)
            elem_idx = i + 1 # JOREK 索引通常从 1 开始
            neighbor_elems = neighbours[i, :]
            
            for neighbor in neighbor_elems:
                if neighbor > 0:
                    # 单元之间的连接
                    edges.add((elem_idx, neighbor))
                    edges.add((neighbor, elem_idx))
                    
        if not edges:
            return torch.empty((2, 0), dtype=torch.long)
            
        edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
        return edge_index

    @staticmethod
    def _load_namelist_text(filepath):
        """Read the run's active namelist stored beside the frame files."""
        candidates = ("jorek_namelist_spitest", "jorek_namelist")
        for name in candidates:
            path = os.path.join(os.path.dirname(filepath), name)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8", errors="replace") as stream:
                    return stream.read()
        return None

    @staticmethod
    def _leaf_name(path):
        return path.rsplit('/', 1)[-1].strip().lower().replace('-', '_')

    @staticmethod
    def _extract_nodal_dataset(dataset, num_nodes, dataset_name):
        """Read a scalar nodal field while selecting JOREK's value DoF (zero)."""
        raw = np.asarray(dataset[()])
        if raw.ndim == 1 and raw.shape[0] == num_nodes:
            return raw
        if raw.ndim >= 2 and raw.shape[-1] == num_nodes:
            index = (0,) * (raw.ndim - 1) + (slice(None),)
            return raw[index]
        raise ValueError(
            f"Source dataset '{dataset_name}' must be nodal with final dimension {num_nodes}; "
            f"got shape {raw.shape}."
        )

    def extract_background_sources(self, h5_file, num_nodes):
        """Read Model 502 background controls from an explicitly named HDF5 schema.

        Accepted layouts are either one vector dataset named ``sources`` (or a
        documented alias) with shape ``[3, ..., nodes]``, or three scalar nodal
        datasets named with BACKGROUND_SOURCE_DATASET_ALIASES.  Partial records
        are rejected because filling a missing source with zero would mislabel
        a forced trajectory as unforced.
        """
        datasets = {}

        def collect(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets[name] = obj

        h5_file.visititems(collect)
        by_leaf = {self._leaf_name(path): (path, obj) for path, obj in datasets.items()}

        for vector_name in BACKGROUND_SOURCE_VECTOR_DATASET_ALIASES:
            entry = by_leaf.get(vector_name)
            if entry is None:
                continue
            path, dataset = entry
            raw = np.asarray(dataset[()])
            if raw.ndim >= 2 and raw.shape[0] == NUM_BACKGROUND_SOURCE_FIELDS and raw.shape[-1] == num_nodes:
                index = (slice(None),) + (0,) * (raw.ndim - 2) + (slice(None),)
                source = torch.as_tensor(raw[index], dtype=torch.float32).transpose(0, 1).contiguous()
                return validate_background_source_tensor(source), {
                    name: path for name in BACKGROUND_SOURCE_FIELD_NAMES
                }
            raise ValueError(
                f"Source vector dataset '{path}' must have shape [3, ..., {num_nodes}], got {raw.shape}."
            )

        source_columns = []
        source_paths = {}
        missing = []
        for field_name in BACKGROUND_SOURCE_FIELD_NAMES:
            entry = None
            for alias in BACKGROUND_SOURCE_DATASET_ALIASES[field_name]:
                if alias in by_leaf:
                    entry = by_leaf[alias]
                    break
            if entry is None:
                missing.append(field_name)
                continue
            path, dataset = entry
            source_columns.append(torch.as_tensor(
                self._extract_nodal_dataset(dataset, num_nodes, path), dtype=torch.float32
            ).view(-1, 1))
            source_paths[field_name] = path

        if missing and source_columns:
            raise ValueError(
                f"Partial Model 502 background source record in {self.filepath}; missing {missing}. "
                "Provide all particle_source/heat_i_source/heat_e_source fields or none."
            )
        if missing:
            return None, {}
        return validate_background_source_tensor(torch.cat(source_columns, dim=1)), source_paths

    def _frame_index(self) -> int | None:
        match = STATE_FRAME_RE.fullmatch(os.path.basename(self.filepath))
        return int(match.group("index")) if match else None

    def _find_projection_file(self):
        """Locate the sibling ``projections<index>.h5`` for this frame, if any."""
        index = self._frame_index()
        if index is None:
            return None
        directory = os.path.dirname(self.filepath)
        for name in sorted(os.listdir(directory)):
            match = PROJECTION_FRAME_RE.fullmatch(name)
            if match is not None and int(match.group("index")) == index:
                return os.path.join(directory, name)
        return None

    def _extract_marker_projections(self, projection_path, num_nodes, frame_handle):
        """Read the five marker projections and verify node/time alignment.

        ``projections*.h5`` stores ``t_now`` in physical seconds while the state
        frame stores JOREK units; the conversion factor is the frame's
        ``t_norm`` (verified on real EHL-2 data).  The projection values use the
        use_marker order ``[dE_ion, rad_sink, Z_eff, Z_imp, N_imp]``.
        """
        with h5py.File(projection_path, "r") as handle:
            if "values" not in handle or "t_now" not in handle:
                raise ValueError(
                    f"{os.path.basename(projection_path)} is missing 'values'/'t_now'"
                )
            raw = handle["values"]
            if raw.ndim != 4 or raw.shape[2] != 1:
                raise ValueError(
                    f"{os.path.basename(projection_path)}: expected harmonic dimension 1, "
                    f"got values shape {raw.shape}"
                )
            if raw.shape[0] < NUM_MARKER_PROJECTION_FIELDS:
                raise ValueError(
                    f"{os.path.basename(projection_path)}: expected at least "
                    f"{NUM_MARKER_PROJECTION_FIELDS} channels, got {raw.shape[0]}"
                )
            if raw.shape[-1] != num_nodes:
                raise ValueError(
                    f"{os.path.basename(projection_path)}: node count {raw.shape[-1]} "
                    f"does not match state node count {num_nodes}"
                )
            values = raw[:NUM_MARKER_PROJECTION_FIELDS, 0, 0, :]
            projection_time = float(handle["t_now"][()].item())

        frame_time = float(frame_handle["t_now"][()].item())
        t_norm = float(frame_handle["t_norm"][()].item()) if "t_norm" in frame_handle else 1.0
        frame_seconds = frame_time * t_norm
        # The projection is written within the frame's gather window, so its
        # recorded time can jitter by less than one output step (frame dt is
        # ~4.4e-6 s in the coupled phase).  Accept offsets below 5e-6 s.
        tolerance = 5e-6
        if abs(projection_time - frame_seconds) > tolerance:
            raise ValueError(
                f"{os.path.basename(projection_path)} time {projection_time:.6e}s is not "
                f"aligned with state time {frame_seconds:.6e}s"
            )
        tensor = torch.as_tensor(values, dtype=torch.float32).transpose(0, 1).contiguous()
        return validate_projection_tensor(tensor)

    def parse_to_graph(self) -> Data:
        """
        解析 HDF5 并返回 PyTorch Geometric Data 对象。
        提取 JOREK Model 502 的 9 个核心变量:
        psi, Phi, Z_j, omega, rho, T_i, v_par, rho_n, T_e
        """
        with h5py.File(self.filepath, 'r') as f:
            # 1. 读取节点坐标 (R, Z)
            # JOREK HDF5 中坐标变量名为 'x'，形状为 (2, 4, 1, n_nodes)
            # 维度 0: 0=R, 1=Z
            # 维度 1: 0=节点值, 1=dR, 2=dZ, 3=dRdZ (Bezier 自由度)
            # 我们只提取节点值 (索引 0)
            if 'x' in f:
                # 提取 R 和 Z 的节点值，形状变为 (2, n_nodes)
                x_data = f['x']
                if len(x_data.shape) == 4:
                    coords_raw = x_data[:, 0, 0, :]
                elif len(x_data.shape) == 3:
                    # 假设形状为 (2, 4, n_nodes)
                    coords_raw = x_data[:, 0, :]
                elif len(x_data.shape) == 2:
                    coords_raw = x_data[:, :]
                else:
                    raise ValueError(f"Unexpected shape for 'x': {x_data.shape}")
                # 转置为 (n_nodes, 2)
                coords = torch.tensor(coords_raw, dtype=torch.float32).t().contiguous()
            else:
                raise KeyError("Coordinate array 'x' not found in HDF5.")

            # 2. 读取拓扑连接 (contain_node 或 vertex)
            # 形状为 (n_nodes_per_elem, n_elements)
            if 'contain_node' in f and np.any(f['contain_node'][:]):
                # 转置为 (n_elements, n_nodes_per_elem) 以便 _build_edge_index 处理
                contain_node = f['contain_node'][:].T
                edge_index = self._build_edge_index(contain_node)
            elif 'vertex' in f and np.any(f['vertex'][:]):
                # 如果 contain_node 为空，尝试使用 vertex
                vertex = f['vertex'][:].T
                edge_index = self._build_edge_index(vertex)
            elif 'neighbours' in f and np.any(f['neighbours'][:]):
                # 如果 contain_node 和 vertex 都为空，尝试使用 neighbours
                neighbours = f['neighbours'][:].T
                edge_index = self._build_edge_index_from_neighbours(neighbours)
            else:
                raise KeyError("Topology array 'contain_node', 'vertex' or 'neighbours' not found or empty.")

            # 3. 读取边界拓扑信息
            if 'boundary' in f:
                boundary_raw = np.array(f['boundary'][:]).reshape(-1)
                boundary = torch.tensor(boundary_raw, dtype=torch.long)
            else:
                raise KeyError("Boundary array 'boundary' not found in HDF5.")

            freeboundary_flag = None
            if 'freeboundary' in f:
                fb_val = f['freeboundary'][()]
                if isinstance(fb_val, np.ndarray):
                    fb_val = fb_val.reshape(-1)
                    fb_val = fb_val[0]
                if isinstance(fb_val, (bytes, bytearray)):
                    fb_str = fb_val.decode('utf-8', errors='ignore').strip().upper()
                    freeboundary_flag = fb_str in {"T", "TRUE", "1"}
                elif isinstance(fb_val, str):
                    fb_str = fb_val.strip().upper()
                    freeboundary_flag = fb_str in {"T", "TRUE", "1"}
                else:
                    freeboundary_flag = bool(int(fb_val))

            # 4. 读取物理量 (values)
            # 形状为 (n_var, 4, 1, n_nodes)
            # 根据 JOREK Model 502 源码 (mod_model_settings.f90) 及数据分布分析，变量索引为:
            # 0: Current (j), 1: Potential (Phi), 2: Flux (psi), 3: Vorticity (w)
            # 4: Density (rho), 5: Ions_Temp (Ti), 6: V_parallel (v_par), 7: N_density, 8: Elec_Temp (Te)
            if 'values' in f:
                # 提取所有变量的节点值 (索引 0)，形状变为 (n_var, n_nodes)
                values_data = f['values']
                if len(values_data.shape) == 4:
                    # JOREK writes (n_var, 4 Bezier DoFs, harmonics, nodes).
                    # Phase 1 trains on 2D (ntor=1) runs only: the harmonic
                    # axis must be length one.  Multi-harmonic (ntor11) frames
                    # are excluded by ADR-0004 and must not be silently sliced.
                    if values_data.shape[2] != 1:
                        raise ValueError(
                            f"2D-only contract: 'values' harmonic dimension must be 1, "
                            f"got {values_data.shape[2]}. ntor11/multi-harmonic frames are "
                            "excluded by ADR-0004."
                        )
                    values_raw = values_data[:, 0, 0, :]
                elif len(values_data.shape) == 3:
                    # 假设形状为 (n_var, 4, n_nodes)
                    values_raw = values_data[:, 0, :]
                elif len(values_data.shape) == 2:
                    values_raw = values_data[:, :]
                else:
                    raise ValueError(f"Unexpected shape for 'values': {values_data.shape}")

                if values_raw.shape[0] < 9:
                    raise ValueError(
                        f"'values' has {values_raw.shape[0]} channels, but at least 9 are required for Model 502."
                    )

                x = reorder_h5_values(torch.as_tensor(values_raw, dtype=torch.float32))
            else:
                raise KeyError("Physical values array 'values' not found.")

            # 4.1 零维参数通道：h5 标量 -> namelist -> JOREK 默认值，带逐项 provenance
            namelist_text = self._load_namelist_text(self.filepath)
            condition_data = build_condition_vector(f, namelist_text=namelist_text)
            condition = torch.tensor(condition_data["vector"], dtype=torch.float32)
            condition_names = tuple(condition_data["names"])
            provenance_codes = {"h5": 0, "namelist": 1, "default": 2, "derived": 3}
            condition_provenance = torch.tensor(
                [provenance_codes[condition_data["provenance"][name]] for name in condition_names],
                dtype=torch.long,
            )
            condition_mask = condition_provenance != provenance_codes["default"]

            background_source, background_paths = self.extract_background_sources(f, x.shape[0])

            # Primary source-control channel: marker projections aligned to this frame.
            projection_path = self._find_projection_file()
            marker_source = None
            if projection_path is not None:
                marker_source = self._extract_marker_projections(projection_path, x.shape[0], f)
            if marker_source is None and self.require_source:
                raise ValueError(
                    f"{self.filepath} has no marker projection file "
                    f"(projections<index>.h5 with the same frame index); do not use this "
                    "frame for forced training."
                )

            # 5. 读取时间戳 (如果存在)
            if 't_now' in f:
                t_val = f['t_now'][()]
                if isinstance(t_val, np.ndarray):
                    if t_val.size == 1:
                        t_val = t_val.item()
                    else:
                        t_val = t_val[0]
                time_val = torch.tensor([t_val], dtype=torch.float32)
            else:
                time_val = torch.tensor([0.0], dtype=torch.float32)

            # Physical time conversion: seconds = t_now (JOREK units) * t_norm.
            if 't_norm' in f:
                t_norm_val = torch.tensor([float(f['t_norm'][()].item())], dtype=torch.float32)
            else:
                t_norm_val = torch.tensor([1.0], dtype=torch.float32)

            # 6. 读取归一化常数 (Normalization factors)
            # 这些常数对于物理约束和损失函数的计算至关重要
            norm_factors = {}
            if 'sqrt_mu0_rho0' in f:
                norm_factors['sqrt_mu0_rho0'] = float(f['sqrt_mu0_rho0'][0])
            if 'sqrt_mu0_over_rho0' in f:
                norm_factors['sqrt_mu0_over_rho0'] = float(f['sqrt_mu0_over_rho0'][0])
            if 'F0' in f:
                norm_factors['F0'] = float(f['F0'][0])

            # 7. 提取用于精确可视化的三角形拓扑 (如果存在)
            # JOREK 的 vertex 数组通常包含每个四边形单元的 4 个角点
            # 我们将其拆分为两个三角形以便 matplotlib 渲染
            triangles = None
            if 'vertex' in f:
                vertex = f['vertex'][:] # [4, n_elements]
                # 过滤掉无效的单元 (包含 0 或负数索引)
                valid_mask = (vertex > 0).all(axis=0)
                valid_vertex = vertex[:, valid_mask] - 1 # 转换为 0-based 索引
                
                # 将每个四边形 (0,1,2,3) 拆分为两个三角形 (0,1,2) 和 (0,2,3)
                tri1 = valid_vertex[[0, 1, 2], :].T
                tri2 = valid_vertex[[0, 2, 3], :].T
                triangles = torch.tensor(np.vstack([tri1, tri2]), dtype=torch.long)

            # 8. 计算 edge_attr (几何特征: delta_R, delta_Z, distance)
            if edge_index.numel() > 0:
                row, col = edge_index
                pos_i = coords[row]
                pos_j = coords[col]
                delta_pos = pos_j - pos_i  # [num_edges, 2] (delta_R, delta_Z)
                distance = torch.norm(delta_pos, p=2, dim=-1, keepdim=True)  # [num_edges, 1]
                edge_attr = torch.cat([delta_pos, distance], dim=-1)  # [num_edges, 3]
            else:
                edge_attr = torch.empty((0, 3), dtype=torch.float32)

            # 构建 PyG Data 对象
            graph_data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                pos=coords,
                time=time_val,
                t_norm=t_norm_val,
                norm_factors=norm_factors,
                triangles=triangles,
                condition=condition,
                condition_names=condition_names,
                condition_provenance=condition_provenance,
                condition_mask=condition_mask,
                source_available=torch.tensor(marker_source is not None, dtype=torch.bool),
                background_available=torch.tensor(
                    background_source is not None, dtype=torch.bool
                ),
            )
            if marker_source is not None:
                graph_data.source = marker_source
                graph_data.source_fields = MARKER_PROJECTION_FIELD_NAMES
            if background_source is not None:
                graph_data.background_source = background_source
            if boundary.shape[0] != graph_data.num_nodes:
                raise ValueError(
                    f"Boundary array length {boundary.shape[0]} does not match node count {graph_data.num_nodes}."
                )
            graph_data.boundary = boundary
            if freeboundary_flag is not None:
                graph_data.freeboundary = torch.tensor(int(freeboundary_flag), dtype=torch.int8)
            
            # 防御性编程：验证张量形状
            assert graph_data.x.shape[1] == len(FIELD_NAMES), f"Expected {len(FIELD_NAMES)} features, got {graph_data.x.shape[1]}"
            assert graph_data.pos.shape[0] == graph_data.x.shape[0], "Mismatch between coordinates and features"
            
            return graph_data

if __name__ == "__main__":
    # 测试代码
    parser = JorekDataParser("data/EHL-2/scan_1d22_ang12_massloss_neon/jorek002710.h5")
    graph = parser.parse_to_graph()
    print(graph)
    pass
