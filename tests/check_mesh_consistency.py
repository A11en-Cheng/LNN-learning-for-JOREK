import os
import h5py
import numpy as np
from collections import Counter

def scan_h5_dir(directory):
    files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.h5')]
    mesh_shapes = []
    node_counts = []
    value_shapes = []
    failed_files = []
    for fpath in files:
        try:
            with h5py.File(fpath, 'r') as f:
                if 'x' in f:
                    x = f['x']
                    if len(x.shape) == 4:
                        n_nodes = x.shape[-1]
                    elif len(x.shape) == 3:
                        n_nodes = x.shape[-1]
                    elif len(x.shape) == 2:
                        n_nodes = x.shape[-1]
                    else:
                        n_nodes = -1
                    mesh_shapes.append(x.shape)
                    node_counts.append(n_nodes)
                else:
                    mesh_shapes.append('missing')
                    node_counts.append(-1)
                if 'values' in f:
                    value_shapes.append(f['values'].shape)
                else:
                    value_shapes.append('missing')
        except Exception as e:
            failed_files.append((fpath, str(e)))
    return files, mesh_shapes, node_counts, value_shapes, failed_files

def report_consistency(files, mesh_shapes, node_counts, value_shapes, failed_files):
    print(f"Total files scanned: {len(files)}")
    print(f"Unique mesh shapes: {Counter(mesh_shapes)}")
    print(f"Unique node counts: {Counter(node_counts)}")
    print(f"Unique value shapes: {Counter(value_shapes)}")
    if failed_files:
        print(f"\nFailed to read {len(failed_files)} files:")
        for f, err in failed_files:
            print(f"  {f}: {err}")
    # 详细列出不一致的文件
    most_common_nodes = Counter(node_counts).most_common(1)[0][0] if node_counts else None
    print("\nFiles with uncommon node counts:")
    for f, n in zip(files, node_counts):
        if n != most_common_nodes:
            print(f"  {f}  nodes={n}")
    print("\nFiles with missing or malformed 'x' or 'values':")
    for f, ms, vs in zip(files, mesh_shapes, value_shapes):
        if ms == 'missing' or vs == 'missing':
            print(f"  {f}  mesh_shape={ms}  value_shape={vs}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Check mesh/node/value consistency in a JOREK HDF5 dataset directory.')
    parser.add_argument('directory', type=str, help='Path to directory containing .h5 files')
    args = parser.parse_args()
    files, mesh_shapes, node_counts, value_shapes, failed_files = scan_h5_dir(args.directory)
    report_consistency(files, mesh_shapes, node_counts, value_shapes, failed_files)

if __name__ == "__main__":
    main()
