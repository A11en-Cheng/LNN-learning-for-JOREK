import os
import h5py
import numpy as np

def inspect_h5_file(filepath):
    print(f"Inspecting: {filepath}")
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None

    info = {}
    try:
        with h5py.File(filepath, 'r') as f:
            info['keys'] = list(f.keys())
            
            if 'x' in f:
                info['x_shape'] = f['x'].shape
            else:
                info['x_shape'] = "Missing"
                
            if 'values' in f:
                info['values_shape'] = f['values'].shape
            else:
                info['values_shape'] = "Missing"
                
            # Check topology arrays
            topo_keys = ['contain_node', 'vertex', 'neighbours']
            for k in topo_keys:
                if k in f:
                    info[f'{k}_shape'] = f[k].shape
                else:
                    info[f'{k}_shape'] = "Missing"
                    
            # Check time
            if 't_now' in f:
                t_val = f['t_now'][()]
                info['t_now'] = t_val.item() if isinstance(t_val, np.ndarray) and t_val.size == 1 else (t_val[0] if isinstance(t_val, np.ndarray) else t_val)
            else:
                info['t_now'] = "Missing"
                
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        
    return info

def compare_datasets(file1, file2):
    info1 = inspect_h5_file(file1)
    info2 = inspect_h5_file(file2)
    
    if not info1 or not info2:
        return
        
    print("\n--- Comparison ---")
    print(f"{'Property':<20} | {'Dataset 1 (EHL-2)':<30} | {'Dataset 2 (ITER)':<30}")
    print("-" * 85)
    
    keys_to_compare = ['x_shape', 'values_shape', 'contain_node_shape', 'vertex_shape', 'neighbours_shape', 't_now']
    
    for k in keys_to_compare:
        val1 = str(info1.get(k, 'N/A'))
        val2 = str(info2.get(k, 'N/A'))
        
        # Highlight differences
        marker = "*" if val1 != val2 else " "
        print(f"{marker} {k:<18} | {val1:<30} | {val2:<30}")
        
    print("\nKeys only in Dataset 1:")
    keys1 = set(info1['keys'])
    keys2 = set(info2['keys'])
    print(sorted(list(keys1 - keys2)))
    
    print("\nKeys only in Dataset 2:")
    print(sorted(list(keys2 - keys1)))

if __name__ == "__main__":
    file1 = r"data\test_data_for_pipeline_EHL-2\jorek000000.h5"
    file2 = r"data\test_data_for_ITER_SPI_fast\jorek00000.h5"
    
    compare_datasets(file1, file2)
