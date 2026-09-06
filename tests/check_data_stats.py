import sys
import os
import torch
import h5py
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.jorek_h5_parser import JorekDataParser

def check_data_stats():
    # Use the first file
    filepath = "data/EHL-2/scan_1d22_ang12_massloss_neon/jorek002710.h5"
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    print(f"Checking statistics for {filepath}")
    
    try:
        # Manually parse using JorekDataParser logic to check raw values
        parser = JorekDataParser(filepath)
        data = parser.parse_to_graph()
        print(f"time_now: {data.time}")
        print(f"norm_factors: {data.norm_factors}")
        x = data.x
        print(f"Data shape: {x.shape}")
        
        names = ["Psi", "Phi", "Rho", "T", "V_par"]
        for i in range(5):
            col = x[:, i]
            print(f"{names[i]}: Min={col.min().item():.4e}, Max={col.max().item():.4e}, Mean={col.mean().item():.4e}, Std={col.std().item():.4e}")
            
            if torch.isnan(col).any():
                print(f"  WARNING: {names[i]} contains NaNs!")
                
    except Exception as e:
        print(f"Error parsing file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_data_stats()
