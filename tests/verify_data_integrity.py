import sys
import os
import torch
import numpy as np
from tqdm import tqdm
import glob

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.jorek_h5_parser import JorekDataParser
from src.data.data_manifest import aligned_frames, discover_runs, is_2d_run

def verify_data_integrity():
    DATA_DIR = "data/EHL-2"
    # Only 2D (ntor=1) marker-aligned training frames are in scope (ADR-0004).
    files = []
    for run in discover_runs(DATA_DIR):
        if is_2d_run(run.path):
            files.extend(frame.state_path for frame in aligned_frames(run.path))
    files = sorted(files, key=lambda path: str(path))
    
    if not files:
        print(f"No aligned 2D frames found under {DATA_DIR}")
        return

    print(f"Found {len(files)} files. Starting integrity check...")
    print(f"{'File':<20} | {'Nodes':<10} | {'Edges':<10} | {'Status':<10} | {'Issues'}")
    print("-" * 80)

    previous_num_nodes = None
    issues_found = False

    for filepath in tqdm(files, desc="Checking files"):
        filename = os.path.basename(filepath)
        parser = JorekDataParser(filepath)
        
        try:
            data = parser.parse_to_graph()
            
            # 1. Check for NaN/Inf in Features (x)
            if torch.isnan(data.x).any() or torch.isinf(data.x).any():
                status = "FAIL"
                issue = "NaN/Inf in Features (x)"
                issues_found = True
            
            # 2. Check for NaN/Inf in Coordinates (pos)
            elif torch.isnan(data.pos).any() or torch.isinf(data.pos).any():
                status = "FAIL"
                issue = "NaN/Inf in Coordinates (pos)"
                issues_found = True
                
            # 3. Check Grid Alignment (Topology Consistency)
            # JOREK simulations might use adaptive mesh refinement. 
            # If nodes count changes, simple reshaping in LNN might fail.
            elif previous_num_nodes is not None and data.num_nodes != previous_num_nodes:
                status = "WARN"
                issue = f"Node count changed: {previous_num_nodes} -> {data.num_nodes}"
                # This is not necessarily an error in data, but can cause issues in fixed-size tensor batches
            
            else:
                status = "OK"
                issue = ""

            if status != "OK":
                print(f"{filename:<20} | {data.num_nodes:<10} | {data.num_edges:<10} | {status:<10} | {issue}")

            previous_num_nodes = data.num_nodes

        except Exception as e:
            print(f"{filename:<20} | {'ERROR':<10} | {'-':<10} | {'FAIL':<10} | Exception: {str(e)}")
            issues_found = True

    if not issues_found:
        print("\nPASS: All files passed integrity check. No NaNs, Infs, or read errors found.")
        print("   Rank-1 tensor shapes (grids) appear consistent (or consistent change warnings not triggered).")
    else:
        print("\nWARNING: Issues were found in the dataset. See above for details.")

if __name__ == "__main__":
    verify_data_integrity()
