import sys
import torch
import numpy as np

def check_env():
    print("="*30)
    print("HyPhy-LNN Environment Check")
    print("="*30)

    # 1. Python Version
    print(f"[OK] Python version: {sys.version.split()[0]}")

    # 2. PyTorch & CUDA
    try:
        print(f"[OK] PyTorch version: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"[OK] CUDA is available. Device: {torch.cuda.get_device_name(0)}")
            x = torch.tensor([1.0, 2.0]).cuda()
            print(f"     GPU Tensor Test: {x} (Success)")
        else:
            print("[WARNING] CUDA not available. Training will be slow on CPU.")
    except Exception as e:
        print(f"[FAIL] PyTorch/CUDA check failed: {e}")

    # 3. NCPS (Liquid Neural Networks)
    try:
        from ncps.torch import CfC
        from ncps.wirings import AutoNCP
        print(f"[OK] NCPS (Liquid Neural Networks) library imported successfully.")
        
        # 简单实例化测试
        rnn = CfC(input_size=10, units=20)
        print("     CfC Layer instantiated successfully.")
    except ImportError:
        print("[FAIL] 'ncps' library not found. Did you run 'pip install -r requirements.txt'?")
    except Exception as e:
        print(f"[FAIL] NCPS check failed: {e}")

    # 4. H5PY (JOREK Data)
    try:
        import h5py
        print(f"[OK] h5py imported (ready for JOREK data).")
    except ImportError:
        print("[FAIL] h5py not found.")

    print("="*30)
    print("Environment Verification Complete.")
    print("="*30)

if __name__ == "__main__":
    check_env()