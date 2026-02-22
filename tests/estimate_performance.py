import sys
import os
import torch
import torch.nn as nn
import time
from tqdm import *
import psutil
from torch_geometric.data import Data, Batch

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.gnn_encoder import JorekGNNEncoder
from src.models.lnn_processor import JorekLNNProcessor
from src.models.gnn_decoder import JorekGNNDecoder

def get_gpu_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        return f"{allocated:.2f} MB (Allocated), {reserved:.2f} MB (Reserved)"
    return "N/A"

def estimate_performance():
    print("==================================================")
    print("🚀 Estimating GPU Performance & Training Time")
    print("==================================================")

    device_count = torch.cuda.device_count()
    if device_count == 0:
        print("❌ No GPU found. Training will be extremely slow on CPU.")
        device = torch.device('cpu')
    else:
        for i in range(device_count):
            print(f"✅ GPU {i}: {torch.cuda.get_device_name(i)}")
        device = torch.device('cuda:0')

    # Mock Data Parameters
    BATCH_SIZE = 6
    SEQ_LEN = 10
    NUM_NODES = 4125
    NUM_EDGES = 32186 # Approx based on JOREK grid
    INPUT_DIM = 5
    HIDDEN_DIM = 100
    LATENT_DIM = 32
    
    # Create Mock Batch
    print(f"\nConstructing Mock Batch (Batch Size: {BATCH_SIZE}, Seq Len: {SEQ_LEN})...")
    
    # Single graph mock data
    x = torch.randn(NUM_NODES, INPUT_DIM)
    edge_index = torch.randint(0, NUM_NODES, (2, NUM_EDGES))
    pos = torch.randn(NUM_NODES, 2)
    batch_vec = torch.zeros(NUM_NODES, dtype=torch.long) # Single graph batch vec
    
    # Create batch of graphs
    data_list = [Data(x=x, edge_index=edge_index, pos=pos) for _ in range(BATCH_SIZE)]
    batch = Batch.from_data_list(data_list)
    
    # Create sequence of batches
    batch_sequence = [batch.clone() for _ in range(SEQ_LEN)]
    
    # Move to device
    print(f"Moving data to {device}...")
    try:
        batch_sequence = [b.to(device) for b in batch_sequence]
        print(f"Memory Usage after loading data: {get_gpu_memory_usage()}")
    except RuntimeError as e:
        print(f"❌ Failed to load data to GPU: {e}")
        return

    # Initialize Models
    print("\nInitializing Models...")
    encoder = JorekGNNEncoder(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM, num_layers=4).to(device)
    processor = JorekLNNProcessor(latent_dim=LATENT_DIM, hidden_size=HIDDEN_DIM, num_layers=2).to(device)
    decoder = JorekGNNDecoder(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, output_dim=INPUT_DIM, num_layers=4).to(device)
    
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(processor.parameters()) + list(decoder.parameters()),
        lr=1e-4
    )
    
    print(f"Memory Usage after loading models: {get_gpu_memory_usage()}")

    # Benchmark Loop
    print("\nStarting Benchmark Loop (10 iterations)...")
    encoder.train(); processor.train(); decoder.train()
    
    timings = []
    
    # Warmup
    for _ in range(2):
        optimizer.zero_grad()
        hidden_state = None
        for t in range(SEQ_LEN - 1):
            current_data = batch_sequence[t]
            latent_x = encoder(current_data.x, current_data.edge_index, current_data.pos)
            next_latent_x, hidden_state = processor(latent_x, current_data.batch, state=hidden_state)
            pred_x, pred_p = decoder(next_latent_x, current_data.edge_index, current_data.batch, current_data.pos)
            loss = pred_x.mean() + pred_p.mean()
            loss.backward(retain_graph=True)
        optimizer.step()
        torch.cuda.synchronize() if torch.cuda.is_available() else None

    # Measurement
    for i in range(10):
        start_time = time.time()
        
        optimizer.zero_grad()
        hidden_state = None
        
        # Unroll sequence
        for t in tqdm(range(SEQ_LEN - 1)):
            current_data = batch_sequence[t]
            # 1. Perception
            latent_x = encoder(current_data.x, current_data.edge_index, current_data.pos)
            # 2. Cognition
            next_latent_x, hidden_state = processor(latent_x, current_data.batch, state=hidden_state)
            # 3. Reconstruction
            pred_x, pred_p = decoder(next_latent_x, current_data.edge_index, current_data.batch, current_data.pos)
            
            # Mock Loss
            loss = pred_x.sum() + pred_p.sum()
            loss.backward(retain_graph=True)
        
        optimizer.step()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        end_time = time.time()
        timings.append(end_time - start_time)
        print(f"  Iteration {i+1}: {timings[-1]:.4f} s")

    avg_time = sum(timings) / len(timings)
    print(f"\nAverage Step Time: {avg_time:.4f} s")
    
    # Estimation
    # Assume dataset size ~ 100 samples (from check_data_stats ~ 116)
    DATASET_SIZE = 116
    STEPS_PER_EPOCH = DATASET_SIZE // BATCH_SIZE
    EPOCH_TIME = avg_time * STEPS_PER_EPOCH
    
    print("-" * 50)
    print(f"ESTIMATED TRAINING TIME (for Dataset Size: {DATASET_SIZE})")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Steps per Epoch: {STEPS_PER_EPOCH}")
    print(f"  Time per Epoch: {EPOCH_TIME:.2f} s (~ {EPOCH_TIME/60:.2f} min)")
    print(f"  100 Epochs: {EPOCH_TIME * 100 / 60:.2f} min (~ {EPOCH_TIME * 100 / 3600:.2f} hours)")
    print("-" * 50)
    print("Recommendation: Use this to tune Batch Size/Seq Len if training is too slow.")

if __name__ == "__main__":
    estimate_performance()
