from matplotlib.colors import LogNorm
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from ncps.torch import CfC
from ncps.wirings import AutoNCP

def minimal_lnn_run():
    print("\nRunning Minimal LNN Test Case (Sine Wave Prediction)...")
    
    # -----------------------------------------
    # 1. 制造“非均匀时间”数据 (Irregular Time Sampling)
    # LNN 的强项在于处理 dt 变化的数据
    # -----------------------------------------
    batch_size = 1
    seq_len = 1000
    input_size = 1
    hidden_size = 16
    
    # 生成时间轴：随机的时间步长 dt
    # 模拟 JOREK 的自适应步长
    dt = torch.rand(batch_size, seq_len) * 0.1 + 0.01  # dt in [0.01, 0.11]
    times = torch.cumsum(dt, dim=1) # 绝对时间 t
    print('total time:', times[0,-1].item(), '\n dt = ', dt[0,:10])
    # 目标函数：y = sin(t)
    # 输入特征：x = sin(t) + noise (稍微加点噪声)
    target = 0.5* torch.sin(times).unsqueeze(-1) + 0.5*torch.cos(2*times).unsqueeze(-1) # [Batch, Seq, 1]
    input_data = target + 0.05 * torch.randn_like(target)

    print(f"Data shape: {input_data.shape}")
    print(f"Time spans shape: {times.shape} (Simulating adaptive time-steps)")

    # -----------------------------------------
    # 2. 定义简单的 HyPhy-LNN 原型
    # -----------------------------------------
    class SimpleLNN(nn.Module):
        def __init__(self, in_features, hidden_features, out_features):
            super().__init__()
            # 使用 Closed-form Continuous (CfC) 单元
            # 它可以明确接收 times 或 dt 作为输入
            self.lnn = CfC(input_size=in_features, 
                           units=hidden_features, 
                           proj_size=out_features,
                           return_sequences=True)
        
        def forward(self, x, times):
            # ncps 库中的 forward 稍微有些不同，通常不需要显式传递 dt，
            # 但如果数据非均匀，最好的做法是把 times 传进去让内部 ODE 求解器处理
            # 这里演示最基础的调用，CfC 会学习隐含的时间常数
            output, _ = self.lnn(x) 
            return output

    model = SimpleLNN(input_size, hidden_size, 1)
    
    # 如果有 GPU，转移到 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_data = input_data.to(device)
    target = target.to(device)
    
    # -----------------------------------------
    # 3. 简单的训练循环 (验证反向传播)
    # -----------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print("Starting training loop (2000 epochs)...")
    losses = []
    
    for epoch in range(3000):
        optimizer.zero_grad()
        
        # Forward pass
        # 注意：这里我们简化了，没有显式传 times 给 forward (ncps 默认假设 dt=1)
        # 在正式项目中，你需要根据 ncps 文档处理 variable dt
        prediction = model(input_data, times) 
        
        loss = criterion(prediction, target)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/3000 | Loss: {loss.item():.6f}")
            
        

    print("Training finished.")
    epoches = np.arange(1, len(losses)+1)
    plt.figure(figsize=(6,4),dpi=300)
    plt.plot(epoches, losses, label="Training Loss")
    plt.yscale('log')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig("lnn_training_loss.png")
    print("Training loss plot saved to 'lnn_training_loss.png'")
    plt.close()

    # -----------------------------------------
    # 4. 可视化结果
    # -----------------------------------------
    with torch.no_grad():
        pred_np = prediction.cpu().numpy()[0, :, 0]
        target_np = target.cpu().numpy()[0, :, 0]
        time_np = times.cpu().numpy()[0, :]

    try:
        plt.figure(figsize=(10, 4),dpi=300)
        plt.plot(time_np, target_np, label="Ground Truth (Physics)", linestyle="--", color="black")
        plt.plot(time_np, pred_np, label="LNN Prediction", color="red", alpha=0.8)
        plt.scatter(time_np, input_data.cpu().numpy()[0, :, 0], s=10, alpha=0.3, label="Noisy Input")
        plt.title("Minimal LNN Test: Reconstructing Sine Wave from Irregular Steps")
        plt.legend()
        plt.grid(True)
        plt.savefig("lnn_test_result.png")
        print("Result saved to 'lnn_test_result.png'")
    except Exception as e:
        print(f"Plotting failed (maybe no display): {e}")

if __name__ == "__main__":
    minimal_lnn_run()