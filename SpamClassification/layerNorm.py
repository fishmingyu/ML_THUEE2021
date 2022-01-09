import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, device, dimension):
        super(LayerNorm, self).__init__()
        # initialize 参与求梯度和迭代的拉伸和偏移参数
        self.eps = 1e-5
        self.gamma = nn.Parameter(torch.ones(dimension, 1)).to(device)
        self.beta = nn.Parameter(torch.zeros(dimension, 1)).to(device)

    def forward(self, X):
        # 对隐藏神经元做归一化
        mean = X.mean(dim=-1, keepdim=True)
        var = ((X - mean) ** 2).mean(dim=-1, keepdim=True)
        X_hat = (X - mean) / torch.sqrt(var + self.eps)
        return self.gamma * X_hat + self.beta  # 拉伸和偏移
    
