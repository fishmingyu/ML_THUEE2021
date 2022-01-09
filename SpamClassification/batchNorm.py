import torch
import torch.nn as nn

def batch_norm(is_training, X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # judge training
    if not is_training:
        # evaluation
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        mean = X.mean(dim=0)
        var = ((X - mean) ** 2).mean(dim=0)
        # 训练模式下用当前的均值和方差做标准化
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # 动量更新
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # 拉伸和偏移
    return Y, moving_mean, moving_var

class BatchNorm(nn.Module):
    def __init__(self, device, num_features):
        super(BatchNorm, self).__init__()
        shape = (1, num_features)
        # initialize 参与求梯度和迭代的拉伸和偏移参数
        self.gamma = nn.Parameter(torch.ones(shape)).to(device)
        self.beta = nn.Parameter(torch.zeros(shape)).to(device)
        # 不参与求梯度和迭代的变量，全在内存上初始化成0
        self.moving_mean = torch.zeros(shape).to(device)
        self.moving_var = torch.zeros(shape).to(device)

    def forward(self, X):
        # 保存更新过的moving_mean和moving_var, Module实例的traning属性默认为true, 调用.eval()后设成false
        Y, self.moving_mean, self.moving_var = batch_norm(self.training, 
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.95)
        return Y
