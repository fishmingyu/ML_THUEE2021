import torch
import numpy as np
import torch.nn.functional as F


class RoadData(object):
    """
    这个模块主要处理路相关的数据
    """
    def __init__(self, road_situation, road_unique, X):
        self.X = X
        self.road_situation = road_situation
        self.road_unique = road_unique# 去重，这里只取出几个结果
    
    def process(self): # 原始路况有三类。为了适应逻辑回归，采用softmax，数据处理采用标签的编码
        road_process = torch.zeros(len(self.road_situation))
        X_process = []
        for rs in self.road_unique: # 适应最后的交叉熵
            road_index = np.where(self.road_situation == rs)[0] #返回是个tuple，需要处理一下
            r_index = self.road_unique.tolist().index(rs)
            road_process[road_index] = torch.mul(torch.ones(len(road_index)), r_index) # 对应坐标放入对应的标签值
        road_process = road_process.long()
        X_process = self.uniform(self.X)
        return road_process, X_process

    def uniform(self, X_process): # 按照最大绝对值每一列归一化
        max_x, _ = torch.max(torch.abs(X_process), dim = 0)
        X_process = torch.div(X_process, max_x)
        return X_process