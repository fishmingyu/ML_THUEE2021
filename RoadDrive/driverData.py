import torch
import numpy as np
import torch.nn.functional as F

class DriverData(object):
    """
    这个模块主要负载处理司机的数据
    """
    def __init__(self, driver_situation, driver_unique, X):
        self.X = X
        self.driver_situation = driver_situation
        self.driver_unique = driver_unique # 去重，这里只取出几个结果
    
    def process(self): # driver有二类。要求使用高斯概率生成模型，因此使用GDA算法。处理数据时使用0，1来标记数据
        X_process = []
        rs = self.driver_unique[0]
        driver_index = np.where(self.driver_situation == rs)[0] #返回是个tuple，需要处理一下
        driver_process = torch.zeros(len(self.driver_situation)) # 生成one hot 编码
        driver_process[driver_index] = torch.ones(len(driver_index)) # 在原始数据中重新找到数据 
        X_process = self.uniform(self.X).numpy()
        return driver_process, X_process

    def uniform(self, X_process): # 按照最大绝对值每一列归一化
        max_x, _ = torch.max(torch.abs(X_process), dim = 0)
        X_process = torch.div(X_process, max_x) 
        return X_process