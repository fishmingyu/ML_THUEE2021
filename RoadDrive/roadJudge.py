import torch
import numpy as np
from Logistic import LogisticRegressionModel

class RoadJudge(object):
    """
    这个模块主要负载判断路的情况
    """
    def __init__(self, dim, road_unique):
        self.dim = dim
        self.road_unique = road_unique 
        self.model = LogisticRegressionModel(dim) # LR model
        
    def train(self, criterion, epochs, X_data, y_data):
        print("start training road model")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.008)
        self.model_train(self.model, criterion, optimizer, epochs, X_data, y_data)

    def test(self, X, y): # 测试样本值
        acc = self.model_test(self.model, X, y)
        print(f"test acc is {acc:.4f}")

    def model_validate(self, pred, data): # 验证模型中最后的准确率
        pred = torch.argmax(pred, dim = 1)
        count = 0
        size = pred.shape[0]
        for i in range(size):
            if pred[i] == data[i]:
                count += 1
        acc = count/size
        return acc

    def model_test(self, model, X, y):
        y_pred = model(X)
        acc = self.model_validate(y_pred, y)
        return acc

    def model_train(self, model, criterion, optimizer, epochs, X, y_data):
        for epoch in range(epochs):
            y_pred = model(X)
            loss = criterion(y_pred, y_data) # 采用交叉熵做loss
            optimizer.zero_grad()
            if (epoch + 1) % 100 == 0: 
                print(f"| epoch: {epoch:5} | loss is {loss.detach().numpy():6.4f} |")
            loss.backward()
            optimizer.step()
        acc = self.model_validate(y_pred, y_data)
        print(f"train acc is {acc:.4f}")