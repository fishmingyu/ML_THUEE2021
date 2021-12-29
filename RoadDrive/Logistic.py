import torch
import torch.nn as nn
import torch.nn.functional as F 


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear1 = nn.Linear(dim, 45)
        self.linear2 = nn.Linear(45, 25)
        self.linear3 = nn.Linear(25, 3)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        y_pred = self.softmax(self.linear3(x))
        return y_pred
