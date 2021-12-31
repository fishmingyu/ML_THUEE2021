import pandas as pd
from pandas.core.frame import DataFrame 
import numpy as np

class BagSampler(object):
    def __init__(self, dataFrame, dataRate=1): # dataRate 代表要从dataFrame数据中每次采样的数据占比
        self.dataFrame = dataFrame
        self.dataRate = dataRate
        self.dataNum = round(len(dataFrame) * dataRate)
    def getTrainSample(self):
        randarray = np.random.randint(0, self.dataNum, self.dataNum) # 模拟bagging算法，在df个数据中进行采样，是有放回的采样，因此最后是有重复的。
        df = self.dataFrame.iloc[randarray, :]
        df = df.reset_index(drop=True)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        return [X, y]
    def getTestSample(self): # test 直接取就行
        df = self.dataFrame
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        return [X, y]

class DataLoader(object):
    def __init__(self, file, train_frac, dataRate=1):
        self.df = pd.read_csv(file, index_col = 0, parse_dates = True)
        self.train_frac = train_frac
        self.dataRate = dataRate

    def genData(self):
        alldata = self.df.sample(frac=1).reset_index() #从数据中重新shuffle排序
        bound = round(self.train_frac * len(alldata)) #区分训练集和测试集
        train_data = alldata.iloc[:bound, :]
        test_data = alldata.iloc[bound:, :]
        train_loader = BagSampler(train_data, self.dataRate)
        test_loader = BagSampler(test_data, self.dataRate)
        atts = list(alldata.iloc[:, :-1])
        return train_loader, test_loader
