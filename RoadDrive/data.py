import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import os

class DataLoader(object): 
    """
    这个模块主要处理csv文件
    """
    def __init__(self, input_folder):
        self.folder = input_folder
        self.target = ['roadSurface', 'traffic', 'drivingStyle'] # target 最终cols是str不用改成float

    def load(self):
        filepath = []
        df_frag = []
        for dirname, _, filenames in os.walk(self.folder): #遍历输入文件夹
            for filename in filenames:
                filepath.append(os.path.join(dirname, filename))
        for file in filepath:
            df_tmp = pd.read_csv(file, delimiter=';') #原始文件采用;作为分隔符
            df_frag.append(df_tmp)
        df_new = pd.concat(df_frag, axis = 0, ignore_index=True)
        df_new = df_new.fillna(0) #填充非0项目，原始文件中有非常多的NA值
        df_new = self.preprocess(df_new)
        train_data, test_data = self.train_spliter(df_new)
        train_data = self.source_target(train_data)
        test_data = self.source_target(test_data)
        return train_data, test_data

    def preprocess(self, df_new):
        """
        这个模块主要预处理数据。原始数据的小数点使用","表示
        """
        cols = list(df_new)
        for index, row in df_new.iterrows(): #遍历数据
            for col in cols:
                if col in self.target:
                    continue
                value = df_new.at[index, col]
                if type(value) == str:
                    df_new.at[index, col] = float(value.replace(",", "."))
        return df_new
    def train_spliter(self, df):
        """
        这个模块主要切分训练集和测试集
        """
        alldata = df.sample(frac=1).reset_index(drop=True) #从数据中重新shuffle排序
        bound = round(0.8 * len(alldata)) #区分训练集和测试集
        train_data = alldata.iloc[:bound, :]
        test_data = alldata.iloc[bound:, :]
        return train_data, test_data
    def source_target(self, data):
        """
        这个模块主要区分源和目标的数据，切成X以及y
        """
        cols = list(data)
        Xlen = len(cols) - len(self.target)
        X = np.array(data.iloc[:, :Xlen]).astype(float) #强制转换成float
        road = np.array(data.iloc[:, Xlen])
        traffic = np.array(data.iloc[:, Xlen+1])
        driver = np.array(data.iloc[:, Xlen+2]) 
        X = torch.from_numpy(X).float()
        return X, road, traffic, driver
