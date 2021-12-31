from ID3Tree import DecisionTreeID3
from tqdm import tqdm
import collections
import numpy as np

class RandomForest(object):
    """
    使用随进森林进行对员工的预测，森林中每一颗树采用ID3树构造生成
    """
    def __init__(self, data, numberOfTrees=20, charac_m = 2):
        self.data = data
        self.numberOfTrees = numberOfTrees # 构建树的个数
        self.charac_m = charac_m
        self.forest = [] 

        for _ in range(numberOfTrees):
            bag = data.getTrainSample() # 采用bagging的方法进行采样
            self.forest.append(DecisionTreeID3(bag, charac_m))   # 每一个树拿一个bag进行训练

    def train(self):
        '''
        Train the random forest trees.
        '''
        for tree in tqdm(self.forest):
            tree.fit()

    def classify(self, sample):
        '''
        对样本进行预测分类
        '''
        # 构建一个的array，每一个里面装一个最后判断的值
        size = len(sample)
        predict = np.empty((self.numberOfTrees, size))
        # 构建一个大的数组，每一行代表一棵树处理的所有sample的结果，每一个sample按列对齐
        # 遍历每一棵树，得到所有样本的结果，由于最后target都是数值结果，所以我们可以用numpy高效的处理
        for i in range(self.numberOfTrees):
            res = self.forest[i].predict(sample)
            predict[i, :] = res
        sum_res = np.sum(predict, axis = 0)
        np.putmask(sum_res, sum_res <= (self.numberOfTrees / 2), 0) 
        np.putmask(sum_res, sum_res > (self.numberOfTrees / 2), 1) 
        
        return sum_res.tolist()

    def test(self, data_loader):
        sample = data_loader.getTestSample()
        X = sample[0]
        label = sample[1].values
        predict = np.array(self.classify(X))
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        for i in range(predict.shape[0]):
            if predict[i] == 1 and label[i] == 1:
                TP += 1
            elif predict[i] == 0 and label[i] == 0:
                TN += 1
            elif predict[i] == 1 and label[i] == 0:
                FP += 1
            else:
                FN += 1
        acc = 1 - sum(abs(label - predict)) / len(predict)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        print(f"test acc is {acc:.4f}, precision is {precision:.4f}, recall is {recall:.4f}")
        return acc
