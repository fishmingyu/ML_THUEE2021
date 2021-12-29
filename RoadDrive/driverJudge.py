from sklearn.naive_bayes import GaussianNB


class DriverJudge(object):
    """
    这个模块主要负载判断司机的情况
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.clf = GaussianNB() #调用sklearn 中朴素贝叶斯的高斯分类
    def train(self):
        print("start training driver judge...")
        self.clf.fit(self.X, self.y)
        acc = self.validate(self.X, self.y)
        print(f"train acc is {acc:.4f}")
    def test(self, sample, truth):
        acc = self.validate(sample, truth)
        print(f"test acc is {acc:.4f}")
        return acc
    def validate(self, sample, truth): # 验证最后预测结果的正确率
        res = self.clf.predict(sample)
        count = 0
        for i in range(len(sample)):
            if res[i] == truth[i]:
                count += 1
        acc = count / len(sample)
        return acc