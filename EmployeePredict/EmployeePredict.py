from RandomForest import RandomForest
from Data import BagSampler, DataLoader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--trees', type=int, default=10, help="number of decision trees")
parser.add_argument('--charac_m', type=int, default=2, help="number of character choices while training decison tree")
args = parser.parse_args()

train_frac = 0.8 #切分训练集和测试集的超参数
dataLoader = DataLoader('Employee.csv', 0.8)
train_loader, test_loader = dataLoader.genData()
rf = RandomForest(train_loader, args.trees, args.charac_m)
rf.train()
rf.test(test_loader)