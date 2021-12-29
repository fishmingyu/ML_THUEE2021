import torch
import argparse
import numpy as np
from data import DataLoader
from sklearn.naive_bayes import GaussianNB
from roadJudge import RoadJudge
from roadData import RoadData
from driverData import DriverData
from driverJudge import DriverJudge


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default='./input')
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--path_model', type=str, default='./model.pkl')
    args = parser.parse_args()

    data = DataLoader(args.input_folder)
    train_data, test_data = data.load() # 原始处理csv dataloader
    X_train, road_train, traffic_train, driver_train = train_data
    X_test, road_test, traffic_test, driver_test = test_data 

    # road task
    road_unique = np.unique(road_train) 
    dim = X_train.shape[1]
    rdata_train = RoadData(road_train, road_unique, X_train) # 处理train 数据
    rdata_test = RoadData(road_test, road_unique, X_test) # 处理test 数据
    y_data, X_data = rdata_train.process()
    
    model = RoadJudge(dim, road_unique) # 判断road的任务

    criterion = torch.nn.CrossEntropyLoss() # 交叉熵损失函数
    model.train(criterion, args.epochs, X_data, y_data) # 训练
    torch.save(model, args.path_model)

    y_data, X_data = rdata_test.process()
    print("start testing road judge...")
    model = torch.load(args.path_model)
    model.test(X_data, y_data) # 测试

    # driver task
    driver_unique = np.unique(road_train) 
    print(driver_unique)
    ddata_train = DriverData(driver_train, driver_unique, X_train)
    ddata_test = DriverData(driver_test, driver_unique, X_test)
    y_data, X_data = ddata_train.process()
    model = DriverJudge(X_data, y_data) # 判断driver 任务
    model.train()
    print("start testing driver judge...") # 训练
    y_data, X_data = ddata_test.process()
    model.test(X_data, y_data) # 测试
