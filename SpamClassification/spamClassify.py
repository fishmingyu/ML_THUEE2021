import torch.nn as nn
import math
import argparse
import torch
from tqdm import tqdm
from dataLoader import DataLoader
from RNN import RNN

def train(model, device, criterion, optimizer, epochs, train_data, batch_size):
    X, y = train_data
    train_num = X.shape[0]
    X = torch.from_numpy(X).long().to(device)
    y = torch.Tensor(y).to(device)

    loss_list = []
    # batch training
    for epoch in range(epochs):
        loss = 0
        
        num_batch = math.ceil(train_num/batch_size)
        for i in range(num_batch): # 按照范围遍历
            if (i+1)*batch_size > train_num:
                batch_X = X[i*batch_size:]
                batch_y = y[i*batch_size:]
            else:
                batch_X = X[i*batch_size:i*batch_size+batch_size]
                batch_y = y[i*batch_size:i*batch_size+batch_size] 
            out = model(batch_X) # (batch, seq_len)
            batch_loss = criterion(out, batch_y)
            optimizer.zero_grad()
            loss += batch_loss
            batch_loss.backward()
            optimizer.step()
        loss_list.append(loss.detach().cpu().numpy().tolist())
        if (epoch + 1) % 5 == 0: 
            print(f"| epoch: {epoch:5} | loss is {loss.detach().cpu().numpy()/num_batch:6.4f} |")
        file1 = open('loss_file', 'w')
        file1.write(str(loss_list))

def evaluate(model, device, data):
    model.eval()
    X, y = data
    X = torch.from_numpy(X).long().to(device)
    y = torch.Tensor(y).to(device)
    out = model(X)
    count = 0
    for i in range(len(out)): # 按照0.5为阈值进行截断判决
        if out[i] > 0.5:
            if y[i] == 1:
                count += 1 
        else:
            if y[i] == 0:
                count += 1
    acc = count / len(out)
    print(f"acc: {acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='spam.csv')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--hidden', type=int, default=36)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--embed_dim', type=int, default=16)
    parser.add_argument('--norm', type=str, default='ln', help="choose ln for layer norm, bn for batch norm")
    args = parser.parse_args()
    file = args.file
    device = 0
    dl = DataLoader(file)
    
    model = RNN(device, args.embed_dim, args.hidden, dl.word_num, args.norm) # RNN 模型
    model.to(device)
    criterion = nn.BCELoss() # 交叉熵
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train(model, device, criterion, optimizer, args.epochs, dl.train_data, args.batch_size)
    evaluate(model, device, dl.test_data)
    
