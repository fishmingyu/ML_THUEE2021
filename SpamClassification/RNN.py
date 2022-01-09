import torch.nn as nn
from batchNorm import BatchNorm
from layerNorm import LayerNorm

class RNN(nn.Module):
    def __init__(self, device, embed_dim, hidden, word_num, norm_layer):
        super(RNN, self).__init__()
        self.device = device
        self.norm_layer = norm_layer
        self.rnn = nn.LSTM(    
            input_size=embed_dim,      # 输入的数据的feature
            hidden_size=hidden,     # rnn hidden unit
            num_layers=1,       # RNN layers
            batch_first=True,   # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, seq_len, feature)
        )
        self.embedding = nn.Embedding(word_num, embed_dim)
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, 1)    # 输出层
        self.dropout = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()
        self.bn = BatchNorm(self.device, hidden)
        self.ln = LayerNorm(self.device, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # h_n shape (n_layers, batch, hidden_size)   LSTM 有两个 hidden states, h_n 是分线, h_c 是主线
        # h_c shape (n_layers, batch, hidden_size)
        word_embed = self.embedding(x)
        word_embed = nn.functional.normalize(word_embed)
        r_out, (h_n, h_c) = self.rnn(word_embed, None)   # None 表示 hidden state 会用全0的 state

        # 选取最后一个时间点的 r_out 输出
        # print(r_out[:, -1, :])
        # exit()
        # out = self.dropout(r_out[:, -1, :])
        out = r_out[:, -1, :]
        if self.norm_layer == "ln":
            tmp = self.ln(out)
        elif self.norm_layer == "bn":
            tmp = self.bn(out)
        tmp = self.relu(self.fc1(out))
       
        pred = self.sigmoid(self.fc2(tmp)).squeeze(1)
        return pred
