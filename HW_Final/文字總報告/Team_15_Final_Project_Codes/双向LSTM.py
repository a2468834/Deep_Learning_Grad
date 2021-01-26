import pandas as pd
import math
from torch import nn
import torchvision
import torch
from torchvision.models import *
import numpy as np
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
data = pd.read_csv("test_data4.csv")
data = data.sort_values(by=['ID'])
data_x = data.drop(["乳量"],axis=1)
data_y = data["乳量"]
data_test = data_x[data["乳量"] < 0]
data_train_x = data_x[data["乳量"] >= 0]
data_train_y = data_y[data["乳量"] >= 0]
X_train, X_test, y_train, y_test = train_test_split(data_train_x,data_train_y,test_size=0.2)
hidden1 = 512
hidden2 = 256
class Net(nn.Module):
    def __init__(self, in_features, hidden_size=300, out_size=512):
        super(Net, self).__init__()

        self.rnn = nn.LSTM(in_features, hidden_size // 2, batch_first=True, bidirectional=True)

        self.fc = nn.Linear(hidden_size, out_size)
        nn.init.uniform_(self.fc.weight, -0.5, 0.5)
        nn.init.uniform_(self.fc.bias, -0.1, 0.1)

        self.dropout = nn.Dropout()

        self.cls = nn.Linear(out_size, 1)
        nn.init.uniform_(self.cls.weight, -0.1, 0.1)
        nn.init.uniform_(self.cls.bias, -0.1, 0.1)


    def forward(self, x):
        x = x.unsqueeze(1)
        # print(x.shape)
        # 转换数据维度，以便通过rnn
        # x = x.permute(0, 2, 1)
        # print(x.shape)
        x, _ = self.rnn(x)
        # print(x.shape)
        x = self.fc(x.mean(1))
        # print(x.shape)
        x = self.dropout(x)
        # print(x.shape)
        x = self.cls(x)  # batch_size * word_len
        # print(x.shape)
        return x

device = torch.device("cpu")
model = Net(X_train.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
train_loss = 0
bz = 128
from tqdm import tqdm
for epoch in range(1000):
    index_list = np.random.permutation(X_train.shape[0])
    index_list = np.array_split(index_list, X_train.shape[0] // bz, axis=0)
    trian_loss = 0

    for index in tqdm(index_list):

        x = np.array(X_train)[index]

        train_batch_x = torch.from_numpy(x).float().to(device)
        train_batch_y = torch.from_numpy(np.array(y_train)[index]).to(device)

        model.train()

        out = model(train_batch_x)

        loss = (out - train_batch_y).pow(2).mean()
        trian_loss  += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("epoch",epoch,"训练集",(trian_loss / len(index_list)) ** 0.5,end="  ")

    model.eval()
    x = np.array(X_test)

    valid_batch_x = torch.from_numpy(x).float().to(device)
    valid_batch_y = torch.from_numpy(np.array(y_test)).to(device)
    out = model(valid_batch_x)
    loss = (out - valid_batch_y).pow(2).mean()
    print("   测试集",loss.pow(0.5).item())

    x = np.array(data_test)
    test_batch_x = torch.from_numpy(x).float().to(device)
    out = model(test_batch_x)
    pd.DataFrame({"ID": pd.read_csv("data/submission.csv")["ID"], "1": out.view(-1).tolist()}).to_csv("LSTM.csv", index=False)


