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

data = pd.read_csv("re_data.csv")
data = data.sort_values(by=['ID'])
data_x = data.drop(["乳量"],axis=1)
data_y = data["乳量"]
data_test = data_x[data["乳量"] < 0]
data_train_x = data_x[data["乳量"] >= 0]
data_train_y = data_y[data["乳量"] >= 0]
X_train, X_test, y_train, y_test = train_test_split(data_train_x,data_train_y,test_size=0.2)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        model_resnet = resnet18(pretrained=False)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                            self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

        self.out = nn.Linear(model_resnet.fc.in_features, 1)
        self.out.weight.data.normal_(0, 0.01)
        self.out.bias.data.fill_(0.0)

    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        y = self.out(x)
        return y

device = torch.device("cpu")
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
train_loss = 0
bz = 128
from tqdm import tqdm
for epoch in range(1000):
    index_list = np.random.permutation(X_train.shape[0])
    index_list = np.array_split(index_list, X_train.shape[0] // bz, axis=0)
    trian_loss = 0

    for index in tqdm(index_list):

        x = np.array(X_train)[index].reshape(-1, 1, 25, 21)
        x = np.tile(x,(1,3,1,1))

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
    x = np.array(X_test).reshape(-1, 1, 25, 21)
    x = np.tile(x,(1,3,1,1))

    valid_batch_x = torch.from_numpy(x).float().to(device)
    valid_batch_y = torch.from_numpy(np.array(y_test)).to(device)
    out = model(valid_batch_x)
    loss = (out - valid_batch_y).pow(2).mean()
    print("   测试集",loss.pow(0.5).item())

    x = np.array(data_test).reshape(-1, 1, 25, 21)
    x = np.tile(x,(1,3,1,1))
    test_batch_x = torch.from_numpy(x).float().to(device)
    out = model(test_batch_x)
    pd.DataFrame({"ID": pd.read_csv("data/submission.csv")["ID"], "1": out.view(-1).tolist()}).to_csv("resnet.csv", index=False)


