import torch
import numpy as np
import pandas as pd 
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import transforms
import torch.nn as nn 
import torch.nn.functional as F

train_set = pd.read_csv('./mnist/train.csv')
test_set = pd.read_csv('./mnist/test.csv')

class DatasetMNIST(Dataset):
    def __init__(self, data, transform=None, labeled=True):
        self.data = data
        self.transform = transform
        self.labeled = labeled

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):  # 重写，定义数据载入行为
        item = self.data.iloc[index]
        if self.labeled:  # 处理已标注数据
            x = item[1:].values.astype(np.uint8).reshape((28, 28))  # 图像像素数据
            y = item[0]  # 标注数字
        else:  # 处理未标注数据（缺少标注列，dimension不同，本文选择分别处理）
            x = item[0:].values.astype(np.uint8).reshape((28, 28))  # 图像像素数据
            y = 0  # 仅用于占位，数值不会被使用也不影响行为

        if self.transform is not None:
            x = self.transform(x)

        return x, y

# 训练集变换
transform_train = transforms.Compose([
    transforms.ToPILImage(), # 将张量tensor或ndarray转换为PIL图像，允许RandomAffine对数据进行操作
    transforms.RandomAffine(degrees=10, translate=(0, 0.1), scale=(0.9, 1.1)), # 随机旋转, 位移, 与缩放
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  
])

# 测试集变换
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
    
# 创建训练，验证，与测试集
train_data = DatasetMNIST(train_set,transform=transform_train)
test_data = DatasetMNIST(test_set,labeled=False,transform=transform_test)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.dr = nn.Dropout(p=0.4)
        self.conv1 = nn.Conv2d(1, 32, (3, 3))  # 输出大小转变为 26x26
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, (3, 3))  # 输出大小转变为 24x24
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, (5, 5), stride=(2, 2))  # 输出大小转变为 10x10
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, (3, 3))  # 输出大小转变为 8x8
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, (3, 3))  # 输出大小转变为 6x6
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, (5, 5), stride=(2, 2))  # 输出大小转变为 1x1
        self.bn6 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 1 * 1, 128)  # 特征数 * 输出维度，全连接层神经元数量 
        self.bn7 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):  # 重写，正向传播
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.dr(self.bn3((F.relu(self.conv3(x)))))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.bn5(F.relu(self.conv5(x)))
        x = self.dr(self.bn6(F.relu(self.conv6(x))))
        x = x.view(-1, 64 * 1 * 1)
        x = self.dr(self.bn7(F.relu(self.fc1(x))))
        x = self.fc2(x)
        
        return x

network = ConvNet()

n_epochs = 5   #训练的轮次
batch_size = 64   #batch_size
learning_rate = 0.01   #学习率
momentum = 0.9     #momentum梯度下降时的超参数

train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(dataset=test_data)

optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
loss_f = nn.CrossEntropyLoss()  #这里我们使用交叉熵损失函数

def train(epoch):
    network.train()  #表示进入了测试模式
    for batch_idx, (data, target) in enumerate(train_dataloader):
        optimizer.zero_grad()  #旧的梯度清零
        output = network(data)  #正向传播
        loss = loss_f(output,target)  #计算损失
        loss.backward()    #计算梯度
        optimizer.step()   #梯度下降
        if batch_idx%100==0 :
            print("epoch:",epoch," idx:",batch_idx," loss: ",loss.item())

for i in range(n_epochs):
    train(i+1)


with torch.no_grad():
    network.eval()
    results = torch.ShortTensor()
    for predict_images, _ in test_dataloader:
        predict_images = predict_images.reshape(-1, 1, 28, 28)
        predict_outputs = network(predict_images)
        test_predictions = predict_outputs.data.max(1, keepdim=True)[1]
        results = torch.cat((results, test_predictions), dim=0)

    submission = pd.DataFrame(np.c_[np.arange(1, len(test_set) + 1)[:, None], results.numpy()],
                              columns=['ImageId', 'Label'])
    submission.to_csv('./mnist/submission.csv', index=False)