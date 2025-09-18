import torch
import torchvision
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import random
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm  # 進捗バー用
from datetime import datetime

# random seedを設定
seed = 2025
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# 正規化されたTensorに変換
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 画像の表示
def imshow(img):  # 画像の表示関数
    img = img / 2 + 0.5     # 正規化を戻す
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# 適当な訓練セットの画像を取得
dataiter = iter(trainloader)
images, labels = next(dataiter)

# # 画像の表示
# imshow(torchvision.utils.make_grid(images))
# # ラベルの表示
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

device = "cuda"
net = Net().to(device)



# 学習部分
def train(net, opt, criterion, num_epochs=10):
    """
    データの繰り返し処理を実行し、入力データをネットワークに与えて最適化します
    """
    # 学習経過を格納するdict
    history = {"loss":[], "accuracy":[], "val_loss":[], "val_accuracy":[]}

    for epoch in range(num_epochs):
        train_loss = 0.
        train_acc = 0.
        valid_loss = 0.
        valid_acc = 0.
        train_total = 0
        valid_total = 0

        # 学習
        for data in tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            inputs, labels = data[0].to(device), data[1].to(device)
            opt.zero_grad() # 勾配情報をリセット
            pred = net(inputs)  # モデルから予測を計算(順伝播計算)：tensor(BATCH_SIZE, 確率×10)
            loss = criterion(pred, labels) # 誤差逆伝播の微分計算
            train_loss += loss.item() # 誤差(train)を格納
            loss.backward()
            opt.step()  # 勾配を計算
            _, indices = torch.max(pred.data, axis=1)  # 最も確率が高いラベルの確率と引数をbatch_sizeの数だけ取り出す
            train_acc += (indices==labels).sum().item() # labelsと一致した個数
            train_total += labels.size(0) # データ数(=batch_size)

        history["loss"].append(train_loss)  # 1epochあたりの誤差の平均を格納
        history["accuracy"].append(train_acc/train_total) # 正解数/使ったtrainデータの数

        # 学習ごとの検証
        with torch.no_grad():
            for data in tqdm(testloader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                inputs, labels = data[0].to(device), data[1].to(device)
                pred = net(inputs)  # モデルから予測を計算(順伝播計算)：tensor(BATCH_SIZE, num_class)
                loss = criterion(pred, labels) # 誤差の計算
                valid_loss += loss.item()  # 誤差(valid)を格納
                values, indices = torch.max(pred.data, axis=1)  # 最も確率が高い引数をbatch_sizeの数だけ取り出す
                valid_acc += (indices==labels).sum().item()
                valid_total += labels.size(0) # データ数(=batch_size)

        history["val_loss"].append(valid_loss)  # 1epochあたりの検証誤差の平均を格納
        history["val_accuracy"].append(valid_acc/valid_total) # 正解数/使ったtestデータの数
        # 5の倍数回で結果表示
        if (epoch+1)%5==0:
            print(f'Epoch：{epoch+1:d} | loss：{history["loss"][-1]:.3f} accuracy: {history["accuracy"][-1]:.3f} val_loss: {history["val_loss"][-1]:.3f} val_accuracy: {history["val_accuracy"][-1]:.3f}')
    return net, history



# トレーニング
criterion = nn.CrossEntropyLoss()
# opt = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
opt = optim.Adam(net.parameters(), lr=0.001)
net, history = train(net=net, opt=opt, criterion=criterion)


# 学習経過の描画
def plot_fig(history):
    plt.figure(1, figsize=(13,4))
    plt.subplots_adjust(wspace=0.5)

    # 学習曲線
    plt.subplot(1, 2, 1)
    plt.plot(history["loss"], label="train")
    plt.plot(history["val_loss"], label="valid")
    plt.title("train and valid loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid()

    # 精度表示
    plt.subplot(1, 2, 2)
    plt.plot(history["accuracy"], label="train")
    plt.plot(history["val_accuracy"], label="valid")
    plt.title("train and valid accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.grid()

    plt.show()

plot_fig(history=history)

# 画像を受け取って、分類結果を返す処理を行うコード
def predict(image_tensor, model, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor.unsqueeze(0))  # バッチ次元追加
        _, predicted = torch.max(output, 1)
    return predicted.item()

# モデルを保存する
timestamp = datetime.now().strftime("%m%d%H%M")  # 09181508 のような形式
filename = f"cifar10_model_{timestamp}.pth"

torch.save(net.state_dict(), filename)
print(f"モデルを保存しました: {filename}")


