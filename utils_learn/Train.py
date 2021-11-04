import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils_learn.Model import My


#准备数据集
train_data = torchvision.datasets.CIFAR10(root="../dataset", train=True,transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10(root="../dataset", train=False, transform=torchvision.transforms.ToTensor(), download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))

#dataloader加载数据集
train_dataloader  = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

#引入网络模型
my = My()

#创建损失函数
loss_fn = nn.CrossEntropyLoss()

#优化器
learning_rate=1e-2
optimizer = torch.optim.SGD(my.parameters(), lr=learning_rate)

#添加tensorboard
writer = SummaryWriter("../logs")

#设置训练网络的以下参数
#记录训练的次数
total_train_step=0
#记录测试次数
total_test_step=0
#训练轮数
epoch=10
for i in range(epoch):
    print("------第{}轮训练开始------".format(i+1))
    for data in train_dataloader:
        #开始训练
        imgs, targets = data
        output = my(imgs)
        loss = loss_fn(output, targets)

        #优化器模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        total_trian_step=total_train_step+1
        if total_trian_step%100 ==0:
            print("训练次数：{},loss:{}".format(total_trian_step,loss.item()))
            writer.add_scalar("train_loss", loss.item, total_trian_step)

    #以测试数据集的正确率来评估模型训练
    #要求整个数据集上的loss：设置一个变量
    total_test_loss=0
    total_accuracy=0 #整体正确
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets=data
            outputs = my(imgs)
            loss=loss_fn(outputs, targets)
            total_test_loss=total_test_loss+loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy+=accuracy
    print("整体数据集上的loss:{}".format(total_test_loss))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    total_test_step+=1

    torch.save(my, "my_{}.pth".format(i))
    print("模型已保存")

    print("整体测试集上的正确率：{}".format(total_accuracy/total_test_step))
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
writer.close()

# 要在训练步骤开始的时候，进入训练状态
my.train()
# 要在训练步骤结束的时候
my.eval()