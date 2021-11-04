import torch.optim
import torchvision.datasets
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

path="../dataset"
dataset = torchvision.datasets.CIFAR10(path, train=False, transform=torchvision.transforms.ToTensor(), download=True)
dataloader=DataLoader(dataset, batch_size=1)
class My(nn.Module):
    def __init__(self):
        super(My, self).__init__()
        self.model1=Sequential(
            Conv2d(3,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64,10)
        )
    def forward(self,x):
        x=self.model1(x)
        return x

loss=nn.CrossEntropyLoss()
my=My()
optim=torch.optim.SGD(my.parameters(),lr=0.01)

for epoch in range(20): #每一轮
    running_loss=0.0
    for data in dataloader:
        imgs,targets=data
        outputs=my(imgs)
        result_loss=loss(outputs,targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        print(result_loss)
        running_loss=running_loss+result_loss
    print(running_loss)