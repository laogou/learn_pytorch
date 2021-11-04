import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset=torchvision.datasets.CIFAR10("../dataset",train=False, download=True, transform=torchvision.transforms.ToTensor())
datasetloader=DataLoader(dataset, batch_size=64)
input=torch.tensor([[1,2,0,3,1],
                    [0,1,2,3,1],
                    [1,2,1,0,0],
                    [5,2,3,1,1],
                    [2,1,0,1,1]], dtype=torch.float32)


# 这是第一部分
# input=torch.reshape(input,(-1,1,5,5))
# print(input.shape)

class My(nn.Module):
    def __init__(self):
        super(My,self).__init__()
        self.maxpool1=MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self,input):
        output=self.maxpool1(input)
        return output
# my=My()
# output=my(input)
# print(output)

# 这是第二部分
my = My()
writer = SummaryWriter("../logs")
step=0
for data in datasetloader:
    imgs, targets = data
    writer.add_images("input",imgs,step)
    writer.add_images("output",my(imgs),step)
    step+=1