import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("../dataset", train=False, transform=torchvision.transforms.ToTensor(),download=True)
dataloader = DataLoader(dataset, batch_size=64, drop_last=True)


class My(nn.Module):

    def __init__(self):
        super(My, self).__init__()
        self.linear1 = Linear(196608,10)
    def forward(self,input):
        output = self.linear1(input)
        return output

my = My()


for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    # output = torch.reshape(imgs, (1,1,1,-1)) #reshape的功能更加强大，下面的latten只能变成一行。
    output =torch.flatten(imgs)
    print(output.shape)
    output=my(output)
    print(output.shape)