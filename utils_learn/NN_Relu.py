import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1,-0.5],
                      [-1,3]])
input = torch.reshape(input,(-1,1,2,2))
print(input.shape)

dataset=torchvision.datasets.CIFAR10("../dataset", train=False, download=True, transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)

class My(nn.Module):
    def __init__(self):
        super(My,self).__init__()
        self.relu1=ReLU()
        self.sigmoid1 = Sigmoid()
    def forward(self, input):
        output = self.sigmoid1(input)
        return output

my = My()
step =0
writer = SummaryWriter("../logs")
for data in dataloader:
    imgs, targets = data
    writer.add_images("input2", imgs, global_step=step)
    output = my(imgs)
    writer.add_images("output2", output, step)
    step+=1

writer.close()