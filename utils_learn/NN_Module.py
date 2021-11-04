import torch
from torch import nn


class MyNN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,input):
        output = input+1
        return output

my = MyNN()
x=torch.tensor(1.0)
output = my(x)
print(output)

