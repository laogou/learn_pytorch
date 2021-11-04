import torch


#方式1加载模型
import torchvision.models

model = torch.load("vgg16_method1.pth")
#方式2加载模型
vgg16=torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
print(vgg16)
