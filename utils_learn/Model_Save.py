import torch
import torchvision.models

vgg16=torchvision.models.vgg16(pretrained=False)

# 保存方式1
torch.save(vgg16, "vgg16_method1.pth")


#保存方式2:只保存网络模型的参数，官方推荐
torch.save(vgg16.state_dict(),"vgg16_method2.pth")

#陷阱：采用方式1保存模型时，当需要加载的时候，我们需要在加载页进行定义的声明（from my import My）加载类
