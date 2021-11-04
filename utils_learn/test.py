import torch
import torchvision
from PIL import Image
from torch import nn

img_path = "../imgs/000.png"
image = Image.open(img_path)
print(image)
#要加image=image.convert('RGB')因为png格式是四个通道，除了RGB三通道外，还有一个透明度通道，所以调用以上函数，保留其颜色通道，当然如果图片片本来就是三个颜色通道，经过此操作，不变，加上这一步后可以适用各种图片
image=image.convert('RGB')
#发现大小是180x187，但是模型需要32x32，需要改变
transform=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),
                                          torchvision.transforms.ToTensor()])
image = transform(image)
print(image.shape)



class My(nn.Module):
    def __init__(self):
        super(My, self).__init__()
        self.model1=nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64,10)
        )
    def forward(self,x):
        x=self.model1(x)
        return x

image = torch.reshape(image, (1,3,32,32))

# 注意在gpu上面训练的模型要想单纯的在cpu上面跑，要指明，如下
model = torch.load("my_0.pth",map_location=torch.device('cpu'))
print(model)
model.eval()#把模型转变为测试类
with torch.no_grad():
    output = model(image)

print(output)

print(output.argmax(1))
