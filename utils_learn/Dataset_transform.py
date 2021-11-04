import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_set =torchvision.datasets.CIFAR10(root="../dataset", train=True, transform=dataset_transform, download=True)
test_set =torchvision.datasets.CIFAR10(root="../dataset", train=False, transform=dataset_transform, download=True)


# print(test_set[0]) #第0个样本
# print(test_set.classes)  #包含的属性
# img, target = test_set[0]  #获取样本图片和标签
# print(img)  #输出图片
# print(target) #输出标签
#
# print(test_set.classes[target]) #输出属性中编号为3的具体属性名
# img.show()
#

# 这个原始图片是PIL Image，要给pytorch使用，就要转化为tensor类型，于是有了第三行代码
print(test_set[0])
# 这里经过print查看后确认已经转化为tensor类型，就可以用tensorboard进行查看
writer=SummaryWriter("../logs")
# 显示10中图片
for i in range(10):
    img, target =test_set[i]
    writer.add_image("test_set", img, i)

writer.close()