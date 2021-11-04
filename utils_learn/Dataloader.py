import torchvision
from torch.utils.data import DataLoader

# 准备的测试机
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("../dataset", train=False, transform=torchvision.transforms.ToTensor())
test_load=DataLoader(dataset=test_data, batch_size=4, shuffle=True, num_workers=0, drop_last=False)  # 设置了随机抓取
img, target = test_data[0]
# 测试数据集中第一张
print(img.shape)
print(target)

writer=SummaryWriter("Dataloader")
step=0
# 要循环进行
for epoch in range(2):
    for data in test_load:
        imgs, targets =data
        writer.add_images("Epoch:{}".format(epoch), imgs,step)
        step=step+1

writer.close()

