from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
#
# # python的用法-》 tensor数据类型，要通过transform对一些图片等进行转换为tensor数据类型  tensor的中文翻译为张量
# # 通过transoms.totensor去解决两个问题  1是transform如何使用，2为什么需要tensor数据类型
# # Totensor()方法是将一个PIL护着numpy.ndarray的数据类型转换为tensor类型
#
# img_path=r"D:\Environment\support\pytorchTrain\train\ants_image\0013035.jpg"
# img=Image.open(img_path)
#
#
# writer=SummaryWriter("logs")
#
# # 问题1
# tensor_trans=transforms.ToTensor()
# tensor_img=tensor_trans(img)
#
# print(tensor_img)
#
# # 为什么要使用张量呢？
# writer.add_image("Tensor_img",tensor_img)
# writer.close()

# 常用transfrom
img_path=r"C:\Users\Administrator\Desktop\info\1123.jpg"
img=Image.open(img_path)
writer=SummaryWriter("../logs")
trans_totensor=transforms.ToTensor()
img_tensor=trans_totensor(img)
writer.add_image("Totensor", img_tensor)


# 归一化（Normalize）
print(img_tensor[0][0][0]) #归一化之前地内容，输出进行对比
trans_norm=transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])  #归一化操作初试
img_norm=trans_norm(img_tensor) #进行归一化操作
print(img_norm[0][0][0]) #归一化之前地内容，输出进行对比
writer.add_image("Normalize",img_norm)
print(img.size)

# Resize

trans_resize=transforms.Resize((512,512))
# img PIL -> resize->img_resize PIL
img_resize=trans_resize(img)
# img_resize PIL-> totensro ->img_resize tensor
img_resize=trans_totensor(img_resize)
writer.add_image("resize", img_resize, 0)
print(img_resize)


# Compose - resize-2
trans_resize_2 = transforms.Resize(512)
# PIL ->PIL -> tensor
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
img_resize_2 = trans_compose(img)
writer.add_image("resize2",img_resize_2,1)

# RandomCrop 随机裁剪
trans_random=transforms.RandomCrop((500,100))
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCropHW", img_crop, i)

writer.close()