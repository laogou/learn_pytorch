from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
writer = SummaryWriter("../logs") #实例化一个类，执行后面的函数之后会生成一个log文件夹，里面就是tensorbard生成的图表，
# 打开生成的图表命令是在conda控制台：tensorboard --logdir=logs --port=6007，默认打开6006端口，可以自行指定，即logdir=事件文件所在文件夹名
# writer.add_image() #添加一个图片方法

image_path = r"D:\Environment\support\pytorchTrain\train\ants_image\0013035.jpg"
img = Image.open(image_path)
img_array = np.array(img)

writer.add_image("test", img_array,1,dataformats='HWC')
#例子：绘制y=2x的图像
for i in range(100):
    writer.add_scalar("y=2x",2*i,i)

# writer.add_scalar() #添加一个数 X轴是global_step，y轴是scala_value
writer.close()

#利用opencv读取图片，获得numpy型图片数据
