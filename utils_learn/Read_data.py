from torch.utils.data import Dataset
from PIL import Image
import os




#数据集加载
class MyData(Dataset):
    #为这个提供一个全局变量等
    def __init__(self, root_dir, label_dir):
        self.root_dir=root_dir
        self.label_dir=label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)


    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img=Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)


root_dir = r"D:\Environment\support\pytorchTrain\train"
ants_label_dir = r"ants_image"
bees_label_dir = r"bees_image"
ants_dataset = MyData(root_dir,ants_label_dir)
bees_dataset = MyData(root_dir,ants_label_dir)

train_dataset = ants_dataset + bees_dataset
