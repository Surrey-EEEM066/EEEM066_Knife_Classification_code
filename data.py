from torchvision.transforms.transforms import ColorJitter, RandomRotation, RandomVerticalFlip
from utils import *
from config import *
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
import torchvision.transforms.functional as F
import pathlib
from torchvision.io import read_image
import numpy as np 
import cv2

# create dataset class
class knifeDataset(Dataset):
    def __init__(self,images_df,mode="train"):
        self.images_df = images_df.copy()
        self.images_df.Id = self.images_df.Id
        self.mode = mode

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self,index):
        X,fname = self.read_images(index)
        if not self.mode == "test":
            labels = self.images_df.iloc[index].Label
        else:
            y = str(self.images_df.iloc[index].Id.absolute())
        if self.mode == "train":
            X = T.Compose([T.ToPILImage(),
                    T.Resize((config.img_weight,config.img_height)),
                    T.ColorJitter(brightness=0.2,contrast=0,saturation=0,hue=0),
                    T.RandomRotation(degrees=(0, 180)),
                    T.RandomVerticalFlip(p=0.5),
                    T.RandomHorizontalFlip(p=0.5),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(X)
        elif self.mode == "val":
            X = T.Compose([T.ToPILImage(),
                    T.Resize((config.img_weight,config.img_height)),
                    T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(X)
        return X.float(),labels, fname

    def read_images(self,index):
        row = self.images_df.iloc[index]
        filename = str(row.Id)
        im = cv2.imread(filename)[:,:,::-1]
        return im, filename


