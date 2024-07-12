from torchvision.transforms.transforms import ColorJitter, RandomRotation, RandomVerticalFlip
from utils import *
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
import torchvision.transforms.functional as F
import pathlib
from torchvision.io import read_image
import numpy as np 
import cv2

from args import argument_parser
args = argument_parser()

# create dataset class
class knifeDataset(Dataset):
    def __init__(self,images_df,mode="train"):
        self.images_df = images_df.copy()
        self.mode = mode
        self.transforms = self.build_transforms()

    def __len__(self):
        return len(self.images_df)


    def build_transforms(self):
        if self.mode == "train":
            # Apply different transformations based on args
            transform_list = [T.ToPILImage(),
                              T.Resize((args.resized_img_weight, args.resized_img_height))]
            
            if args.brightness > 0 or args.contrast > 0 or args.saturation > 0 or args.hue > 0:
                transform_list.append(T.ColorJitter(brightness=args.brightness,
                                                    contrast=args.contrast,
                                                    saturation=args.saturation,
                                                    hue=args.hue))
            
            if args.random_rotation > 0:
                transform_list.append(T.RandomRotation(degrees=(0, args.random_rotation)))
            
            if args.vertical_flip > 0:
                transform_list.append(T.RandomVerticalFlip(p=args.vertical_flip))
            
            if args.horizontal_flip > 0:
                transform_list.append(T.RandomHorizontalFlip(p=args.horizontal_flip))
            
            transform_list.extend([
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            return T.Compose(transform_list)

        elif self.mode == "val":
            return T.Compose([
                T.ToPILImage(),
                T.Resize((args.resized_img_weight, args.resized_img_height)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self,index):
        X,fname = self.read_images(index)
        if not self.mode == "test":
            labels = self.images_df.iloc[index].Label
        else:
            y = str(self.images_df.iloc[index].Id.absolute())
        
        X = self.transforms(X)
        return X.float(),labels, fname

    def read_images(self,index):
        row = self.images_df.iloc[index]
        filename = str(row.Id)
        filename_path = os.path.join(args.dataset_location, filename)
        im = cv2.imread(filename_path)[:,:,::-1]
        return im, filename


