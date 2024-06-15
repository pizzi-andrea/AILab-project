'''
Dataset structure example

imgs -> folder that contains the images
labels.csv -> file that containts the labels for the images

labels.csv:
img1.jpg,0
img2.jpg,1
img3.jpg,2
img4.jpg,0

0 = dog
1 = cat
2 = car
3 = airplane
'''
from locale import atoi
from pathlib import Path
import pandas as pd
from torch import Tensor
from torchvision.io import read_image
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision.transforms import v2 
from torch.nn import Module
import torch
from __global__ import *

class GTSRB_Dataset(Dataset):
    
    def __init__(self,labels_path:Path, imgs_dir:Path,transform:v2.Compose = None) -> None:
        super().__init__()
        self.imgs_dir = imgs_dir
        self.transform = transform
        self.labels = pd.read_csv(labels_path)
        self.classes = [ v for v in  range(0, 42)]

    def __len__(self) -> int:
        return len(self.labels)
    
    def labels(self) -> list:
        return self.classes

    def __getitem__(self, index:int) -> tuple[Tensor, int]: # type: ignore
        # create the imgs path

        imgs_path = Path.joinpath(self.imgs_dir,self.labels.iloc[index,7]) # type: ignore

        # read the image with the path
        image = read_image(imgs_path) # cv2.imread(imgs_path) for opencv # type: ignore

        # read the label
        label = self.labels.iloc[index,6]

        # apply transformations
        if self.transform:
            out = self.transform(image)
        else:
            out = image

        # return the pair image,label
        return out ,label # type: ignore

if __name__ == '__main__':
    dataset_train = GTSRB_Dataset(labels_path=LABELS_PATH_TRAIN, imgs_dir=IMGS_PATH_TRAIN)
    dataset_test  = GTSRB_Dataset(labels_path=LABELS_PATH_TEST, imgs_dir=IMGS_PATH_TEST)


     
