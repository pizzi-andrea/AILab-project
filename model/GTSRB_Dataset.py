#Various imports
from locale import atoi
from pathlib import Path
import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import v2 
import cv2 as cv
from __global__ import *

#defining class for the GTSRB dataset
class GTSRB_Dataset(Dataset):
    
    def __init__(self,labels_path:Path, imgs_dir:Path,transform:v2.Compose = None) -> None:
        super().__init__()
        self.imgs_dir = imgs_dir #setting the path of the images directory
        self.transform = transform #setting transform
        self.labels = pd.read_csv(labels_path) #reading all labels
        self.classes = [ v for v in  range(0, 42)] # getting all classes

    def __len__(self) -> int: #return the number of labels in the dataset
        return len(self.labels)
    
    def labels(self) -> list: #return the classes
        return self.classes

    def __getitem__(self, index:int) -> tuple[Tensor, int]: # type: ignore
        
        # create the imgs path
        imgs_path = Path.joinpath(self.imgs_dir,self.labels.iloc[index,7]) # type: ignore

        # read the image with the path
        image_bgr = cv.imread(imgs_path) # for opencv
        image = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
        
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


     
