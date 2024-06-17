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
import cv2 as cv
from __global__ import *
    
class GTSDB_Dataset(Dataset):
    def __init__(self, csv_path: Path, root_dir: Path, transform:v2.Compose = None):
        self.dataframe = pd.read_csv(csv_path, delimiter=';')
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = self.dataframe.iloc[:, 0].unique()
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        
        img_name = self.image_files[idx]
        
        img_path = self.root_dir.joinpath(img_name)

        image = cv.imread(img_path)
        
        # records = self.dataframe.where(self.dataframe.iloc[:,0] == img_name)
        records = self.dataframe[self.dataframe.iloc[:,0] == img_name]
    
        boxes = records.iloc[:, [1, 2, 3, 4]].values
        
        labels = records.iloc[:, 5].values
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        if self.transform:
            out = self.transform(image)
        else:
            out = image
        
        target = {
            'boxes'  : boxes,
            'labels' : labels
        }

        
        if self.transform:
            out = self.transform(image)
        else:
            out = image
        
        return out, target