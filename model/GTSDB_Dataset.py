#various imports
from pathlib import Path
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import v2 
import torchvision
import torch
import cv2 as cv
from __global__ import *
from os import listdir
    
#class for the GTSDB dataset    
class GTSDB_Dataset(Dataset):
    def __init__(self, root_dir: Path, csv_path: Path = None, transform:v2.Compose = None):
        self.dataframe = pd.read_csv(csv_path, delimiter=';') if csv_path is not None else  None #reading data
        self.root_dir = root_dir #setting root directory of dataset
        self.transform = transform #setting transform
        self.image_files = self.dataframe.iloc[:, 0].unique()  if self.dataframe is not None else  listdir(root_dir) #setting images of dataset
        
    def __len__(self): #return the number of images in the dataset
        return  len(self.image_files)
    def __getitem__(self, idx: int):  #gets one image from the dataset via index
        
        if self.dataframe is None:
            image_bgr = cv.imread( self.root_dir.joinpath( self.image_files[idx] ))
            image = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
            
            if self.transform:
                out = self.transform(image) #trasform image if necessary
            else:
                out = image
                
            return out, self.image_files[idx]
        else:
            img_name = self.image_files[idx] #pick the image
            
            img_path = self.root_dir.joinpath(img_name) #get the path of the image

            image_bgr = cv.imread(img_path) #read the image
            image = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
            
            
            
            # records = self.dataframe.where(self.dataframe.iloc[:,0] == img_name)
            records = self.dataframe[self.dataframe.iloc[:,0] == img_name]
        
            boxes = records.iloc[:, [1, 2, 3, 4]].values #getting coordinates of the diagonal
            
            labels = records.iloc[:, 5].values #getting the label of image
            
            boxes = torch.as_tensor(boxes, dtype=torch.float32) # converting the boxes to tensor
            labels = torch.as_tensor(labels, dtype=torch.int64) # converting labels to tensor
            
            if self.transform:
                out = self.transform(image) #trasform image if necessary
            else:
                out = image
            
            #defining dictionary target with boxes, labels and image
            target = {
                'boxes'   : boxes,
                'labels'  : labels
            }
            return out, target