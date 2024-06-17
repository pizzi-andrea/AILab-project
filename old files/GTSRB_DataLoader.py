#Various imports for the code
import torch 
import os
import torch.utils
from torchvision import datasets, transforms
import collections
import time

from GTSRB_Dataset import GTSRB_Dataset 

# Import PyTorch
from torch.utils.data import DataLoader
from torch import nn

from tqdm.auto import tqdm

# Import matplotlib for visualization
import matplotlib.pyplot as plt

import torch.optim as optim
from torchvision.models import resnet50


dataset_dir = "../GTSRB" # defining the path of the dataset directory

h, w = 32, 32 #defining height and width of images

BATCH_SIZE = 32 # number of batches
EPOCH = 10  # number of epochs to do

#defining the trasformations that the images used for training will undergo before being used
transform_train_list = transforms.Compose([
        #transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((h, w), interpolation=3), # resizing the images to the h,w sizes
        # transforms.Pad(10),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # trasforming to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #normalizing with the set values
        ])

#defining the trasformations that the images used for the test will undergo before being used
transform_test_list = transforms.Compose([
        #transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((h, w), interpolation=3), # resizing the images to the h,w sizes
        # transforms.Pad(10),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # trasforming to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #normalizing with the set values
])

image_dataset_train = datasets.ImageFolder(os.path.join(dataset_dir, 'Train'),
                                                transform_train_list)

image_dataset_test = GTSRB_Dataset("../GTSRB/Test.csv",
                                    dataset_dir,
                                                transform_train_list)

train_dataset_size = len(image_dataset_train) 
test_dataset_size = len(image_dataset_test)

train_dataloader = DataLoader(image_dataset_train, batch_size=BATCH_SIZE,
                    shuffle=True, num_workers=2, pin_memory=True,
                    prefetch_factor=2, persistent_workers=True) # 8 workers may work faster

test_dataloader = DataLoader(image_dataset_test, batch_size=BATCH_SIZE) 


# Setup device agnostic code
device =  "cuda" if torch.cuda.is_available()  else "cpu"

# Instantiate the network
model = resnet50(num_classes=43)
# print(model)


# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),  lr=0.005, momentum=0.9, weight_decay=5e-4)





