import torch 
import os
import torch.utils
from torchvision import datasets, transforms
import collections
import time

# Import PyTorch
import torch
from torch.utils.data import DataLoader
from torch import nn

from tqdm.auto import tqdm

# Import matplotlib for visualization
import matplotlib.pyplot as plt

import torch.optim as optim
from torchvision.models import resnet50


dataset_dir = "../GTSRB"

h, w = 32, 32

BATCH_SIZE = 32
EPOCH = 10

transform_train_list = transforms.Compose([
        #transforms.RandomResizedCrop(size=128, scale=(0.75,1.0), ratio=(0.75,1.3333), interpolation=3), #Image.BICUBIC)
        transforms.Resize((h, w), interpolation=3),
        # transforms.Pad(10),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

image_dataset  = datasets.ImageFolder(os.path.join(dataset_dir, 'train'),
                                                transform_train_list)

dataset_size = len(image_dataset) 
train_dataloader = DataLoader(image_dataset, batch_size=BATCH_SIZE,
                    shuffle=True, num_workers=2, pin_memory=True,
                    prefetch_factor=2, persistent_workers=True) # 8 workers may work faster


# Setup device agnostic code
device =  "cuda" if torch.cuda.is_available()  else "cpu"

# Instantiate the network
model = resnet50(num_classes=43)
# print(model)


# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),  lr=0.005, momentum=0.9, weight_decay=5e-4)


def trainModel(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader,
               loss: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               epoch = 15,
               device: torch.device = device):
    
    since = time.time()
    model.to(device)

    for epoch in range(epoch):
        print("epoch {}".format(epoch))
        

        model.train(True)
        
        
        pbar = tqdm()
        pbar.reset(total=len(dataloader.dataset))
        ordered_dict = collections.OrderedDict(phase="", Loss="", Acc="")

        running_loss = 0.0
        running_corrects = 0.0

        for iter, data in enumerate(dataloader):

            X, y = data

            X = X.to(device)
            y = y.to(device)

            now_batch_size,c,h,w = X.shape
            pbar.update(now_batch_size)  
            if now_batch_size<BATCH_SIZE:
                continue

            optimizer.zero_grad()

            
            y_pred = model(X)
            loss = loss_fn(y_pred, y)

            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_corrects += float(torch.sum(y_pred.argmax(dim = 1) == y) / BATCH_SIZE) 


            ordered_dict["phase"] = "train"
            ordered_dict["Acc"] = f"{float(torch.eq(y, y_pred.argmax(dim = 1)).sum().item()/len(y_pred.argmax(dim = 1))):.4f}"
            ordered_dict["Loss"] = f"{loss.item():.4f}"

            pbar.set_postfix(ordered_dict=ordered_dict)
            #pbar.close()
            
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = running_corrects / len(dataloader)

        ordered_dict["phase"] = 'train'
        ordered_dict["Loss"] = f"{epoch_loss:.4f}"
        ordered_dict["Acc"] = f"{epoch_acc:.4f}"

        pbar.set_postfix(ordered_dict=ordered_dict)
        pbar.close()



    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

trainModel(model, train_dataloader, loss_fn, optimizer, EPOCH, device)
                


