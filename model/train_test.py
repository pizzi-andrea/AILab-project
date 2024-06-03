import torch 
import collections
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from time import time
from tqdm.auto import tqdm
from torchmetrics.classification import MulticlassAccuracy
from Model import ModelCNN
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
from GTSRB_Dataset import GTSRB_Dataset as Dataset
from __global__ import *
import torch


def trainModel(model: nn.Module, 
               dataloader: DataLoader,
               loss_fn: nn.Module, 
               optimizer: Optimizer,
               device: torch.device,
               batch_size,
               epoch = 15):
    """
        Main loop for traning the model
    """
    since = time()
    model.to(device)

    metric_acc = MulticlassAccuracy(num_classes=43,validate_args=True)
    # Main loop for traning
    for epoch in range(epoch):
        print("epoch {}".format(epoch))
        
        
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

            optimizer.zero_grad()

            
            y_pred = model(X)
            loss = loss_fn(y_pred, y)

            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_corrects += float(torch.sum(y_pred.argmax(dim = 1) == y) / batch_size) 


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



    # Show final metrics
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

def testModel(model: nn.Module, 
               dataloader: DataLoader,
               loss_fn: nn.Module, 
               device: torch.device,
               epoch = 15):
    """
        Main loop for testing the model
    """
    since = time()
    model.to(device)
    
    metric_acc = MulticlassAccuracy(num_classes=43,validate_args=True)
    # Main loop for traning
    with torch.no_grad():
        for epoch in range(epoch):
            print("epoch {}".format(epoch))
            
            pbar = tqdm()
            pbar.reset(total=len(dataloader.dataset))
            ordered_dict = collections.OrderedDict(phase="", Loss="", Acc="")
            
            running_loss = 0.0

            for batch, data in enumerate(dataloader):

                X, y = data

                X = X.to(device)
                y = y.to(device)

                pbar.update(batch)  
                
                # make prediction for X test data
                y_pred = model(X)

                loss = loss_fn(y_pred, y)

                # Calculate metrics

                running_loss += loss.item()

                ordered_dict["phase"] = "train"
                ordered_dict["Acc"] = metric_acc(y_pred, y)
                ordered_dict["Loss"] = f"{loss.item():.4f}"

                pbar.set_postfix(ordered_dict=ordered_dict)
                
            
            # Print metrics after one epochs
            epoch_loss = running_loss / len(dataloader)
            epoch_acc = metric_acc.compute()

            ordered_dict["phase"] = "train"
            ordered_dict["Loss"] = f"{epoch_loss:.4f}"
            ordered_dict["Acc"] = f"{epoch_acc:.4f}"

            pbar.set_postfix(ordered_dict=ordered_dict)
            pbar.close()


    # Show final metrics
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    

if __name__ == '__main__':

    EPOCH = 10
    seq = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((32, 32), interpolation=InterpolationMode.NEAREST_EXACT),
        v2.RandomHorizontalFlip(),
        v2.RandomAutocontrast(p=1.0),
        #v2.RandomEqualize()
        
    ])

    dataset_train = Dataset(labels_path=LABELS_PATH_TRAIN, imgs_dir=IMGS_PATH_TRAIN, transform=seq)
    dataset_test = Dataset(labels_path=LABELS_PATH_TEST, imgs_dir=IMGS_PATH_TEST, transform=seq)


    loader_train = DataLoader(dataset_train, batch_size=64, shuffle=True, num_workers=3)
    loader_test = DataLoader(dataset_test, batch_size=64, shuffle=False, num_workers=3)

    model = ModelCNN(input_shape=3, hidden_units=64, output_shape=43)
    device =  "cuda" if torch.cuda.is_available()  else "cpu"

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(),  lr=0.005, momentum=0.9, weight_decay=5e-4)

    #trainModel(model, loader_train, loss_fn, optimizer, device, 64)
    testModel(model, loader_test, loss_fn, device)