import torch 
import collections
from torch import nn
import torchmetrics
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from time import time
from tqdm.auto import tqdm
from torchmetrics.classification import MulticlassAccuracy
import torch



def trainModel(model: nn.Module, 
               dataloader: DataLoader,
               loss_fn: nn.Module, 
               optimizer: Optimizer,
               device: torch.device):
    """
        Main loop for traning the model
    """

    pbar = tqdm()
    
    model.to(device)
    model.train(True)

    BATCH_SIZE = dataloader.batch_size
    metric_acc = torchmetrics.Accuracy(task='multiclass',num_classes=43).to(device)
    ordered_dict = collections.OrderedDict(phase="", Loss="", Acc="")
    ordered_dict["Acc"] = 0
    ordered_dict["Loss"] = 0
    ordered_dict["phase"] = "train"

    # Main loop for traning
    pbar.reset(total=len(dataloader.dataset) + 2)

    pbar.set_postfix(ordered_dict=ordered_dict)
    pbar.update()
    
    running_loss = 0.0

    for batch, data in enumerate(dataloader):

        X, y = data

        X = X.to(device)
        y = y.to(device)

        batch_s, _,_,_ = X.shape
        pbar.update(batch_s) 
        optimizer.zero_grad()
        
        # make prediction for X test data
        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        # Calculate metrics

        running_loss += loss.item()

        ordered_dict["Acc"] = f"{metric_acc(y_pred, y).cpu().numpy():.4f}"
        ordered_dict["Loss"] = f"{loss.item():.4f}"

        pbar.set_postfix(ordered_dict=ordered_dict)
        
    
    # Print metrics after one epochs
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = metric_acc.compute().cpu().numpy()

    ordered_dict["Loss"] = f"{epoch_loss:.4f}"
    ordered_dict["Acc"] = f"{epoch_acc:.4f}"
    pbar.set_postfix(ordered_dict=ordered_dict)

    pbar.close()


def testModel(model: nn.Module, 
               dataloader: DataLoader,
               loss_fn: nn.Module, 
               device: torch.device):
    """
        Main loop for testing the model
    """
    since = time()
    model.to(device)
    
    metric_acc = MulticlassAccuracy(num_classes=43,validate_args=True).to(device)
    ordered_dict = collections.OrderedDict(phase="", Loss="", Acc="")

    y_tot_pred = torch.empty(0)
    y_tot_label = torch.empty(0)
    empty_tensor = torch.empty(0)
    pbar = tqdm()
    # Main loop for traning
    with torch.no_grad():
        pbar.reset(total=len(dataloader.dataset))

        ordered_dict["Acc"] = 0
        ordered_dict["Loss"] = 0
        ordered_dict["phase"] = "test"

        pbar.set_postfix(ordered_dict=ordered_dict)
        
        running_loss = 0.0

        for batch, data in enumerate(dataloader):
            
            X, y = data
            X = X.to(device)
            y = y.to(device)
            batch_s, _,_,_ = X.shape
            pbar.update(batch_s)  
            
            # make prediction for X test data
            y_pred = model(X)

            loss = loss_fn(y_pred, y)

            y_tot_pred = torch.cat( (y_tot_pred, torch.argmax(y_pred.cpu(), dim=1)) )
            y_tot_label = torch.cat( (y_tot_label, y.cpu()) )

            # Calculate metrics

            running_loss += loss.item()

            
            ordered_dict["Acc"] = f"{metric_acc(y_pred, y).cpu().numpy():.4f}"
            ordered_dict["Loss"] = f"{loss.item():.4f}"

            pbar.set_postfix(ordered_dict=ordered_dict)
            

        # Print metrics after one epochs
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = metric_acc.compute().cpu().numpy() 

        ordered_dict["phase"] = "test"
        ordered_dict["Loss"] = f"{epoch_loss:.4f}"
        ordered_dict["Acc"] = f"{epoch_acc:.4f}"

        pbar.set_postfix(ordered_dict=ordered_dict)

    pbar.close()

    return y_tot_pred, y_tot_label




   


    