#Various imports
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

#Defining the model needed for the training
def trainModel(model: nn.Module, 
               dataloader: DataLoader,
               loss_fn: nn.Module, 
               optimizer: Optimizer,
               device: torch.device):

    pbar = tqdm() #progress bar
    
    model.to(device) #passing the model to the device
    model.train(True) #setting the mode to training

    BATCH_SIZE = dataloader.batch_size #costant that contains the size of the batch
    metric_acc = torchmetrics.Accuracy(task='multiclass',num_classes=43).to(device) #initializing internal module state with 43 classes and passing it to the device
    ordered_dict = collections.OrderedDict(phase="", Loss="", Acc="") #defining the dict for values about phases ,loss and accuracy
    
    #init values for dict ordered_dict
    ordered_dict["Acc"] = 0
    ordered_dict["Loss"] = 0
    ordered_dict["phase"] = "train"
    
    # preparing the progress bar
    pbar.reset(total=len(dataloader.dataset) + 2)
    pbar.set_postfix(ordered_dict=ordered_dict)
    pbar.update()
    
    running_loss = 0.0 #inizializing value for the running loss

    for batch, data in enumerate(dataloader):

        X, y = data

        X = X.to(device) #sending the data for the training to the device
        y = y.to(device) #sending the result of the training to the device

        batch_s, _,_,_ = X.shape
        pbar.update(batch_s) #updating progress bar
        optimizer.zero_grad()
        
        
        y_pred = model(X) # make prediction for X test data

        loss = loss_fn(y_pred, y) #calculating the loss
        loss.backward()
        optimizer.step()

        
        # Calculating metrics
        running_loss += loss.item()

        ordered_dict["Acc"] = f"{metric_acc(y_pred, y).cpu().numpy():.4f}" #inserting accuracy in dictionary
        ordered_dict["Loss"] = f"{loss.item():.4f}" #inserting loss in dictionary

        pbar.set_postfix(ordered_dict=ordered_dict) 
        
    
    # Print metrics after one epochs
    epoch_loss = running_loss / len(dataloader) #calc of total loss of the epoch
    epoch_acc = metric_acc.compute().cpu().numpy() #calc of the total accuracy of the epoc

    ordered_dict["Loss"] = f"{epoch_loss:.4f}" #inserting accuracy in dictionary
    ordered_dict["Acc"] = f"{epoch_acc:.4f}" #inserting loss in dictionary
    pbar.set_postfix(ordered_dict=ordered_dict)

    pbar.close() #closing the progress bar

    return epoch_acc, epoch_loss


#Defining the model needed for the test
def testModel(model: nn.Module, 
               dataloader: DataLoader,
               loss_fn: nn.Module, 
               device: torch.device):
    
    since = time() # actual time spent for the epoch
    model.to(device) #passing the model the device
    
    metric_acc = MulticlassAccuracy(num_classes=43,validate_args=True).to(device) #initializing internal module state with 43 classes and passing it to the device
    ordered_dict = collections.OrderedDict(phase="", Loss="", Acc="") #defining the dict for values about phases ,loss and accuracy

    y_tot_pred = torch.empty(0) #creating a tensor for the predictions
    y_tot_label = torch.empty(0) #creating a tensor for the matched labels
    pbar = tqdm()   #progress bar
    
    
    # Main loop for traning
    with torch.no_grad():
        pbar.reset(total=len(dataloader.dataset)) #resetting the progress bar at the start  

        #default values for the dictionary
        ordered_dict["Acc"] = 0 
        ordered_dict["Loss"] = 0
        ordered_dict["phase"] = "test"

        pbar.set_postfix(ordered_dict=ordered_dict)
        
        running_loss = 0.0 #initializing total loss

        for batch, data in enumerate(dataloader):
            
            X, y = data #getting the data from dataloader
            X = X.to(device) #passing the test data to the device
            y = y.to(device) #passing the result of the test to the device
            batch_s, _,_,_ = X.shape
            pbar.update(batch_s)  # updating progress bar
            
            # make prediction for X test data
            y_pred = model(X)

            loss = loss_fn(y_pred, y) #calculating the loss

            y_tot_pred = torch.cat( (y_tot_pred, torch.argmax(y_pred.cpu(), dim=1)) ) #tensor for matched values
            y_tot_label = torch.cat( (y_tot_label, y.cpu()) ) #tensor for matched labels


            # Calculate metrics
            running_loss += loss.item() # adding the loss

            ordered_dict["Acc"] = f"{metric_acc(y_pred, y).cpu().numpy():.4f}" #inserting the actual accuracy in the dictionary
            ordered_dict["Loss"] = f"{loss.item():.4f}" #inserting the actual loss in the dictionary

            pbar.set_postfix(ordered_dict=ordered_dict)
            

        # Print metrics after one epochs
        epoch_loss = running_loss / len(dataloader) #calc the total loss for the epoc
        epoch_acc = metric_acc.compute().cpu().numpy()  #calc the toal accuracy for the epoch

        ordered_dict["phase"] = "test" 
        ordered_dict["Loss"] = f"{epoch_loss:.4f}"  #inserting total loss in the dictionary
        ordered_dict["Acc"] = f"{epoch_acc:.4f}" #inserting total accuracy in the dictionary

        pbar.set_postfix(ordered_dict=ordered_dict)

    pbar.close() #closing progress bar

    return epoch_acc, epoch_loss, y_tot_pred, y_tot_label




   


    