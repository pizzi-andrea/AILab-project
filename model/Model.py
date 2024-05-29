import torch 
import collections
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from time import time
from tqdm.auto import tqdm
from torchmetrics.classification import MulticlassAccuracy



def trainModel(model: nn.Module, 
               dataloader: DataLoader,
               loss_fn: nn.Module, 
               optimizer: Optimizer,
               device: torch.device,
               epoch = 15):
    """
        Main loop for traning the model
    """
    since = time()
    model.to(device)
    if model.train(True).training:
        BATCH_SIZE = dataloader.batch_size
        ordered_dict["phase"] = "train"
    else:
        return 1

    metric_acc = MulticlassAccuracy(num_classes=43,validate_args=True)
    # Main loop for traning
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
            optimizer.zero_grad()
            
            # make prediction for X test data
            y_pred = model(X)

            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()

            # Calculate metrics

            running_loss += loss.item()

            ordered_dict["Acc"] = metric_acc(y_pred, y)
            ordered_dict["Loss"] = f"{loss.item():.4f}"

            pbar.set_postfix(ordered_dict=ordered_dict)
            
        
        # Print metrics after one epochs
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = metric_acc.compute()

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

                ordered_dict["Acc"] = metric_acc(y_pred, y)
                ordered_dict["Loss"] = f"{loss.item():.4f}"

                pbar.set_postfix(ordered_dict=ordered_dict)
                
            
            # Print metrics after one epochs
            epoch_loss = running_loss / len(dataloader)
            epoch_acc = metric_acc.compute()

            ordered_dict["Loss"] = f"{epoch_loss:.4f}"
            ordered_dict["Acc"] = f"{epoch_acc:.4f}"

            pbar.set_postfix(ordered_dict=ordered_dict)
            pbar.close()



    # Show final metrics
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))