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
               epoch = 15):
    """
        Main loop for traning the model
    """
    ordered_dict = dict()
    since = time()
    pbar = tqdm()
    
    model.to(device)
    model.train(True)
    BATCH_SIZE = dataloader.batch_size
        


    metric_acc = MulticlassAccuracy(num_classes=43,validate_args=True)
    ordered_dict = collections.OrderedDict(phase="", Loss="", Acc="")
    # Main loop for traning
    for epoch in range(epoch):
        print("epoch {}".format(epoch + 1))
        pbar.reset(total=len(dataloader.dataset))
        ordered_dict["Acc"] = 0
        ordered_dict["Loss"] = 0
        ordered_dict["phase"] = "train"
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

            ordered_dict["Acc"] = metric_acc(y_pred, y).numpy()
            ordered_dict["Loss"] = f"{loss.item():.4f}"

            pbar.set_postfix(ordered_dict=ordered_dict)
            
        
        # Print metrics after one epochs
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = metric_acc.compute().numpy()

        ordered_dict["Loss"] = f"{epoch_loss:.4f}"
        ordered_dict["Acc"] = f"{epoch_acc:.4f}"
        pbar.set_postfix(ordered_dict=ordered_dict)

    pbar.close()



    # Show final metrics
    time_elapsed = time() - since
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
    ordered_dict = collections.OrderedDict(phase="", Loss="", Acc="")

    pbar = tqdm()
    # Main loop for traning
    with torch.no_grad():
        for epoch in range(epoch):
            print("epoch {}".format(epoch + 1))
            #pbar.reset(total=19_503)
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

                # Calculate metrics

                running_loss += loss.item()

                
                ordered_dict["Acc"] = metric_acc(y_pred, y).numpy()
                ordered_dict["Loss"] = f"{loss.item():.4f}"

                pbar.set_postfix(ordered_dict=ordered_dict)
                
    
            # Print metrics after one epochs
            epoch_loss = running_loss / len(dataloader)
            epoch_acc = metric_acc.compute().numpy()

            ordered_dict["phase"] = "test"
            ordered_dict["Loss"] = f"{epoch_loss:.4f}"
            ordered_dict["Acc"] = f"{epoch_acc:.4f}"

            pbar.set_postfix(ordered_dict=ordered_dict)

        pbar.close()


    # Show final metrics
    time_elapsed = time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    

if __name__ == '__main__':

    EPOCH = 3
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

    trainModel(model, loader_train, loss_fn, optimizer, device, epoch=EPOCH)
    testModel(model, loader_test, loss_fn, device, epoch=EPOCH)