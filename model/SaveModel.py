#Various imports
from os import makedirs, listdir
from typing import Any
import torch
import torch.nn as nn
from pathlib import Path
from numpy import float32
import pandas as pd

#Defining 
def SaveModel(weights: dict[str, Any], path: Path, acc_history_train:list[float32] = None, loss_history_train: list[float32] = None, acc_history_test: list[float32] = None, loss_history_test: list[int] = None, epochs: list[int] = None):
    if not path.exists(): # creating the directory if is not already created
        makedirs(path)

    #DataFrame for the loss to write on the csv
    df_loss = pd.DataFrame(data= {
            'epochs'         : epochs,
            'loss_train'     : loss_history_train,
            'loss_test'      : loss_history_test
            
    })
    
     #DataFrame for the accuracy to write on the csv
    df_acc = pd.DataFrame(data= {
            'epochs'         : epochs,
            'accuracy_train' : acc_history_train,
            'accuracy_test'  : acc_history_test,
            
    })


    df_loss.to_csv(path.joinpath('history_loss.csv'), index=False) #sending data about loss to the csv
    df_acc.to_csv(path.joinpath('history_acc.csv'), index = False) #sending data about accuracy to the csv
    torch.save(weights, path.joinpath('model_weights.pth')) #saving the file on the disk
    

    



        


