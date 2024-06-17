#Various imports
from os import makedirs, listdir
from typing import Any
import torch
import torch.nn as nn
from pathlib import Path
from numpy import float32
import pandas as pd

#Defining
def SaveModel(weights: dict[str, Any], path: Path, acc_history:list[float32] = None, loss_history: list[float32] = None, epochs: list[int] = None, override: bool = False):
    if not path.exists(): # creating the directory if is not already created
        makedirs(path)

    

    #DataFrame for the data to write on the csv
    df = pd.DataFrame(data= {
            'epochs'   : epochs,
            'accuracy' : acc_history,
            'loss'     : loss_history,
            
    })

    df.to_csv(path.joinpath('history.csv')) #sending data about epochs, accuracy and loss to the csv
    torch.save(weights, path.joinpath('model_weights.pth')) #saving the file on the disk
    

    



        


