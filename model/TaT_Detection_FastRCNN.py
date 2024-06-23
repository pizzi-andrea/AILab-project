#Various imports
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2 as Net
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2 as weights 
from torchvision.transforms import v2
from GTSDB_Dataset import GTSDB_Dataset as Dataset
import torch
from torch.utils.data import DataLoader
from torch import nn
from os import listdir
from tqdm.auto import tqdm
import collections
from SaveModel import SaveModel
from PlotGrapichs import plot_grapich
from __global__ import *
import numpy as np
import cv2 as cv
import pandas as pd
import matplotlib.pyplot as plt
import pandas
import os
from torch.optim.lr_scheduler import StepLR

#setting default path for the saved model2
DEFAULT_PATH = Path('saved_model/FastRCNN')

#model for training 
def train_model(model: nn.Module, 
                dataloader : DataLoader, 
                optimizer  : nn.Module, 
                device: str = 'cpu',
                lr_scheduler = None):

    pbar = tqdm() #progress bar

    model.to(device) #passing the model to the device
    model.train(True) #setting the mode to training
    
    BATCH_SIZE = dataloader.batch_size #costant that contains the size of the batch
    ordered_dict = collections.OrderedDict(phase="", Loss="") #defining the dict for values about phases and loss

    #init values for dict ordered_dict
    ordered_dict["Loss"] = 0
    ordered_dict["phase"] = "train"

    # preparing the progress bar
    pbar.reset(total=len(dataloader.dataset) + 2)
    pbar.set_postfix(ordered_dict=ordered_dict)
    pbar.update()
    
    running_loss = 0.0 #inizializing value for the running loss
    

    for batch, (images, targets) in enumerate(dataloader):

        images = list(image.to(device) for image in images) # images in tensor format
        targets = [ {k: v.to(device) for k, v in t.items()} for t in targets] #dictionary with all targets in tensor format

        batch_s = len(images) #number of images
        pbar.update(batch_s) #updating progress bar

        model_dict_loss = model(images, targets) #creating model

        losses = sum(loss for loss in model_dict_loss.values()) 
        loss_value = losses.item() #getting the loss value

        optimizer.zero_grad() #resetting gradients
        losses.backward()
        optimizer.step()
        
        if lr_scheduler:
            lr_scheduler.step()

        running_loss += loss_value #summing the total loss
        ordered_dict["Loss"] = f"{loss_value:.4f}" #inserting loss in dictionary
        pbar.set_postfix(ordered_dict=ordered_dict) 
    
    epoch_loss = running_loss / len(dataloader) #calc of total loss of the epoch
    ordered_dict["Loss"] = f"{epoch_loss:.4f}" #inserting accuracy in dictionary
    pbar.set_postfix(ordered_dict=ordered_dict)

    return epoch_loss

#model for the test
def test_model(model: nn.Module, 
                dataloader : DataLoader, 
                th: int,
                device: str = 'cpu'):

    pbar = tqdm() #progress bar

    model.to(device) #passing the model to the device
    results=[]  
    model.eval() #setting model into evaluation mode
    
    BATCH_SIZE = dataloader.batch_size #costant that contains the size of the batch
    ordered_dict = collections.OrderedDict(phase="") #defining the dict for values about phases ,loss and accuracy

    #init values for dict ordered_dict
    ordered_dict["phase"] = "test"

    # preparing the progress bar
    pbar.reset(total=len(dataloader.dataset) + 2)
    pbar.set_postfix(ordered_dict=ordered_dict)
    pbar.update()
    
    running_loss = 0.0 #inizializing value for the running loss

    for batch, (images, id) in enumerate(dataloader):
        images = [ img.to(device) for img in images] #getting the tensor of all images
        batch_s = len(images) #getting len of images
        pbar.update(batch_s) #updating progress bar
        model_dict = model(images) 

        for idx, img in enumerate(images):
            boxes  = model_dict[idx]['boxes'].data.cpu().numpy() #putting the all boxes in a numpy array
            scores = model_dict[idx]['scores'].data.cpu().numpy() #putting all scores in a numpy array
            boxes = boxes[scores >= th].astype(np.int32) #Compare the score of output with the threshold and
            scores = scores[scores >= th]                    #slelect only those boxes whose score is greater
                                                                            # than threshold value  
            #appending results                                                                
            results.append({
                'img': id[0],
                'boxes': boxes,
                'scores': scores
            })

    return results
            
            
#saves the test results 
def saveTest(result: list, path_to_save: Path):
    if not path_to_save.exists(): #creates directory if is necessary
        os.makedirs(path_to_save, exist_ok=False)
    
    #saves the all of the result, one by one, in the png file
    for i, dict_r in enumerate(result):
        path_img = IMGS_PATH_TEST_GTSDB.joinpath(dict_r['img'])
        boxes = dict_r['boxes']
        scores = dict_r['scores']
        img = cv.imread(path_img)
        
        for box, score in zip(boxes, scores):
            cv.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0,255,0), 2)
            cv.putText(img, f'score: {score:.3f}', (box[0] - 10, box[1] - 10),  fontFace=cv.FONT_ITALIC, fontScale=0.4, color=(0,255,0))

        #tries writing, if error happens, it is notified via assert
        
        if not cv.imwrite(path_to_save.joinpath(str(i) + '.png'), img):
            assert 'Error: impossible write file in ' + path_to_save
    
        
# main
if __name__ == '__main__':

    EPOCH = 2 #defining number of epoch
    N_ESP = len (listdir(DEFAULT_PATH)) + 1 if DEFAULT_PATH.exists() else 0 
    model_weights = None

    #variables used for the data history
    loss_history_train = []
    epoch_history = []
    best_loss = None

    #defining the trasformation that the images will under go during training and test
    seq = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True), #converting image to Dtype
        v2.RandomAutocontrast(p=1.0), #applying contrast  
    ])

    #dataset used for training and test
    dataset_train = Dataset(IMGS_PATH_TRAIN_GTSDB, LABELS_PATH_TRAIN_GTSDB, transform=seq)
    dataset_test = Dataset(IMGS_PATH_TEST_GTSDB, transform=seq)
    model = Net() #defining the model
    #model.load_state_dict(torch.load(Path(DEFAULT_PATH.joinpath(f'ts{N_ESP - 1}/model_weights.pth'))))
    #model.load_state_dict(torch.load(Path(DEFAULT_PATH.joinpath(f'pesi_ok/model_weights.pth'))))

  
    loader_train = DataLoader(dataset_train, batch_size=3, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
    loader_test = DataLoader(dataset=dataset_test, batch_size=1, shuffle=True)
    
    device =  "cuda" if torch.cuda.is_available()  else "cpu" #selecting the device


    #defining the optimizer, we use adamW with a learning rate of 1e^10^-4 and a decay of 0
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, weight_decay=0.0005)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0)
    
    for epoch in range(1, EPOCH + 1):
        print("epoch {}".format(epoch))

        #training and getting the loss of the epoch
        epoch_loss = train_model(model, loader_train, optimizer, device)
        #test_model(model, loader_test, 0.45, device)

        #updating the best loss if necessary
        if best_loss is None or best_loss > epoch_loss: 
            best_loss = epoch_loss
            loss_history_train.append(best_loss)
            epoch_history.append(epoch)
            model_weights = model.state_dict()

    #loading weights if is necessary
    if model_weights is None and N_ESP > 0:
        model_weights = torch.load(Path(DEFAULT_PATH.joinpath(f'ts{N_ESP - 1}/model_weights.pth')))
    else:
        model_weights = model.state_dict()
    
    #loading the weights into the testing model
    model.load_state_dict(model_weights, strict=True)
    r = test_model(model, loader_test, 0.60, device) #doing testing
    saveTest(r, Path(DEFAULT_PATH.joinpath(f'ts{N_ESP}'))) #saving datas about the test

    
    save_path = Path(DEFAULT_PATH.joinpath(f'ts{N_ESP}')) #path where we're gonna save the confusion matrix
    SaveModel(model_weights, save_path,  loss_history_train=loss_history_train, epochs=epoch_history) #saving data on a csv


    #plotting the data in the file .csv with pandas
    plot_grapich(pandas.read_csv(save_path.joinpath('history_loss.csv'), index_col=None), 'epochs',  'FastRCNN(Loss)', save_path.joinpath('Loss.png'), 'epochs', 'Loss' )
    
        
        
    



