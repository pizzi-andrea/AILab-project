#Various imports
import pandas
from sklearn.metrics import confusion_matrix
from Model import ModelCNN
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
from GTSRB_Dataset import GTSRB_Dataset as Dataset
import torch
from torch.utils.data import DataLoader
from torch import nn
from SaveModel import SaveModel
from PlotGrapichs import plot_confusion_matrix,plot_grapich
from os import listdir
from TaT import trainModel, testModel
from __global__ import *

#setting default path for the saved model CNNModel
DEFAULT_PATH = Path('saved_model/CNNModel')

#main
if __name__ == '__main__':

    EPOCH = 10 #defining number of epoch
    N_ESP = len (listdir(DEFAULT_PATH)) + 1 if DEFAULT_PATH.exists() else 0
    
    #defining the trasformation that the images will under go during training and test
    seq = v2.Compose([
        v2.ToDtype(torch.float32, scale=True), #converting image to Dtype
        v2.Resize((48, 48), interpolation=InterpolationMode.NEAREST_EXACT),  #Resizing to 40x40
        v2.RandomAutocontrast(p=1.0), #applying contrast
        
    ])

    #dataset used for training and test
    dataset_train = Dataset(labels_path=LABELS_PATH_TRAIN, imgs_dir=IMGS_PATH_TRAIN, transform=seq)
    dataset_test = Dataset(labels_path=LABELS_PATH_TEST, imgs_dir=IMGS_PATH_TEST, transform=seq)

    #dataloaders for training and test
    loader_train = DataLoader(dataset_train, batch_size=64, shuffle=True, num_workers=3) # 94
    loader_test = DataLoader(dataset_test, batch_size=64, shuffle=False, num_workers=3)

    model = ModelCNN(input_channels=3, input_shape=3, hidden_units=32, output_shape=43) # 96 #defining the model
    device =  "cuda" if torch.cuda.is_available()  else "cpu" #selecting the device

    loss_fn = nn.CrossEntropyLoss() #getting the loss
    # lr = 1e-4 , weight_decay = 0 circa 96.7% (senza pre-processing)
    # lr = 0.005 circa 94% acc (senza pre-processing)
    # 98.34%

    #defining the optimizer, we use adamW with a learning rate of 1^(10^-4) and a decay of 0
    optimizer = torch.optim.AdamW(model.parameters(),  lr=1e-4, weight_decay=0)

    best_acc_test = 0
    model_weights = None

    #defining an history for all values of the epoch, accuracy and loss
    epoch_history    = []
    accuracy_history = []
    loss_history     = []

    for epoch in range(EPOCH):

        print("epoch {}".format(epoch + 1))

        #training the model
        trainModel(model, loader_train, loss_fn, optimizer, device)
        
        #doing the test and getting the result datas
        epoch_acc, epoch_loss, all_preds, all_labels = testModel(model, loader_test, loss_fn, device)

        #mechanism to store the best accuracy, if actual is better than the best, updating best accuracy
        if epoch_acc > best_acc_test:
            model_weights = model.state_dict()

            #memorizing values in history
            epoch_history.append(epoch)
            accuracy_history.append(epoch_acc)
            loss_history.append(epoch_loss)

            #switching the best accuracy with the new one
            best_acc_test = epoch_acc

    """
        Save Results
    """

    save_path = Path(DEFAULT_PATH.joinpath(f'ts{N_ESP}')) #path where we're gonna save the confusion matrix
    SaveModel(model_weights, save_path, accuracy_history, loss_history, epoch_history) #saving data on a csv

    #doing the test and obtaining it's result
    epoch_acc, epoch_loss, all_preds, all_labels = testModel(model, loader_test, loss_fn, device)

    #creating and setting the confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm,dataset_test.classes,save_path)

    #plotting the data in the file .csv with pandas
    plot_grapich(pandas.read_csv(save_path.joinpath('history.csv')), 'epochs', 'accuracy', 'CNNModel(Accuracy)', save_path.joinpath('Accuracy.png'), 'epochs', 'accuracy(%)' )
    plot_grapich(pandas.read_csv(save_path.joinpath('history.csv')), 'epochs', 'loss', 'CNNModel(Loss)', save_path.joinpath('Loss.png'), 'epochs', 'Loss' )