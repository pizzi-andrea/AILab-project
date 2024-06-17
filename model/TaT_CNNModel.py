import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from Model import ModelCNN
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
from GTSRB_Dataset import GTSRB_Dataset as Dataset
import torch
from torch.utils.data import DataLoader
from torch import nn


from TaT import trainModel, testModel
from __global__ import *


def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

if __name__ == '__main__':

    EPOCH = 30
    seq = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Resize((48, 48), interpolation=InterpolationMode.NEAREST_EXACT),
        v2.RandomAutocontrast(p=1.0),
        
    ])

    dataset_train = Dataset(labels_path=LABELS_PATH_TRAIN, imgs_dir=IMGS_PATH_TRAIN, transform=seq)
    dataset_test = Dataset(labels_path=LABELS_PATH_TEST, imgs_dir=IMGS_PATH_TEST, transform=seq)


    loader_train = DataLoader(dataset_train, batch_size=64, shuffle=True, num_workers=3)
    loader_test = DataLoader(dataset_test, batch_size=64, shuffle=False, num_workers=3)

    model = ModelCNN(input_channels=3, input_shape=3, hidden_units=64, output_shape=43)
    device =  "cuda" if torch.cuda.is_available()  else "cpu"

    loss_fn = nn.CrossEntropyLoss()
    # lr = 1e-4 , weight_decay = 0 circa 96.7% (senza pre-processing)
    # lr = 0.005 circa 94% acc (senza pre-processing)

    optimizer = torch.optim.AdamW(model.parameters(),  lr=1e-4, weight_decay=0)


    for epoch in range(EPOCH):

        print("epoch {}".format(epoch + 1))

        trainModel(model, loader_train, loss_fn, optimizer, device)
        testModel(model, loader_test, loss_fn, device)
    
    all_labels, all_preds = testModel(model, loader_test, loss_fn, device)

    
   
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm,dataset_test.classes)