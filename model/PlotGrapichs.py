#various imports
from locale import atoi
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import os
import cv2 as cv

#class for plotting the confusion matrix
def plot_confusion_matrix(cm, class_names, path: Path = Path('.')):
    plt.figure(figsize=(10, 7)) #size of the cm

    #creating the heatmap from the confusion matrix
    hmp = sns.heatmap(cm, annot=True, fmt='d', cmap='Paired', cbar=False, linewidths=1, xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 4})

    plt.xlabel('Predicted') #assigning name to the x axis
    plt.ylabel('True') #assigning name to the y axis

    
    #saving the heatmap on disk
    hmp.figure.savefig(path.joinpath('cnfm.svg'), format='svg', pad_inches=1)

#function for plotting the graphic
def plot_grapich(dataFrame: pd.DataFrame, idx: str, title: str, filePath: str = None, x_label: str = None, y_label: str = None ) -> None:
    print(filePath)
    #plotting
    dataFrame.plot(title=title, xlabel= x_label, ylabel= y_label, x=idx)
    
    #saving or showing the graphic
    if filePath: 
        plt.savefig(filePath)
    else:
        plt.show()
    

def plot_predicted(pathImgs: Path, X: list[int], y: list[int]):
    
    file_name = [  label  for label in os.listdir(pathImgs) ] # list of name files in the dir
    labels = [  label.split('.')[0]  for label in os.listdir(pathImgs) ] #getting the labels

    labels_to_images = { atoi(label): pathImgs.joinpath(fn) for label, fn in zip(labels, file_name)} # calc the path for each image

    for x, y in zip(X, y):
        fig, axes = plt.subplots(2, 1, figsize=(12, 12))
        color = ('g' if x == y else 'r') #setting color for the plot

        #reading images
        img_x = cv.imread(labels_to_images[x]) 
        img_y = cv.imread(labels_to_images[y])

        #showing images on the plot
        axes[0].imshow(img_x)
        axes[1].imshow(img_y)
        
        title_text = f"Pred: {x} | Truth: {y}"
        
        axes[0].set_title(title_text, fontsize=10, c=color ) #setting the title of the 0 axis
        plt.show() #showing the plot


