import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
def plot_confusion_matrix(cm, class_names, path: Path = Path('.')):
    plt.figure(figsize=(10, 7))

    hmp = sns.heatmap(cm, annot=True, fmt='d', cmap='Paired', cbar=False, linewidths=1, xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 4})

    plt.xlabel('Predicted')
    plt.ylabel('True')
    #plt.show()
    #plt.savefig(path.joinpath('cnfm.png'), bbox_inches='tight', pad_inches=0.5)
   
    hmp.figure.savefig(path.joinpath('cnfm.svg'), format='svg', pad_inches=1)

def plot_grapich(dataFrame: pd.DataFrame, x: str, y: str, title: str, filePath: str = None, x_label: str = None, y_label: str = None ) -> None:
    
    dataFrame.plot(title=title, x = x, y = y, xlabel= x_label, ylabel= y_label)
    
    if filePath:
        plt.savefig(filePath)
    else:
        plt.show()
    

    
if __name__ == '__main__':

    df = pd.read_csv('saved_model/CNNModel/ts1/history.csv')
    plot_grapich(df, 'epochs', 'accuracy', 'test', '', 'epochs', 'accuracy(%)')