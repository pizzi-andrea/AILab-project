#various import
import torch
from torch import nn
from STN2 import STN
import matplotlib.pyplot as plt
#create convolutional neural network
class Model2(nn.Module):
    def __init__(self, input_channels: int, input_shape: int, output_shape: int):

        super().__init__()
        self.pre_processing = nn.Sequential( #pre processing block
            nn.LayerNorm(normalized_shape=[input_channels, input_shape, input_shape]),
            nn.LocalResponseNorm(size=3)
        )

        self.spatial1 = STN(input_channels, 512) # first trasformer
       
        self.conv1 = nn.Sequential( # first block, convolutional
             #convolutional layer
            nn.Conv2d(in_channels=3, out_channels=200, kernel_size=7, stride=1, padding=2),
            nn.ReLU(inplace=True), #activation layer
            nn.MaxPool2d(kernel_size=2), #reduction layer
            nn.LocalResponseNorm(size=3) #applying local response normalization

        )

        self.conv2 = nn.Sequential( #second block, convolutional
            nn.Conv2d(in_channels=200, out_channels=250, kernel_size=4, padding=2, stride=1), #convolutional layer
            nn.ReLU(), #activation layer
            nn.MaxPool2d(kernel_size=2), #reduction layer
            nn.LocalResponseNorm(size=3) #applying local response normalization
        )



        self.conv3 = nn.Sequential(  #third block, convolutional
            nn.Conv2d(in_channels=250, out_channels=350, kernel_size=4, padding=2, stride=1), #convolutional layer
            nn.ReLU(), # activation layer
            nn.MaxPool2d(kernel_size=2), #reduction layer
            nn.LocalResponseNorm(size=3) #applying local response normalization
        )

        self.classification = nn.Sequential( #forth block, classifier, full-connected
            nn.Flatten(),
            
            nn.Linear(in_features= 350*6*6, out_features=400), # linear (full-connected) layer
            nn.ReLU(), #activation layer
            nn.Linear(in_features=400, out_features=output_shape), # linear (full-connected) layer
        )
    
    def forward(self, x: torch.Tensor):
        x = self.pre_processing(x) #pass x to the pre_proc block
        
        # features extraction
        x = self.spatial1(x) #then we put it in the first trasformer
        
    
  
        x = self.conv1(x) #then we put it in the first convolutional block
        x = self.conv2(x) #then we put it in the second convolutional block
        x = self.conv3(x) #then we put it in the third convolutional block
        
        
        # classification
        x = self.classification(x) # then we classify it with the classifier block

        return x


if __name__ == '__main__':
    #creating an instance of model2
    m = Model2(input_channels=3, input_shape= 46, output_shape=43)

    test = torch.rand( size=(46, 46, 3), dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)

    print(test.shape)

    m(test)





    
        
        