#various import
import torch
from torch import nn
from STN import SpatialTransformer

# Create a convolutional neural network
class ModelCNN(nn.Module):
    def __init__(self, input_channels:int, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.pre_processing = nn.Sequential( #pre-processing block
            nn.LayerNorm(normalized_shape=[input_channels, input_shape, input_shape]),
            nn.LocalResponseNorm(size=3)
        )

        self.sp1 = SpatialTransformer(input_channels) # first trasformer
        
        self.block_1 = nn.Sequential( # first block, convolutional
            
            #convolutional layer
            nn.Conv2d(in_channels=input_channels,
                      out_channels=hidden_units,
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1),# options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number
            nn.BatchNorm2d(hidden_units), #applying batch normalization
            nn.ReLU(), #activation layer

            #convolution layer
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),

            nn.BatchNorm2d(hidden_units), #applying batch normalization
            nn.ReLU(), #activation layer

            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )

        self.sp2 = SpatialTransformer(hidden_units) #second trasformer
        
        
        self.block_2 = nn.Sequential( #second block, convolutional
            nn.Conv2d(hidden_units, hidden_units*2, 3, padding=1), #convolutional layer
            nn.BatchNorm2d(hidden_units*2), #applying batch normalization
            nn.ReLU(), #activation layer

            nn.Conv2d(hidden_units*2, hidden_units, 3, padding=1), #convolutional layer
            nn.BatchNorm2d(hidden_units), #applying batch normalization
            nn.ReLU(), #activation layer

            nn.MaxPool2d(2) # reduction layer
        )
        self.sp3 = SpatialTransformer(hidden_units) #third trasformer
        
        self.block_3 = nn.Sequential( #third block, convolutional
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1), #convolutional layer
            nn.BatchNorm2d(hidden_units), #applying batch normalization
            nn.ReLU(), #activation layer 

            nn.Conv2d(hidden_units, hidden_units, 3, padding=1), #convolutional layer
            nn.BatchNorm2d(hidden_units), #applying batch normalization
            nn.ReLU(), #activation layer

            nn.Conv2d(hidden_units, hidden_units, 3, padding=1), #convolutional layer
            nn.BatchNorm2d(hidden_units), #applying batch normalization
            nn.ReLU(), #activation layer

            nn.MaxPool2d(2) # reduction layer
        )

        self.classifier = nn.Sequential( #forth block, classifier, full-connected
            nn.Flatten(),
            nn.Linear(in_features=3_456, 
                      out_features=3_456), # linear (full-connected) layer
            nn.Linear(in_features=3_456,
                      out_features=output_shape), # linear (full-connected) layer
            
        )

    #defining the forwarding operation 
    def forward(self, x: torch.Tensor):
        # print(x.shape)
        x = self.pre_processing(x) #pass x to the pre_proc block

        x = self.sp1(x) #then we put it in the first trasformer
        # print(x.shape)
        x = self.block_1(x) #then we put it in the first convolutional block
        # print(x.shape)
        x = self.sp2(x) #then we put it in the second trasformer
        x = self.block_2(x) #then we put it in the second convolutional block
        # print(x.shape)
        x = self.sp3(x) #then we put it in the third trasformer
        # print(x.shape)
        x = self.block_3(x) #then we put it in the third convolutional block
        # print(x.shape)
        x = self.classifier(x) # then we classify it with the classifier block
        # print(x.shape)
        return x

