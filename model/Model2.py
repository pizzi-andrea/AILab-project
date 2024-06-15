import torch
from torch import nn
from STN import SpatialTransformer as STN

class Model2(nn.Module):

    def __init__(self, input_channels: int, input_shape: int, output_shape: int):

        super().__init__()
        self.pre_processing = nn.Sequential(
            nn.LayerNorm(normalized_shape=[input_channels, input_shape, input_shape]),
            nn.LocalResponseNorm(size=3)
        )

        self.spatial1 = STN(input_channels=input_channels)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=200, kernel_size=5, stride=1, padding=2
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)

        )

        self.spatial2 = STN(input_channels=200)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=200, out_channels=250, kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )


        self.spatial3 = STN(input_channels=250)

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=250, out_channels=350, kernel_size=5, padding=2, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        

        self.classification = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features= 8750, out_features=400),
            nn.ReLU(),
            nn.Linear(in_features=400, out_features=output_shape),
            nn.Softmax(dim=0)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.pre_processing(x)
        
        # fectures exstraction
        x = self.spatial1(x)
        x = self.conv1(x)
      

        x = self.spatial2(x)
        x = self.conv2(x)

        x = self.spatial3(x)
        x = self.conv3(x)
        
        
        # classification
        x = self.classification(x)

        return x


if __name__ == '__main__':
    m = Model2(input_channels=3, input_shape= 46, output_shape=43)

    test = torch.rand( size=(46, 46, 3), dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)

    print(test.shape)

    m(test)





    
        
        