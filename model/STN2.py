from torch import nn
from torch.functional import F
import torch
class STN(nn.Module):
    def __init__(self, input_channels, l):
        super(STN, self).__init__()
       
        # spatial transformer localization network
        self.localization = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        # tranformation regressor for theta
        self.fc_loc = nn.Sequential(
            nn.Linear(8192, 256),
            nn.ReLU(True),
            nn.Linear(256, 3 * 2)
        )
        # initializing the weights and biases with identity transformations
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], 
                                                    dtype=torch.float))
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, xs.size(1)*xs.size(2)*xs.size(3))
        # calculate the transformation parameters theta
        theta = self.fc_loc(xs)
        # resize theta
        theta = theta.view(-1, 2, 3) 
        # grid generator => transformation on parameters theta
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        # grid sampling => applying the spatial transformations
        x = F.grid_sample(x, grid, align_corners=False)
        return x
    def forward(self, x):
        # transform the input
        x = self.stn(x)
        
        return x
