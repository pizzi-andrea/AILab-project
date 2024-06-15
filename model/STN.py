import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialTransformer(nn.Module):
    def __init__(self, input_channels):
        super(SpatialTransformer, self).__init__()
        # Localisation network for images with any number of channels
        self.localization = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=3, padding=1),
            nn.AdaptiveMaxPool2d( (2, 2) ),
            nn.ReLU(True)
        )

        # Placeholder for input size of the fully connected layer
        self.fc_loc_input_size = None

        # Fully connected layer to output the 2x3 affine transformation matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 2 * 2, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def _get_fc_loc_input_size(self, input_shape):
        dummy_input = torch.zeros(1, *input_shape).to(next(self.parameters()).device)
        dummy_output = self.localization(dummy_input)
        return int(torch.prod(torch.tensor(dummy_output.size()[1:])))

    def forward(self, x):
        # Forward pass through the localisation network
        xs = self.localization(x)
        xs = xs.view(xs.size(0), -1)
        
        # Dynamically set the input size of the fully connected layer
        if self.fc_loc_input_size is None:
            self.fc_loc_input_size = self._get_fc_loc_input_size(x.shape[1:])
            self.fc_loc[0] = nn.Linear(self.fc_loc_input_size, 32).to(x.device)
            self.fc_loc[2] = nn.Linear(32, 3 * 2).to(x.device)
            self.fc_loc[2].weight.data.zero_()
            self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        # Generate the grid using the predicted theta
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        
        # Sample the input image with the generated grid
        x = F.grid_sample(x, grid, align_corners=False)

        return x

# Example usage
# Assume input is a batch of images with size (N, C, H, W)


#print("Input shape:", input_tensor.shape)
#print("Transformed input shape:", transformed_input.shape)
