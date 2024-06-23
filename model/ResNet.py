import torch
from torch import nn
from STN import SpatialTransformer

#basic block convolutional
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1)  #convolutional layer
        self.bn1 = nn.BatchNorm2d(out_channels)  #applying batch normalization
        self.relu = nn.ReLU(inplace=True) #activation layer
        self.conv2 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=3,
                               stride=1, padding=1) #convolution layer
        self.bn2 = nn.BatchNorm2d(out_channels * self.expansion) #applying batch normalization
        self.downsample = downsample #setting downsample 
        self.stride = stride #setting stride
    
    def forward(self, x):
        shortcut = x 
        out = self.conv1(x) # pass x to the first convolution layer
        out = self.bn1(out) #then to te batch normalization layer
        out = self.relu(out) # then to the activation layer

        out = self.conv2(out) #then to the second convolutional layer
        out = self.bn2(out) #then to the second batch normalization layer
        if self.downsample is not None:
            shortcut = self.downsample(x)
        out += shortcut
        out = self.relu(out) #then to the activation layer
        return out

# class for the resnet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()
        self.inplanes = 64


        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=2, padding=1)
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3) #convolutional layer
        self.bn1 = nn.BatchNorm2d(self.inplanes) #applying batch normalization
        self.relu = nn.ReLU(inplace=True) #activation layer
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) #reduction layer
        
        # backbone
        self.layer1 = self._make_layer(block, 64, layers[0]) #first block 
        self.sp1 = SpatialTransformer(64) #first trasformer
       
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) #second block
        self.sp2 = SpatialTransformer(128) #second trasformer

        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)#third block
        #self.sp3 = SpatialTransformer(256)

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2) #forth block
        
        # classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) #reduction layer
        self.fc = nn.Linear(512 * block.expansion, num_classes) #full-connected layer


    #create a layer based on the params passsed
    def _make_layer(self, block, planes, blocks, stride=1): 
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            #creation of the downsample
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1,
                          stride=stride, bias=False), #convolutional layer
                nn.BatchNorm2d(planes * block.expansion) #layer for batch normalization
            )
        layers = [] #defining the list of layers
        layers.append(block(self.inplanes, planes,stride, downsample)) #appending the block to the layer
        self.inplanes = planes * block.expansion #defining the numbler of inplanes
        for i in range(1, blocks): #appending all layers
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers) #returning a neural network of various layers
    
    def forward(self, x):

        out = self.conv1(x) #pass x to the first convolutional layer
        out = self.bn1(out) #then pass to the batch normalization layer
        out = self.relu(out) #then to the activation layer
        out = self.maxpool(out) #then to the reduction layer

        out = self.layer1(out) #then to the first block layer
        out = self.sp1(out) #then to the first trasformer 

        out = self.layer2(out) #then to the second block layer
        #print(out.shape)
        out = self.sp2(out)  #then to the second trasformer 

        out = self.layer3(out)  #then to the third block layer
        #out = self.sp3(out)

        out = self.layer4(out) #then to the fourth block layer

        out = self.avgpool(out) #reduction layer
        out = out.flatten(1) # applying the flatten
        out = self.fc(out) #then to the full-connected layer
        return out

def resnet32(num_classes=1000): #return a resnet 32
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)

def resnet18(num_classes=1000): #return a resnet18
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)