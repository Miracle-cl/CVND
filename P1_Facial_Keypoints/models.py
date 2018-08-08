## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

def weights_init(m):
    # weights initialization
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        I.xavier_uniform_(m.weight.data) # glorot uniform initialization
        m.bias.data.fill_(0)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs

        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        #self.conv1 = nn.Conv2d(1, 32, 5)

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.layer1 = nn.Sequential(nn.Conv2d(1, 32, kernel_size=4),
                                    nn.ELU(),
                                    nn.MaxPool2d(2, 2),
                                    nn.Dropout(p=0.1))

        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3),
                                    nn.ELU(),
                                    nn.MaxPool2d(2, 2),
                                    nn.Dropout(p=0.2))

        self.layer3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=2),
                                    nn.ELU(),
                                    nn.MaxPool2d(2, 2),
                                    nn.Dropout(p=0.3))

        self.layer4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1),
                                    nn.ELU(),
                                    nn.MaxPool2d(2, 2),
                                    nn.Dropout(p=0.4))

        self.layer5 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1), 
                                    nn.ELU(),
                                    nn.MaxPool2d(2, 2),
                                    nn.Dropout(p=0.4))

        self.layer6 = nn.Sequential(nn.Linear(256*6*6, 1024), nn.ELU(), nn.Dropout(0.5))

        self.layer7 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.6))

        self.fc = nn.Linear(512, 136)


    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))

        x = self.layer1(x) # 1x224x224 - 32x221x221 - 32x110x110
        x = self.layer2(x) # 32x110x110 - 64x108x108 - 64x54x54
        x = self.layer3(x) # 64x54x54 - 128x53x53 - 128x26x26
        x = self.layer4(x) # 128x26x26 - 256x26x26 - 256x13x13
        x = self.layer5(x) # 256x13x13 - 256x13x13 - 256x6x6
        x = x.view(x.size(0), -1)
        x = self.layer6(x) # 9216 - 1024
        x = self.layer7(x) # 1024 - 512
        x = self.fc(x) # 512 - 136

        # a modified x, having gone through all the layers of your model, should be returned
        return x
