"""Convolutional neural network for features extraction from images

:return: Convolutional neural network
:rtype: torch.nn.Module
"""
from torch import nn
from torch.nn.functional import relu as ReLU

class ConvolutionalNetwork(nn.Module):
    def __init__(self, input_channel: int, pooling: bool = False, pooling_type: str = "Max") -> None:
        """ Init method for Convolutional neural network
        
        """
        super(ConvolutionalNetwork, self).__init__()
        ks = (3,3)
        self.pooling = pooling
        
        #%224x224
        self.conv1_1 = nn.Conv2d(in_channels=input_channel, out_channels=16, kernel_size=ks, stride=1)
        self.conv1_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=ks, stride=1)
        if self.pooling:
            if pooling_type not in ["Max", "Avg"]:
                raise ValueError("Pooling type has not one of the value 'Max' or 'Avg'")
            elif pooling_type is "Max":
                self.conv1_3 = nn.MaxPool2d(kernel_size=ks, stride=1)
            elif pooling_type is "Avg":
                self.conv1_3 = nn.AvgPool2d(kernel_size=ks, stride=1)
        else:
            self.conv1_3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=ks, stride=2)
        #%112x112
        if self.pooling:
            self.conv2_1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=ks, stride=1)
        else:
            self.conv2_1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=ks, stride=1)
        self.conv2_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=ks, stride=1)
        if self.pooling:
            if pooling_type not in ["Max", "Avg"]:
                raise ValueError("Pooling type has not one of the value 'Max' or 'Avg'")
            elif pooling_type is "Max":
                self.conv2_3 = nn.MaxPool2d(kernel_size=ks, stride=1)
            elif pooling_type is "Avg":
                self.conv2_3 = nn.AvgPool2d(kernel_size=ks, stride=1)
        else:
            self.conv2_3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=ks, stride=2)
        #%56x56
        if self.pooling:
            self.conv3_1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=ks, stride=1)
        else:
            self.conv3_1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=ks, stride=1)
        self.conv3_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=ks, stride=1)
        if self.pooling:
            if pooling_type not in ["Max", "Avg"]:
                raise ValueError("Pooling type has not one of the value 'Max' or 'Avg'")
            elif pooling_type is "Max":
                self.conv3_3 = nn.MaxPool2d(kernel_size=ks, stride=1)
            elif pooling_type is "Avg":
                self.conv3_3 = nn.AvgPool2d(kernel_size=ks, stride=1)
        else:
            self.conv3_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=ks, stride=2)
        #%28x28
        if self.pooling:
            self.conv4_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=ks, stride=1)
        else:
            self.conv4_1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=ks, stride=1)
        self.conv4_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=ks, stride=1)
        if self.pooling:
            if pooling_type not in ["Max", "Avg"]:
                raise ValueError("Pooling type has not one of the value 'Max' or 'Avg'")
            elif pooling_type is "Max":
                self.conv4_3 = nn.MaxPool2d(kernel_size=ks, stride=1)
            elif pooling_type is "Avg":
                self.conv4_3 = nn.AvgPool2d(kernel_size=ks, stride=1)
        else:
            self.conv4_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=ks, stride=2)
        #%14x14
        if self.pooling:
            self.conv5_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=ks, stride=1)
        else:
            self.conv5_1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=ks, stride=1)
        self.conv5_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=ks, stride=1)
        if self.pooling:
            if pooling_type not in ["Max", "Avg"]:
                raise ValueError("Pooling type has not one of the value 'Max' or 'Avg'")
            elif pooling_type is "Max":
                self.conv5_3 = nn.MaxPool2d(kernel_size=ks, stride=1)
            elif pooling_type is "Avg":
                self.conv5_3 = nn.AvgPool2d(kernel_size=ks, stride=1)
        else:
            self.conv5_3 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=ks, stride=2)
        #%7x7
        if self.pooling:
            self.conv6_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=ks, stride=1)
        else:
            self.conv6_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=ks, stride=1)
        self.conv6_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=ks, stride=1)
        self.conv6_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=ks, stride=1)
        
        
        
    def forward(self, input_img):
        """Forward pass for the convolutional neural network

        :param input_img: the input image
        :type input_img: torch.Tensor
        :return: convolutional output that is the extracted features from the images ad different layer
        :rtype: tuple[torch.Tensor]
        """
        #256x256
        x = ReLU(self.conv1_1(input_img))
        x = self.conv1_2(x)
        x = ReLU(x)
        conv1_out = x.clone().detach()
        #128x128
        if self.pooling:
            x = self.conv1_3(x)
        else:
            x = ReLU(self.conv1_3(x))
        x = ReLU(self.conv2_1(x))
        x = ReLU(self.conv2_2(x))
        conv2_out = x.clone().detach()
        #56x56
        if self.pooling:
            x = self.conv2_3(x)
        else:
            x = ReLU(self.conv2_3(x))
        x = ReLU(self.conv3_1(x))
        x = ReLU(self.conv3_2(x))
        conv3_out = x.clone().detach()
        #28x28
        if self.pooling:
            x = self.conv3_3(x)
        else:
            x = ReLU(self.conv3_3(x))
        x = ReLU(self.conv4_1(x))
        x = ReLU(self.conv4_2(x))
        conv4_out = x.clone().detach()
        #14x14
        if self.pooling:
            x = self.conv4_3(x)
        else:
            x = ReLU(self.conv4_3(x))
        x = ReLU(self.conv5_1(x))
        x = ReLU(self.conv5_2(x))
        conv5_out = x.clone().detach()
        #7x7
        if self.pooling:
            x = self.conv5_3(x)
        else:
            x = ReLU(self.conv5_3(x))
        x = ReLU(self.conv6_1(x))
        x = ReLU(self.conv6_2(x))
        x = ReLU(self.conv6_3(x))
        conv6_out = x.clone().detach()
        
        return x, conv1_out, conv2_out, conv3_out, conv4_out, conv5_out, conv6_out
