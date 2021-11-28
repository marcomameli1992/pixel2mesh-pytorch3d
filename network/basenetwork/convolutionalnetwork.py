"""Convolutional neural network for features extraction from images

:return: Convolutional neural network
:rtype: torch.nn.Module
"""
from torch import nn

class ConvolutionalNetwork(nn.Module):
    def __init__(self, input_channel:int) -> None:
        """ Init method for Convolutional neural network
        
        """
        super(ConvolutionalNetwork, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=input_channel, out_channels=64, kernel_size=(3,3))
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3))
        
        self.pooling = nn.AvgPool2d((2,2))
        
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3))
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3))

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3))
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3))
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3))

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3))
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3))
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=5112, kernel_size=(3,3))

        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3))
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3))
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=5112, kernel_size=(3,3))
        
        
    def forward(self, input_img):
        """Forward pass for the convolutional neural network

        :param input_img: the input image
        :type input_img: torch.Tensor
        :return: convolutional output that is the extracted features from the images ad different layer
        :rtype: torch.Tensor
        """
        conv_out1 = self.conv1_1(input_img)
        conv_out1 = self.conv1_2(conv_out1)
        pool_out = self.pooling(conv_out1)
        conv_out2 = self.conv2_1(pool_out)
        conv_out2 = self.conv2_2(conv_out2)
        conv_out2 = self.pooling(conv_out2)
        conv_out3 = self.conv3_1(conv_out2)
        conv_out3 = self.conv3_2(conv_out3)
        conv_out3_1 = self.conv3_3(conv_out3)
        conv_out3_1 = self.pooling(conv_out3_1)
        conv_out4 = self.conv4_1(conv_out3_1)
        conv_out4 = self.conv4_2(conv_out4)
        conv_out4 = self.conv4_3(conv_out4)
        conv_out4 = self.pooling(conv_out4)
        conv_out5 = self.conv5_1(conv_out4)
        conv_out5_1 = self.conv5_2(conv_out5)
        conv_out5_1 = self.conv5_3(conv_out5_1)
        
        return conv_out5_1, conv_out5, conv_out3, conv_out1
