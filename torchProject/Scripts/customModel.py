import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn import Linear, Conv2d, Module, MaxPool2d, ReLU, LogSoftmax, Flatten, Sequential, Tanh, AvgPool2d, Softmax

class LeNet(Module):
    def __init__(self):
        super(LeNet,self).__init__()
        
        self.layer1 = Sequential( Conv2d(in_channels=1,out_channels=6,kernel_size=[5,5],padding=2), #Shape [28,28,6]
                                  Tanh(),
                                  AvgPool2d(kernel_size=[2,2], stride=2) #Shape [14,14,6]
                                )
        
        self.layer2 = Sequential(Conv2d(in_channels=6, out_channels=16, kernel_size=[5,5]), #Shape[10,10,16]
                                 Tanh(),
                                 AvgPool2d(kernel_size=[2,2],stride=2) #Shape[5,5,16]
                                 ) 
        self.layer3 = Sequential(Flatten(), #Shape[400]
                                 Linear(in_features=(400),out_features=(128)),
                                 Tanh())
        self.layer4 = Sequential(Linear(in_features=(128),out_features=(84)),
                                 Tanh())
        self.layer5 = Sequential(Linear(in_features=(84),out_features=(10)),
                                 Softmax())
         
        

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        return out



