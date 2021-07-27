#AlexNet & MNIST

import numpy as np
import cv2
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
import time
from torch.optim.lr_scheduler import StepLR

class Net(nn.Module):


    def __init__(self):
        super(Net,self).__init__()
        
        self.conv = torch.nn.Sequential(nn.Conv2d(1,64,1,padding=1),
                                        nn.Conv2d(64,64,3,padding=1),
                                        nn.MaxPool2d(2, 2),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(),
                                        nn.Conv2d(64,128,3,padding=1),
                                        nn.Conv2d(128, 128, 3,padding=1),
                                        nn.MaxPool2d(2, 2, padding=1),
                                        nn.BatchNorm2d(128),
                                        nn.ReLU()
                                        )
        self.dense = torch.nn.Sequential(torch.nn.Linear(8*8*128,512),
                                         torch.nn.ReLU(),
                                         torch.nn.Dropout(p=0.5),
                                         torch.nn.Linear(512, 10))

    def forward(self,x):
        x = self.conv(x)
        x = x.view(-1,128*8*8)
        x = self.dense(x)

        return x