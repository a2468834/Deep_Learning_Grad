#   Homework 2: Convolutional Autoencoder - Module
#    
#   Date:        2020/11/06
#   CourseID:    10910COM526000
#   Course:      Deep Learning (Graduated)
#   
#   Writer_ID:   109062631
#   Writer_Name: Wang, Chuan-Chun
#   Environment: 
#      Software: Python 3.8.5 on 64-bit Windows 10 Pro (2004)
#      Hardware: Intel i7-10510U, 16GB DDR4 non-ECC ram, and no discrete GPU
import math
import matplotlib.pyplot as pyplot
import numpy
import os
import pandas
import pathlib
import pickle
import random
import re
import sys
import time
import torch
import torch
import torch.nn
import torch.nn.functional
#from   torch.utils.data import DataLoader
#from   torch.utils.data.sampler import SubsetRandomSampler
#from   torchvision import transforms
#from   torch import optim
#import warnings


########## Classes ##########
class CONST:
    # Define some constants
    row_pixel      = lambda : 26
    col_pixel      = lambda : 26
    input_channel  = lambda : 3
    output_class   = lambda : 9
    pass

class FUNC:
    # Define some basic functions
    dReLU = lambda x : (x > 0.0).astype(numpy.float)        # 1st Derivative of reLU w.r.t. 'x', i.e., unit step function
    reLU  = lambda x : x * (x > 0.0)
    
    #C-E Loss = $-\sum{\mathrm{label}_i{\ }log_{e}(\mathrm{predict}_i)}$, where $i{\in}$ vector's index set
    def crossEntropy(predict_vec, truth_vec):
        if predict_vec.shape != truth_vec.shape:
            print("Wrong dimension of predict vector and truth vector in cross entropy.")
            exit()
        else:
            # Add tiny amount 1e-8 to avoid put zero into numpy.log()
            return (-1.0) * numpy.sum(truth_vec * numpy.log(predict_vec.T + 1e-8))

# Convolutional AutoEncoder 
class convAutoEncdr(torch.nn.Module):
    def __init__(self):
        super(convAutoEncdr, self).__init__()
        # Declaration each layer in convolutional autoencoder 
        # input data : width x height x depth = 26 x 26 x 3
        
        # Encoder
        self.conv1 = torch.nn.Sequential(
                         torch.nn.Conv2d(in_channels=CONST.input_channel(), out_channels=10, kernel_size=5, stride=1), # output = 22x22x10
                         torch.nn.ReLU(), # output = 22x22x10 (un-changed)
                         torch.nn.MaxPool2d(kernel_size=2) # output = 11x11x10, stride=kernel_size
                     )
        self.conv2 = torch.nn.Sequential(
                         torch.nn.Conv2d(in_channels=10, out_channels=5, kernel_size=5, stride=1), # output = 7x7x5
                         torch.nn.ReLU(), # output = 7x7x5 (un-changed)
                         torch.nn.MaxPool2d(kernel_size=3, stride=1) # output = 5x5x5, stride=kernel_size
                     )
        self.fulconn = torch.nn.Sequential(
                           torch.nn.Linear(5 * 5 * 5, 20),
                           torch.nn.ReLU()
                       )
        
        # Decoder
        
        
        self.output  = torch.nn.Linear(20, CONST.output_class())
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fulconn(x)
        x = self.out(x)
        return x


def readNPY():
    working_dir    = pathlib.Path().absolute()
    data_npy_path  = os.path.join(working_dir, 'data.npy')
    label_npy_path = os.path.join(working_dir, 'label.npy')
    
    data_ndarray  = numpy.load(data_npy_path)
    label_ndarray = numpy.load(label_npy_path)
    
    # Don't forget to convet numpy ndarray to torch tensor
    return torch.from_numpy(data_ndarray), torch.from_numpy(label_ndarray)


def printImage(enable_print_y, data_x_part, data_y_part=None):
    # Print data into a gray image
    pyplot.imshow(data_x_part.reshape(CONST.row_pixel(), CONST.col_pixel()), cmap=pyplot.cm.gray)
    if enable_print_y == True: pyplot.title("%s"%str(data_y_part))
    pyplot.show()
