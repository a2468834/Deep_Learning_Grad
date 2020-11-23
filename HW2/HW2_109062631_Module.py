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
#from   imblearn.over_sampling import RandomOverSampler
#import re
#import sys
#import time

########## Packages ##########
import math
import matplotlib.pyplot as pyplot
import numpy
import os
import pathlib
import pickle
import random
import time
import torch
from   torch import optim
import torch.nn
import torch.nn.functional
from   torch.utils.data import DataLoader
from   torchvision import datasets


########## Program Constants ##########
class CONST:
    # Define some constants
    epoch_num     = lambda : 200
    row_pixel     = lambda : 26
    col_pixel     = lambda : 26
    input_channel = lambda : 3
    output_class  = lambda : 9
    batch_size    = lambda : 32
    learning_rate = lambda : 0.05
    OS_number     = lambda : [314, 403, 108, 373, 107, 388, 330, 332, 1]


########## Convolutional AutoEncoder ##########
class convAutoEncdr(torch.nn.Module):
    def __init__(self, input_channel=CONST.input_channel(), h_dim=13*13*4, z_dim=16):
        super(convAutoEncdr, self).__init__()
        
        # input data : width x height x depth(i.e., channel) = 26 x 26 x 3
        # Encoder
        self.encoder = torch.nn.Sequential(
                           torch.nn.Conv2d(in_channels=input_channel, out_channels=16, kernel_size=3, padding=1),
                           torch.nn.ReLU(),
                           torch.nn.MaxPool2d(kernel_size=2),
                           torch.nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, padding=1),
                           torch.nn.ReLU(),
                           torch.nn.MaxPool2d(kernel_size=1)
                       )
        
        # Variational latent layer
        self.fc_mu     = torch.nn.Linear(in_features=h_dim, out_features=z_dim)
        self.fc_logvar = torch.nn.Linear(in_features=h_dim, out_features=z_dim)
        self.fc_rev    = torch.nn.Linear(in_features=z_dim, out_features=h_dim)
        
        # Decoder
        self.decoder = torch.nn.Sequential(
                           torch.nn.ConvTranspose2d(in_channels=4, out_channels=16, kernel_size=1, stride=1),
                           torch.nn.ReLU(),
                           torch.nn.ConvTranspose2d(in_channels=16, out_channels=input_channel, kernel_size=2, stride=2),
                           torch.nn.Sigmoid()
                       )
    
    def forward(self, x):
        if self.training:
            x = self.encoder(x)
            x = self.decoder(x)
            return x
        else:
            x = self.encoder(x)
            x = x + randomNoise(x)
            x = self.decoder(x)
            return x
    
    def reParameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.empty_like(std).normal_()
        z = eps * std + mu
        return z


########## Functions ##########
def lossFunc(x_recon, x):
    BCE_loss = torch.nn.functional.binary_cross_entropy(x_recon, x)
    
    return BCE_loss


def readNPY():
    working_dir    = pathlib.Path().absolute()
    data_npy_path  = os.path.join(working_dir, 'data.npy')
    label_npy_path = os.path.join(working_dir, 'label.npy')
    
    data_ndarray  = numpy.load(data_npy_path)
    label_ndarray = numpy.load(label_npy_path)
    
    # Change image channel ordering from (# of samples, col, row, CH) to (# of samples, CH, col, row)
    data_ndarray = numpy.rollaxis(data_ndarray, 3, 1)
    
    return data_ndarray, label_ndarray


def ndarrayToList(data_ndarray, label_ndarray):
    data_list  = [data_ndarray[index, :, :, :] for index in range(data_ndarray.shape[0])]
    label_list = [label_ndarray[index, :] for index in range(label_ndarray.shape[0])]
    
    return data_list, label_list


# Notice: 'CH' must be placed at the second dimension of ndarray/torsor.
def printImage(data_to_print, is_ndarray=False, save_file_name=None):
    if is_ndarray:
        data_tensor = torch.Tensor(data_to_print)
    else:
        data_tensor = data_to_print
    
    img_tensor = torch.zeros([data_tensor.shape[1], data_tensor.shape[2]], dtype=torch.int8)
    
    for index1 in range(data_tensor.shape[1]):
        for index2 in range(data_tensor.shape[2]):
            img_tensor[index1, index2] = torch.argmax(data_tensor[:, index1, index2]).item()
    
    if save_file_name != None:
        pyplot.imsave(save_file_name, img_tensor, cmap=pyplot.cm.Greens)
    else:
        pyplot.imshow(img_tensor, cmap=pyplot.cm.Greens)
        pyplot.show()


# Find max value from one of 3 channels and set it to 1, other channels set to 0
def setMaxCHToOne(data_tensor):
    result_tensor =torch.zeros(data_tensor.shape)
    
    for index1 in range(data_tensor.shape[1]):
        for index2 in range(data_tensor.shape[2]):
            if torch.argmax(data_tensor[:, index1, index2]).item() == 0:
                result_tensor[0, index1, index2] = 1.0
                result_tensor[1, index1, index2] = 0.0
                result_tensor[2, index1, index2] = 0.0
            elif torch.argmax(data_tensor[:, index1, index2]).item() == 1:
                result_tensor[0, index1, index2] = 0.0
                result_tensor[1, index1, index2] = 1.0
                result_tensor[2, index1, index2] = 0.0
            elif torch.argmax(data_tensor[:, index1, index2]).item() == 2:
                result_tensor[0, index1, index2] = 0.0
                result_tensor[1, index1, index2] = 0.0
                result_tensor[2, index1, index2] = 1.0
    
    return result_tensor


def have_CUDA():
    if torch.cuda.is_available():
        return 'cuda:0'
    else:
        return 'cpu'


def randomNoise(latent_tensor):
    noise_tensor = torch.zeros(latent_tensor.size())
    noise_tensor = torch.empty_like(latent_tensor).normal_()
    return noise_tensor
