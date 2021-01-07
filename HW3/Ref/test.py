import torch
from   torch import optim
import torch.nn
import torch.nn.functional
from   torch.utils.data import DataLoader
from   torchvision import datasets
import sys
import numpy
from   scipy.special import softmax as scipy_softmax
import cv2
from   HW3_109062631_Module import *


class testNet(torch.nn.Module):
    def __init__(self):
        super(testNet, self).__init__()
        self.conv = torch.nn.Conv2d(1, 8, kernel_size=(4, 4), stride=1)
        self.linr = torch.nn.Linear(in_features=1024, out_features=3364, bias=False)
        self.relu = torch.nn.ReLU()
        
    def forward(self, x):
        x = self.relu(x)
        return x

if __name__ == '__main__':
    # Print numpy ndarray without truncation
    numpy.set_printoptions(threshold=sys.maxsize)
    
    loss_f = torch.nn.CrossEntropyLoss()
    input_ = torch.Tensor([[0.25,0.25,0.25,0.25], [0.01,0.01,0.01,0.96]])
    target = torch.Tensor([3, 3]).long()
    output = loss_f(input_, target)
    print(input_)
    print(target)
    print(output)
    
    '''
    tensor(1.0152)
    '''
    
    in_np = numpy.array([[0.25,0.25,0.25,0.25], [0.01,0.01,0.01,0.96]])
    tg_np = numpy.array([3, 3])
    tg_np = oneHotEncoding(tg_np)
    
    aa = crossEntropy(in_np, tg_np)
    print(aa)
    
    
    
    
    '''
    aa_th = torch.randn(1, 1, 32, 32) # (N, C_in, H, W)
    
    TN = testNet()
    print(TN.linr.weight.size())
    '''
    '''
    print(aa_th.size()) # torch.Size([1, 1, 32, 32])
    print(bb_th.size()) # torch.Size([1, 8, 29, 29])
    '''
    #scipy_softmax(aa_th., axis=0)
    
    
#### TODO: linearBP
