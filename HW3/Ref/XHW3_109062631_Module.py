#  Homework 3:  CNN from Scratch - Module
#  
#  Date:        2020/12/14
#  CourseID:    10910COM526000
#  Course:      Deep Learning (Graduated)
#  
#  Writer_ID:   109062631
#  Writer_Name: Wang, Chuan-Chun
#  Environment:
#    [Configuration 1]
#      SW:  Python 3.8.5 on 64-bit Windows 10 Pro (2004)
#      HW:  Intel i7-10510U, 16GB DDR4 non-ECC ram, and no discrete GPU
#    [Configuration 2]
#      SW:  Python 3.8.5 on Ubuntu 20.04.1 LTS (Linux 5.4.0-54-generic x86_64)
#      HW:  AMD Ryzen 5 3400G, 64GB DDR4 non-ECC ram, and no discrete GPU
import cv2
import math
import matplotlib.pyplot as pyplot
from   multiprocessing import Pool
import numpy
import os
import pathlib
import pickle
import random
import re
from   scipy.special import softmax as scipy_softmax
import struct
import sys
import time
import warnings
#from   memory_profiler import profile

########## Classes ##########
class CONST:
    # Define some constants
    row_num      = lambda : 32                                # height of a image in pixels
    col_num      = lambda : 32                                # width of a image in pixels
    input_dim    = lambda : CONST.row_num() * CONST.col_num()
    output_dim   = lambda : 10
    batch_size   = lambda : 2**8
    v_proportion = lambda : 0.3                               # size(validation) / size(total dataset)
    perplexity   = lambda : 30


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


class MODELPARA:
    def __init__(self):
        pass


class CNNMODEL:
    def __init__(self, network_param):
        self.network_param = network_param


########## Functions ##########
# Read image file with gray scale into a numpy ndarray
def readOneImg(img_file_path):
    return cv2.imread(img_file_path, cv2.IMREAD_GRAYSCALE)


# Return a numpy ndarray with shape = (xx, 32, 32) (xx is the # of images in that dir_path)
def readImgs(dir_path):
    imgs_ndarray_list = []
    for file_name in os.listdir(dir_path):
        imgs_ndarray_list.append(readOneImg(dir_path + "/" + file_name))
    return numpy.stack(imgs_ndarray_list, axis=0) # Join a sequence of ndarrays along the first (new) axis


def readPK():
    files_exist = os.path.isfile('train-images.pk')
    files_exist = files_exist and os.path.isfile('train-labels.pk')
    files_exist = files_exist and os.path.isfile('test-images.pk')
    files_exist = files_exist and os.path.isfile('test-labels.pk')
    
    if not files_exist:
        print("Generate PK files.")
        genPKFromSource()
    
    with open('train-images.pk', 'rb') as f1, open('train-labels.pk', 'rb') as f2:
        train_x_ndarray = pickle.load(f1)
        train_y_ndarray = pickle.load(f2)
    with open('test-images.pk', 'rb') as f1, open('test-labels.pk', 'rb') as f2:
        test_x_ndarray = pickle.load(f1)
        test_y_ndarray = pickle.load(f2)
    
    return train_x_ndarray, train_y_ndarray, test_x_ndarray, test_y_ndarray


def genPKFromSource():
    # Read training data
    train_dir_path = './Data_train'
    train_x_part, train_y_part = [], []
    train_x_part += [readImgs(train_dir_path + '/Carambula')]
    train_y_part += [numpy.full((train_x_part[-1].shape[0], ), 0)] # train_x_part[-1].shape[0] is # of images read in
    train_x_part += [readImgs(train_dir_path + '/Lychee')]
    train_y_part += [numpy.full((train_x_part[-1].shape[0], ), 1)]
    train_x_part += [readImgs(train_dir_path + '/Pear')]
    train_y_part += [numpy.full((train_x_part[-1].shape[0], ), 2)]
    train_x_part = numpy.concatenate(train_x_part, axis=0)
    train_y_part = numpy.concatenate(train_y_part, axis=0)
    
    # Read testing data
    test_dir_path = './Data_test'
    test_x_part, test_y_part = [], []
    test_x_part += [readImgs(test_dir_path + '/Carambula')]
    test_y_part += [numpy.full((test_x_part[-1].shape[0], ), 0)] # test_x_part[-1].shape[0] is # of images read in
    test_x_part += [readImgs(test_dir_path + '/Lychee')]
    test_y_part += [numpy.full((test_x_part[-1].shape[0], ), 1)]
    test_x_part += [readImgs(test_dir_path + '/Pear')]
    test_y_part += [numpy.full((test_x_part[-1].shape[0], ), 2)]
    test_x_part = numpy.concatenate(test_x_part, axis=0)
    test_y_part = numpy.concatenate(test_y_part, axis=0)
    
    # Store data into PK files
    with open('train-images.pk', 'wb') as f:
        pickle.dump(train_x_part, f)
    with open('train-labels.pk', 'wb') as f:
        pickle.dump(train_y_part, f)
    with open('test-images.pk', 'wb') as f:
        pickle.dump(test_x_part, f)
    with open('test-labels.pk', 'wb') as f:
        pickle.dump(test_y_part, f)


def shuffleNDA(ndarray_x_part, ndarray_y_part):
    if ndarray_x_part.shape[0] != ndarray_y_part.shape[0]:
        print("Can only shuffle two ndarrays with the same shape[0].")
        exit()
    else:
        shuffle = numpy.random.permutation(ndarray_x_part.shape[0])
        numpy.take(ndarray_x_part, shuffle, axis=0, out=ndarray_x_part) # Apply shuffling in-placed
        numpy.take(ndarray_y_part, shuffle, axis=0, out=ndarray_y_part)


def trainValidSplit(ndarray, v_proportion=CONST.v_proportion()):
    train_amount  = int(ndarray.shape[0] * (1.0-v_proportion))
    
    train_indices = list(range(0, train_amount)) # The list of indices for training dataset
    valid_indices = list(range(train_amount, ndarray.shape[0]))
    
    ndarray_train = numpy.take(ndarray, train_indices, axis=0)
    ndarray_valid = numpy.take(ndarray, valid_indices, axis=0)
    
    return ndarray_train, ndarray_valid


def makeDataBatches(batch_size, data_x_part, data_y_part):
    index_list, start = [], 0
    
    while True:
        if (start+batch_size) > data_x_part.shape[1]:
            break
        index_list.append( (start, start+batch_size) )
        start += batch_size
    index_list.append( (start, data_x_part.shape[1]) )
    
    list_of_list = []
    for index in index_list:
        batch_i = [(data_x_part[:, i].reshape(-1, 1), data_y_part[:, i].reshape(-1, 1)) for i in range(index[0], index[1])]
        list_of_list.append(batch_i)
    
    return list_of_list


def printImg(enable_print_y, data_x_part, data_y_part=None):
    # Print data into a gray image
    pyplot.imshow(data_x_part.reshape(CONST.row_num(), CONST.col_num()), cmap=pyplot.cm.gray)
    if enable_print_y == True: pyplot.title("%s"%str(data_y_part))
    pyplot.show()


def oneHotEncoding(data_y_part):
    result = []
    
    for label in data_y_part:
        temp = [0.0] * 10
        temp[label] = 1.0
        result.append(temp)
    
    result = numpy.asarray(result).transpose()
    
    # Avoid zeroes and ones which may cause gradient vanishing/exploding
    #result[result==0.0] = 0.001
    #result[result==1.0] = 0.999
    
    return result



# Work the same as 'torch.nn.Linear(in_features, out_features, bias=False)'
def linearFP(in_ndarray, weight):
    return (in_ndarray).dot(weight.T)


# Work the same as 'torch.nn.ReLU()'
def reLU(in_ndarray):
    return in_ndarray * (in_ndarray > 0)


# Work the same as 'torch.nn.Softmax(dim)'
def softmax(in_ndarray, dim=None):
    if dim == None:
        return scipy_softmax(in_ndarray, axis=0)
    else:
        return scipy_softmax(in_ndarray, axis=dim)


def linearBP():
    pass