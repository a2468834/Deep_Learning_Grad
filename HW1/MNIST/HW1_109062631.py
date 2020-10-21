#   Homework 1: Neural Networks
#    
#   Date:        2020/10/14
#   CourseID:    10910COM526000
#   Course:      Deep Learning (Graduated)
#   
#   Writer_ID:   109062631
#   Writer_Name: Wang, Chuan-Chun
#   Environment: Python 3.8.5 on Windows 10(2004) with Intel Core i7-10510U
import math
import matplotlib.pyplot as pyplot
import numpy
import os
import pathlib
import random
import re
import struct
import sys
# For using t-SNE algorithm
import sklearn.manifold as manifold
import sklearn.datasets as datasets
import time

# Define some constants
class CONST:
    row_num    = lambda : 28
    col_num    = lambda : 28
    input_dim  = lambda : CONST.row_num()*CONST.col_num()
    output_dim = lambda : 10


# Define some basic functions
class FUNC:
    Sigmoid  = lambda x : 1/(1+numpy.exp(-x))
    Softmax  = lambda x : numpy.exp(x) / numpy.sum(numpy.exp(x), axis=0) # Apply softmax column-wisely
    UnitStep = lambda x : 1 if x >= 0 else 0                             # Derivative of ReLU w.r.t. 'x'
    ReLU     = lambda x : x * (x > 0)
    Trans    = lambda x : x.transpose()                                  # Please note that 'x' must be a numpy-array


def loadData(image_file_path, label_file_path):
    return readImageLabel(image_file_path, "image"), readImageLabel(label_file_path, "label")


def readImageLabel(file_path, file_type):
    result = None
    with open(file_path, 'rb') as fptr:
        # Load training sets' images
        # 'row_num' means the height of one image in pixels, and 'col_num' means width in pixels.
        if file_type == "image":
            magic_num, image_num = struct.unpack(">IIII", fptr.read(16))[0:2]
            result = numpy.frombuffer(fptr.read(), dtype=numpy.dtype(numpy.uint8))
            result = result.reshape(-1, CONST.row_num()*CONST.col_num()) # reshape to dim = (xxx, 28*28)
        # Load training sets' labels
        elif file_type == "label":
            magic_num, label_num = struct.unpack(">II", fptr.read(8))
            result = numpy.frombuffer(fptr.read(), dtype=numpy.dtype(numpy.uint8))
        else:
            print("Error type of files.")
            exit()
            result = -1
    return result


# 'network_struct' tells us that the # of nodes at each layer, e.g., [input_dim, 1st_hd_layer, ..., output_dim]
def trainModel(network_struct, num_epochs, train_x_part, train_y_part):
    # 'W' : list of weight matrices, which represents each layer's W. Note that W[0] is W1.
    # 'b' : list of bias vectors, which represents each layer's b. Note that b[0] is b1.
    # 'model' : return value which is a dict with all of 'W' and 'b'.
    W, b = [], []
    model = {}
    numpy.random.seed()
    
    # STEP1: Initialize parameters within all weight matrices 'W' and bias vectors 'b'.
    # input layer -> 1st hidden layer, hidden layer -> hidden layer, and hidden layer -> output layer
    for i in range(len(network_struct)-1):
        W.append( numpy.random.randn(network_struct[i], network_struct[i+1]) / numpy.sqrt(network_struct[0]) ) # Skill from ch03 slide P.42
        b.append( numpy.zeros((network_struct[i+1], 1)) )
    
    # STEP2: Gradient descent
    for epoch in range(0, num_epochs):
        # STEP2.1: Forward propagation
        h = [FUNC.ReLU( (FUNC.Trans(W[0])).dot(train_x_part) + b[0] )] # Column-wise addition
        for layer_i in range(1, len(network_struct)-1):
            if (layer_i) < (len(network_struct)-2):
                h.append(FUNC.ReLU( (FUNC.Trans(W[layer_i])).dot(h[layer_i-1]) + b[layer_i] ))
            # Output layer using softmax activation function
            else:
                h.append(FUNC.Softmax( (FUNC.Trans(W[layer_i])).dot(h[layer_i-1]) + b[layer_i] ))
        
        #STEP2.2: Backward propagation
        
    return model


if __name__ == "__main__":
    # Prepare absolute paths of input files
    working_dir = pathlib.Path().absolute()
    train_image_file_path = os.path.join(working_dir, "train-images.idx3-ubyte")
    train_label_file_path = os.path.join(working_dir, "train-labels.idx1-ubyte")
    test_image_file_path = os.path.join(working_dir, "t10k-images.idx3-ubyte")
    test_label_file_path = os.path.join(working_dir, "t10k-labels.idx1-ubyte")
    
    # Load data from files
    train_x_part, train_y_part = loadData(train_image_file_path, train_label_file_path)
    test_x_part, test_y_part = loadData(test_image_file_path, test_label_file_path)
    
    # Transpose matrices so that every COLUMN (NOT row) in x_part is a training/testing instance.
    train_x_part = FUNC.Trans(train_x_part)
    test_x_part  = FUNC.Trans(test_x_part)
    
    # Shuffle data
    shuffle = numpy.random.permutation(train_x_part.shape[1])
    train_x_part = train_x_part[:, shuffle]
    train_y_part = train_y_part[shuffle]
    shuffle = numpy.random.permutation(test_x_part.shape[1])
    test_x_part = test_x_part[:, shuffle]
    test_y_part = test_y_part[shuffle]
    
    # Splitting TRAINING data into validation set (30%) and training set (70%)
    valid_num = int(train_x_part.shape[1] * 0.3)
    train_x_part_P30 = train_x_part[:, :valid_num]
    train_x_part_P70 = train_x_part[:, valid_num:]
    train_y_part_P30 = train_y_part[:valid_num]
    train_y_part_P70 = train_y_part[valid_num:]
    
    '''
    t_sne = manifold.TSNE(n_components=2, random_state=CONST.row_num())
    print("%%%%")
    t_sne_x_part = t_sne.fit_transform(train_x_part, train_y_part)
    print("@@")
    x_min, x_max = t_sne_x_part.min(0), t_sne_x_part.max(0)
    t_sne_x_part = (t_sne_x_part - x_min) / (x_max - x_min)
    pyplot.figure(figsize=(8, 8))
    for i in range(t_sne_x_part.shape[0]):
        pyplot.text(t_sne_x_part[i, 0], t_sne_x_part[i, 1], str(train_y_part[i]), color=pyplot.cm.Set1(train_y_part[i]), fontdict={'weight': 'bold', 'size': 9})
    pyplot.xticks([])
    pyplot.yticks([])
    pyplot.show()
    '''
    
    '''
    # Print data into gray images
    for i in range(0, 20):
        r = random.randint(0, 10000-1)
        print(r)
        pyplot.imshow(train_x_part[:, r].reshape(28, 28), cmap=pyplot.cm.gray)
        pyplot.title("%s"%(train_y_part[r]))
        pyplot.show()
    '''
    
    network_struct = [CONST.row_num()*CONST.col_num(), 100, 100, CONST.output_dim()]
    trainModel(network_struct, 20000, train_x_part, train_y_part)
    
        


# The # of nodes in the input layer is determined by the dimensionality of training data. => input layer 784 neurons
# The # of nodes in the output layer is determined by the number of classes we have. => 10 classes 10 neurons 