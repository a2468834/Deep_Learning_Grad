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
import traceback
import warnings


# Define some constants
class CONST:
    row_num    = lambda : 28                                # height of a image in pixels
    col_num    = lambda : 28                                # width of a image in pixels
    input_dim  = lambda : CONST.row_num() * CONST.col_num()
    output_dim = lambda : 10
    batch_size = lambda : 4096


# Define some basic functions
class FUNC:
    #Sigmoid  = lambda x : 1/(1+numpy.exp(-x))
    Softmax  = lambda x : numpy.exp(x) / numpy.sum(numpy.exp(x), axis=0) # Apply softmax column-wisely
    UnitStep = lambda x : (x > 0.0).astype(numpy.int)                    # 1st Derivative of ReLU w.r.t. 'x'
    ReLU     = lambda x : x * (x > 0.0)
    
    def CrossEntropy(predict_prob_vector, truth):
        return (-1.0) * numpy.log( sum(predict_prob_vector * truth) ) # element-wise multiplication
    
    def BinaryIndicator(vector, relation, comp_value):
        if relation not in [">", "<", "="]:
            print("Relation used in binary indicator must be '>', '<', or '='.")
            exit()
        else:
            if relation == ">": return (vector > 0.0).astype(numpy.int)
            elif relation == "<": return (vector < 0.0).astype(numpy.int)
            else: return (vector == 0.0).astype(numpy.int)


def loadData(image_file_path, label_file_path):
    # Every COLUMN (NOT row) in 'x_part' is a training/testing instance.
    # Turn labels in y_part into one-hot vectors.
    return readImageLabel(image_file_path, "image"), readImageLabel(label_file_path, "label")


def readImageLabel(file_path, file_type):
    result = None
    with open(file_path, 'rb') as fptr:
        # Load training sets' images
        if file_type == "image":
            magic_num, image_num = struct.unpack(">IIII", fptr.read(16))[0:2]
            result = numpy.frombuffer(fptr.read(), dtype=numpy.dtype(numpy.uint8))
            result = result.reshape(CONST.row_num()*CONST.col_num(), -1, order='F') # reshape to dim = (28*28, xxx)
        # Load training sets' labels
        elif file_type == "label":
            magic_num, label_num = struct.unpack(">II", fptr.read(8))
            result = numpy.frombuffer(fptr.read(), dtype=numpy.dtype(numpy.uint8))
            result = oneHotVector(result)
        else:
            print("Error type of files.")
            result = -1
            exit()
    # Avoid zeroes which may cause gradient vanishing
    return (result + 0.01)


# Return a list of batches [batch_0, batch_1, batch_2,...], where batch_i = (x_part, y_part)
def makeDataBatches(batch_size, data_x_part, data_y_part):
    index_list, start = [], 0
    
    while True:
        if (start+batch_size) > data_x_part.shape[1]:
            break
        index_list.append( (start, start+batch_size) )
        start += batch_size
    index_list.append( (start, data_x_part.shape[1]) )
    
    return [(data_x_part[:, index[0]:index[1]], data_y_part[:, index[0]:index[1]]) for index in index_list]


def oneHotVector(y_part):
    one_hot = numpy.arange(10)
    flag = True # Initial condition flag
    
    for label in y_part:
        one_hot_label = (one_hot == int(label)).astype(numpy.int).reshape(10, 1)
        if flag == True: 
            result, flag = one_hot_label, False
        else:
            result = numpy.append(result, one_hot_label, axis=1)
    
    # Avoid zeroes and ones which may cause gradient vanishing/exploding
    result[result==0], result[result==1] = 0.01, 0.99
    
    return result


# 'network_struct' tells us that the # of nodes at each layer, e.g., [input_dim, 1st_hd_layer, ..., output_dim]
# 'W' : list of weight matrices, which represents each layer's W. Note that W[0] is W1.
# 'b' : list of bias vectors, which represents each layer's b. Note that b[0] is b1.
# 'model' : return value which is a dict with all of 'W' and 'b'.
def trainModel(num_epochs, network_struct, learning_rate, train_x_part, train_y_part):
    W, b, model = [], [], {}
    numpy.random.seed()
    
    # STEP1: Initialize parameters within all weight matrices 'W' and bias vectors 'b'.
    # input layer to 1st hidden layer, hidden layer to hidden layer, and hidden layer to output layer
    for i in range(len(network_struct)-1):
        W.append( numpy.random.randn(network_struct[i], network_struct[i+1]) / numpy.sqrt(network_struct[0]) ) # Skill from ch03 slide P.42
        #b.append( numpy.zeros((network_struct[i+1], 1)) )
    
    # STEP2: Gradient descent, batch size = 2^12 = 4096
    train_batch = makeDataBatches(CONST.batch_size(), train_x_part, train_y_part)
    
    for epoch in range(num_epochs):
        # each_batch = (x_part, y_part)
        for each_batch in train_batch:
            # STEP2.1: Forward propagation
            h = [each_batch[0]]
            for layer_i in range(len(network_struct)-1):
                if layer_i in list(range(0, len(network_struct)-2)):
                    h.append(FUNC.ReLU( (W[layer_i].T).dot(h[-1]) )) # Column-wisely applying ReLU
                # Output layer using softmax activation function
                else:
                    h.append(FUNC.Softmax( (W[layer_i].T).dot(h[-1]) ))
            
            # STEP2.2: Calculate aggregate loss value at this batch
            batch_loss = sum([FUNC.CrossEntropy(h[-1][:, index], each_batch[1][:, index]) for index in range(each_batch[1].shape[1])])
            
            # STEP2.3: Backward propagation
            propa_layer       = len(network_struct)-1
            back_propa_vector = h[-1] - each_batch[1]
            
            while (propa_layer > 0):
                output_vector = h[propa_layer]   # bar{z_{i+1}}
                input_vector  = h[propa_layer-1] # bar{z_{i}}
                
                tmp = back_propa_vector * FUNC.UnitStep(output_vector) # element-wise multiplication
                tmp = input_vector.dot(tmp.T)
                
                W[propa_layer-1] -= learning_rate * tmp
                
                back_propa_vector = W[propa_layer-1].dot(back_propa_vector)
                propa_layer = propa_layer - 1
        print(epoch)
        exit()
    
    return model


# train_x_part : (28*28, 60000) i.e., (784, 60000)
# train_y_part : (10, 60000)
if __name__ == "__main__":
    warnings.simplefilter("ignore")
    print("Open source files.")
    # Prepare absolute paths of input files
    working_dir = pathlib.Path().absolute()
    train_image_file_path = os.path.join(working_dir, "train-images.idx3-ubyte")
    train_label_file_path = os.path.join(working_dir, "train-labels.idx1-ubyte")
    test_image_file_path = os.path.join(working_dir, "t10k-images.idx3-ubyte")
    test_label_file_path = os.path.join(working_dir, "t10k-labels.idx1-ubyte")
    
    print("Load data.")    
    # Load data from files
    train_x_part, train_y_part = loadData(train_image_file_path, train_label_file_path)
    test_x_part, test_y_part = loadData(test_image_file_path, test_label_file_path)
    
    print("Generate training set & validation set.")    
    # Shuffle data
    shuffle = numpy.random.permutation(train_x_part.shape[1])
    train_x_part = train_x_part[:, shuffle]
    train_y_part = train_y_part[:, shuffle]
    shuffle = numpy.random.permutation(test_x_part.shape[1])
    test_x_part = test_x_part[:, shuffle]
    test_y_part = test_y_part[:, shuffle]
    
    # Splitting TRAINING data into validation set (30%) and training set (70%)
    valid_num = int(train_x_part.shape[1] * 0.3)
    train_x_part_P30 = train_x_part[:, :valid_num]
    train_x_part_P70 = train_x_part[:, valid_num:]
    train_y_part_P30 = train_y_part[:, :valid_num]
    train_y_part_P70 = train_y_part[:, valid_num:]
    
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
        pyplot.title("%s"%(train_y_part[:, r]))
        pyplot.show()
    '''
    
    network_struct = [CONST.row_num()*CONST.col_num(), 100, 100, CONST.output_dim()]
    trainModel(20000, network_struct, 0.1, train_x_part_P70, train_y_part_P70)


# The # of nodes in the input layer is determined by the dimensionality of training data. => input layer 784 neurons
# The # of nodes in the output layer is determined by the number of classes we have. => 10 classes 10 neurons 
# Textbook P.139 : forward & backward function
