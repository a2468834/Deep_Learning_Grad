#   Homework 1: Neural Networks
#    
#   Date:        2020/10/14
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
import pathlib
import random
import re
from   scipy.special import *
import struct
import sys
import time
import warnings


# Define some constants
class CONST:
    row_num    = lambda : 28                                # height of a image in pixels
    col_num    = lambda : 28                                # width of a image in pixels
    input_dim  = lambda : CONST.row_num() * CONST.col_num()
    output_dim = lambda : 10
    batch_size = lambda : 4096                              # batch size = 2^12
    v_t_ratio  = lambda : 0.3                               # validation_training_ration


# Define some basic functions
class FUNC:
    #sigmoid  = lambda x : 1/(1+numpy.exp(-x))
    unitStep = lambda x : (x > 0.0).astype(numpy.int)                    # 1st Derivative of reLU w.r.t. 'x'
    reLU     = lambda x : x * (x > 0.0)
    
    def crossEntropy(predict_vector, truth_vector):
        #C-E Loss = $-\sum{\mathrm{label}_i{\ }log_{e}(\mathrm{predict}_i)}$, where $i{\in}$ vector's index set
        return sum((-1.0) * truth_vector * numpy.log(predict_vector)) # element-wise multiplication
    
    def binaryIndicator(vector, relation, comp_value):
        if relation not in [">", "<", "="]:
            print("Relation used in binary indicator must be '>', '<', or '='.")
            exit()
        else:
            if relation == ">": return (vector > 0.0).astype(numpy.int)
            elif relation == "<": return (vector < 0.0).astype(numpy.int)
            else: return (vector == 0.0).astype(numpy.int)


# 'network_struct' tells us that the # of nodes at each layer, e.g., [input_dim, 1st_hd_layer, ..., output_dim]
# 'W' : list of weight matrices, which represents each layer's W. Note that W[0] is W1.
# 'b' : list of bias vectors, which represents each layer's b. Note that b[0] is b1.
class MODEL:
    def __init__(self, network_struct, learning_rate, bias=None):
        # Local variables
        self.network_struct = network_struct
        self.learning_rate = learning_rate 
        self.bias = bias
        self.W = initWeights(network_struct) # Initialize weight metrices


def initWeights(network_struct):
    # Initialize parameters within all weight matrices 'weignts' and bias vectors 'bias'.
    numpy.random.seed()
    weight_list, bias = [], []
    
    # input layer to 1st hidden layer (W[0]), hidden layer to hidden layer, and hidden layer to output layer
    for i in range(len(network_struct)-1):
        # Skill from ch03 slide P.42 (pdf P.21)
        weight_list.append( numpy.random.randn(network_struct[i], network_struct[i+1]) / numpy.sqrt(2.0*network_struct[0]) )
        #b.append( numpy.zeros((network_struct[i+1], 1)) )
    
    return weight_list


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
            result = avoidZeroValue(result)
        # Load training sets' labels
        elif file_type == "label":
            magic_num, label_num = struct.unpack(">II", fptr.read(8))
            result = numpy.frombuffer(fptr.read(), dtype=numpy.dtype(numpy.uint8))
            result = oneHotVector(result)
        else:
            print("Error type of files.")
            result = -1
            exit()
    return result


# Return a list of batches [batch_0, batch_1, batch_2,...], where batch_i = (x_part_numpy, y_part_numpy)
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
    flag = True # First iteration of for loop
    
    for label in y_part:
        one_hot_label = (one_hot == int(label)).astype(numpy.float).reshape(10, 1)
        if flag == True: 
            result, flag = one_hot_label, False
        else:
            result = numpy.append(result, one_hot_label, axis=1)
    
    # Avoid zeroes and ones which may cause gradient vanishing/exploding
    #result[result==0.0] = 0.001
    #result[result==1.0] = 0.999
    
    return result


def trainModel(num_epochs, model, train_x_part, train_y_part):
    # STEP1: Split training data into many mini-batches
    train_batch = makeDataBatches(CONST.batch_size(), train_x_part, train_y_part)
    '''
    for each_batch in train_batch:
        for i in range(each_batch['x'].shape[1]):
            printImage(each_batch['x'][:, i], each_batch['y'][:, i])
    '''
    # STEP2: Gradient descent    
    for epoch in range(num_epochs):
        for each_batch in train_batch:
            weight_delta = []
            for instance_index in range(each_batch['x'].shape[1]):
                # STEP2.1: Forward propagation
                h = forwardPropagate(model, each_batch['x'][:, instance_index])
                
                # STEP2.2: Calculate aggregate loss value at this batch
                instance_loss = FUNC.crossEntropy(h[-1]['post_act'], each_batch['y'][:, instance_index])
                
                # STEP2.3: Backward propagation &
                updateWeightsMiniBatch()
                propa_layer       = len(model.network_struct)-1
                back_propa_vector = h[-1] - each_batch['y'] # Derivative of softmax see P.138
                
                while (propa_layer > 0):
                    output_vector = h[propa_layer]   # bar{z_{i+1}}
                    input_vector  = h[propa_layer-1] # bar{z_{i}}
                    
                    tmp = back_propa_vector * FUNC.unitStep(output_vector) # element-wise multiplication
                    tmp = input_vector.dot(tmp.T)
                    
                    model.W[propa_layer-1] -= model.learning_rate * tmp
                    
                    back_propa_vector = model.W[propa_layer-1].dot(back_propa_vector)
                    propa_layer = propa_layer - 1
                
        #print(epoch)
        #exit()
    
    return model


def forwardPropagate(model, data_x_part):
    # 'h' is a list of dicts which stores intermediate values (i.e., pre_activation_value and
    #  post_activation_value) at hidden layers and output layer.
    # NOTE: h[0]['post_act'] = input data x part
    h = [{'pre_act': None, 'post_act': data_x_part.reshape(-1, 1)}]
    
    for layer_index in range(len(model.network_struct)-1):
        if layer_index in list(range(len(model.network_struct)-2)): # hidden layer
            pre_act  = calcPreActive(model.W[layer_index], h[-1]['post_act'])
            post_act = actFunc(pre_act, "hidden") # The activation function is ReLU
            h.append({'pre_act': pre_act, 'post_act': post_act})
        else: # output layer
            pre_act  = calcPreActive(model.W[layer_index], h[-1]['post_act'])
            post_act = actFunc(pre_act, "output") # The activation function is softmax, not cross-entropy
            h.append({'pre_act': pre_act, 'post_act': post_act})
    return h


# Return 'nabla_W'
# 'nabla_W' is a layer-by-layer python-list of numpy arrays, similar to 'model.W'
def backwardPropagate(model, FP_intermediates, truth_y):
    nabla_W = [numpy.zeros(each_W.shape) for each_W in model.W] # Initialization
    
    # BP for output layer
    delta_temp = deActFunc("output", predict_y=FP_intermediates[-1]['post_act'], truth_y=truth_y.reshape(-1, 1)) # ${\delta}^{L}$
    nabla_W[-1] = FP_intermediates[-2]['post_act'].dot(delta_temp.T) # $a^{L-1} {\delta}^{L}$
    
    # BP for hidden layers
    for layer_index in reversed(range(len(model.network_struct)-2)):
        # z^{l}   = FP_intermediates[layer_index+1]['pre_act']
        # w^{l+1} = model.W[layer_index+1]
        # a^{l-1} = FP_intermediates[layer_index]['post_act']
        sigma_prime = deActFunc("hidden", z_vector=FP_intermediates[layer_index+1]['pre_act'])
        delta_temp = (model.W[layer_index+1]).dot(delta_temp) * sigma_prime
        nabla_W[layer_index] = FP_intermediates[layer_index]['post_act'].dot(delta_temp.T)
    
    return nabla_W


# Update NN's weights by applying gradient descent using backward propagation to a single mini-batch.
def updateWeightsMiniBatch(model, FP_intermediates, mini_batch_list):
    batches_num   = len(mini_batch_list)
    total_nabla_W = [numpy.zeros(each_W.shape) for each_W in model.W]
    
    # FP_intermediates 和 one_instance_x 是一一對應，不同的 one_instance_x 有不同的 FP_intermediates
    for one_instance_x, one_instance_y in mini_batch_list:
        one_instance_nabla_W = backwardPropagate(model, FP_intermediates, one_instance_y)
        total_nabla_W = addTwoListOfNumpy(total_nabla_W, one_instance_nabla_W)
    
    model.W = [W - (model.learning_rate / batches_num) * n_W for W, n_W in zip(model.W, total_nabla_W)]


def actFunc(pre_activation_value, layer_type):
    if layer_type not in ["hidden", "output"]:
        print("Error type of layer in neural network: %s"%layer_type)
        exit()
    else: # Column-wisely applying activation function on pre_activation_value
        if layer_type == "hidden":
            post_activation_value = FUNC.reLU(pre_activation_value)
        else: #layer_type == "output"
            post_activation_value = softmax(pre_activation_value, axis=0)
    return post_activation_value


def deActFunc(layer_type, z_vector=None, predict_y=None, truth_y=None):
    if layer_type not in ["hidden", "output"]:
        print("Error type of layer in neural network: %s"%layer_type)
        exit()
    else:
        if layer_type == "hidden":
            return FUNC.unitStep(z_vector)
        else: # layer_type == "output"
            if predict_y.shape != truth_y.shape:
                print("Error: Calaculate derivative of output layer must use 'truth_y' with dim=(-1, 1).")
                exit()
            else:
                return (predict_y - truth_y).reshape(-1, 1) # The derivative of function which is combined softmax and C-E loss


def calcPreActive(weight, data_x_part):
    pre_act_value = (weight.T).dot(data_x_part)
    if len(pre_act_value.shape)==1: pre_act_value = pre_act_value.reshape(-1, 1)
    return pre_act_value


def inference(model, data_x_part):
    one_hot_prediction = forwardPropagate(model, data_x_part)[-1]
    return numpy.argmax(one_hot_prediction, axis=0)


def evaluation(predict_y, truth_y):
    predict_label = numpy.argmax(predict_y, axis=0)
    truth_label = numpy.argmax(truth_y, axis=0)
    return True if predict_label == truth_label else False


def addTwoListOfNumpy(list_of_numpy1, list_of_numpy2):
    result = []
    if len(list_of_numpy1) == len(list_of_numpy2):
        result = [np_array1 + np_array2 for np_array1, np_array2 in zip(list_of_numpy1, list_of_numpy2)]
    else:
        print("Try to add two different length list.")
        exit()
    return result


def avoidZeroValue(numpy_vector):
    # Add 0.01 to all elements of numpy.ndarray to avoid zeros which caused gradient vanishing
    return numpy_vector + 0.01


def printImage(data_x_part, data_y_part):
    # Print data into a gray image
    pyplot.imshow(data_x_part.reshape(28, 28), cmap=pyplot.cm.gray)
    pyplot.title("%s"%str(data_y_part))
    pyplot.show()


# train_x_part : (28*28, 60000) i.e., (784, 60000)
# train_y_part : (10, 60000)
if __name__ == "__main__":
    start_time = time.time()
    warnings.simplefilter("ignore")
    
    print("Open source files.")
    # Prepare absolute paths of input files
    working_dir           = pathlib.Path().absolute()
    train_image_file_path = os.path.join(working_dir, "train-images.idx3-ubyte")
    train_label_file_path = os.path.join(working_dir, "train-labels.idx1-ubyte")
    test_image_file_path  = os.path.join(working_dir, "t10k-images.idx3-ubyte")
    test_label_file_path  = os.path.join(working_dir, "t10k-labels.idx1-ubyte")
    
    print("Load data (may take a few minutes).")
    # Load data from files
    train_x_part, train_y_part = loadData(train_image_file_path, train_label_file_path)
    test_x_part, test_y_part   = loadData(test_image_file_path, test_label_file_path)
    
    print("Shuffle data.")
    # Shuffle data
    shuffle      = numpy.random.permutation(train_x_part.shape[1])
    train_x_part = train_x_part[:, shuffle]
    train_y_part = train_y_part[:, shuffle]
    shuffle      = numpy.random.permutation(test_x_part.shape[1])
    test_x_part  = test_x_part[:, shuffle]
    test_y_part  = test_y_part[:, shuffle]
    
    print("Generate training set & validation set.")
    # Splitting TRAINING data into validation set (30%) and training set (70%)
    valid_num        = int(train_x_part.shape[1] * CONST.v_t_ratio())
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
    
    net_struct = [CONST.row_num()*CONST.col_num(), 100, 100, CONST.output_dim()]
    HW1_NN     = MODEL(network_struct=net_struct, learning_rate=0.1)
    HW1_NN     = trainModel(20000, HW1_NN, train_x_part_P70, train_y_part_P70)
    
    print("Total Exe. Seconds: %.2f"%(time.time()-start_time))
    

# The # of nodes in the input layer is determined by the dimensionality of training data. => input layer 784 neurons
# The # of nodes in the output layer is determined by the number of classes we have. => 10 classes 10 neurons 
# Textbook P.139 : forward & backward function
# `np_ay_1.dot(np_ay_2)` is equivalent to `numpy.dot(np_ay_1, np_ay_2)`
#activations[layer_xx] i.e., FP_intermediates[layer_xx]['post_act']
#zs[layer_xx] i.e., FP_intermediates[layer_xx]['pre_act']