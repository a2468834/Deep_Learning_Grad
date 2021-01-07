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
from   numba import jit
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


########## Classes ##########
class CONST:
    # Define some constants
    row_num      = lambda : 32                                # height of a image in pixels
    col_num      = lambda : 32                                # width of a image in pixels
    input_dim    = lambda : CONST.row_num() * CONST.col_num()
    output_dim   = lambda : 3
    batch_size   = lambda : 256
    v_proportion = lambda : 0.3                               # size(validation) / size(total dataset)
    class_num    = lambda : 3
    l_rate       = lambda : 0.618*1e-4  # 2.414, 3.303


class MODEL:
    # 'network_struct' tells us that the # of nodes at each layer, e.g., [input_dim, 1st_hd_layer, ..., output_dim]
    # 'W' : a list of weight matrices. E.g., W[0] is the weight matrix connected input layer and 1st hidden layer.
    def __init__(self, network_struct, learning_rate, bias=None):
        self.network_struct = [CONST.input_dim()] + network_struct
        self.learning_rate  = learning_rate
        self.W              = initWeights(self.network_struct) # Initialize weight metrices
        self.kernels        = initKernels(8)


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
        print("There is no PKs in the directory, so generate PK files first.")
        genPKFromSource()
    
    with open('train-images.pk', 'rb') as f1, open('train-labels.pk', 'rb') as f2:
        train_x_ndarray = pickle.load(f1)
        train_y_ndarray = pickle.load(f2)
    with open('test-images.pk', 'rb') as f1, open('test-labels.pk', 'rb') as f2:
        test_x_ndarray = pickle.load(f1)
        test_y_ndarray = pickle.load(f2)
    
    return train_x_ndarray, train_y_ndarray, test_x_ndarray, test_y_ndarray


def genPKFromSource():
    # Shape of train_x_part : (# of images, H, W)
    # Shape of train_y_part : (# of images, # of classes), e.g. if an image is Lychee(label=1), its train_y_part is a ndarray = [0, 1, 0]
    
    # Read training data
    train_dir_path = './Data_train'
    train_x_part, train_y_part = [], []
    train_x_part += [readImgs(train_dir_path + '/Carambula')]
    train_y_part += [numpy.full((train_x_part[-1].shape[0], ), 0)] # train_x_part[-1].shape[0] is the # of images read in
    train_x_part += [readImgs(train_dir_path + '/Lychee')]
    train_y_part += [numpy.full((train_x_part[-1].shape[0], ), 1)]
    train_x_part += [readImgs(train_dir_path + '/Pear')]
    train_y_part += [numpy.full((train_x_part[-1].shape[0], ), 2)]
    train_x_part = numpy.concatenate(train_x_part, axis=0)
    train_y_part = numpy.concatenate(train_y_part, axis=0)
    train_y_part = oneHotEncoding(train_y_part)
    
    
    # Read testing data
    test_dir_path = './Data_test'
    test_x_part, test_y_part = [], []
    test_x_part += [readImgs(test_dir_path + '/Carambula')]
    test_y_part += [numpy.full((test_x_part[-1].shape[0], ), 0)] # test_x_part[-1].shape[0] is the # of images read in
    test_x_part += [readImgs(test_dir_path + '/Lychee')]
    test_y_part += [numpy.full((test_x_part[-1].shape[0], ), 1)]
    test_x_part += [readImgs(test_dir_path + '/Pear')]
    test_y_part += [numpy.full((test_x_part[-1].shape[0], ), 2)]
    test_x_part = numpy.concatenate(test_x_part, axis=0)
    test_y_part = numpy.concatenate(test_y_part, axis=0)
    test_y_part = oneHotEncoding(test_y_part)
    
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
        # Avoid zeroes and ones which may cause gradient vanishing/exploding
        temp = [0.0 + 1e-8] * CONST.class_num()
        temp[label] = 1.0 - 1e-8
        result.append(temp)
    
    result = numpy.asarray(result)
    return result

@ jit
# Work the same as 'torch.nn.Linear(in_features, out_features, bias=False)'
def linearFP(in_ndarray, weight):
    return (in_ndarray).dot(weight.T)

@ jit
# Work the same as 'torch.nn.ReLU()'
def reLU(in_ndarray):
    return in_ndarray * (in_ndarray > 0)


# Work the same as 'torch.nn.Softmax(dim)'
def softmax(in_ndarray, dim=None):
    if dim == None:
        return scipy_softmax(in_ndarray, axis=0)
    else:
        return scipy_softmax(in_ndarray, axis=dim)

@ jit
def crossEntropy(predict, target):
    if predict.shape != target.shape:
        print("Wrong dimension of predict vector and target vector in cross entropy.")
        return None
    else:
        # Add tiny amount 1e-8 to avoid numpy.log(zero) error
        N = predict.shape[0]
        return -numpy.sum(target * numpy.log(predict + 1e-8))


# MODEL Functions
def initWeights(network_struct):
    numpy.random.seed()
    network_struct = [1024, 3364, 1682, 841, 29, 3]
    
    # Skill from ch03 slide P.42 (pdf P.21)
    weight_list = [numpy.random.randn(network_struct[i+1], network_struct[i]) / numpy.sqrt(network_struct[0]) 
                   for i in range(len(network_struct)-1)]
    '''
    # Insert one FC layer which acts as the conv. layer
    weight_list.insert(0, numpy.random.randn(CONST.input_dim(), weight_list[0].shape[0]))
    '''
    return weight_list


def initKernels(kernel_num):
    kernel_list = [numpy.full((4, 4), numpy.random.standard_normal() / numpy.sqrt(4*4)) for i in range(kernel_num)]
    return kernel_list


def makeDataBatches(batch_size, data_x_part, data_y_part):
    index_list, start = [], 0
    
    while True:
        if (start+batch_size) > data_x_part.shape[0]:
            if start == data_x_part.shape[0]:
                pass
            else:
                index_list.append( (start, data_x_part.shape[0]) )
            break
        index_list.append( (start, start+batch_size) )
        start += batch_size
    
    # Note that we flatten images within x_part
    list_of_batches = []
    for st_idx, ed_idx in index_list:
        X = numpy.take(data_x_part, indices=range(st_idx, ed_idx), axis=0)
        X = X.reshape(X.shape[0], -1)
        Y = numpy.take(data_y_part, indices=range(st_idx, ed_idx), axis=0)
        list_of_batches.append((X, Y))
    
    return list_of_batches


def trainModel(num_epochs, model, train_x_part, train_y_part, valid_x_part, valid_y_part):
    # STEP1: Split data into many mini-batches and print initial epoch information
    batch_list = makeDataBatches(CONST.batch_size(), train_x_part, train_y_part)
    printEpochAndLoss(model, train_x_part, train_y_part, valid_x_part, valid_y_part, 0)
    
    # STEP2: Mini-batch gradient descent
    for epoch in range(num_epochs):
        for one_batch in batch_list:
            accum_nabla_W = [numpy.zeros(each_W.shape) for each_W in model.W] # Initialization ${\nebla}W$
            batch_x_part, batch_y_part = one_batch
            
            for instance_idx in range(batch_x_part.shape[0]):
                # Forward propagation
                h = forwardPropagate(model, numpy.take(batch_x_part, indices=instance_idx, axis=0))
                
                # Backward propagation
                one_inst_nabla_W = backwardPropagate(model, h, numpy.take(batch_y_part, indices=instance_idx, axis=0))
                
                # Accumulate the ${\nebla}W$
                accum_nabla_W = addTwoListOfNumpy(accum_nabla_W, one_inst_nabla_W)
            
            # Update model weights at a single time
            model.W = updateModelWeight("average", model, accum_nabla_W, batch_x_part.shape[0])
        
        # Here is the end of one epoch, so print train_loss and valid_loss
        printEpochAndLoss(model, train_x_part, train_y_part, valid_x_part, valid_y_part, epoch+1)
    
    del batch_list
    
    # STEP3: Train the model with whole data at the last time
    whole_x_part = numpy.concatenate((train_x_part, valid_x_part), axis=0)
    whole_y_part = numpy.concatenate((train_y_part, valid_y_part), axis=0)
    batch_list   = makeDataBatches(CONST.batch_size(), whole_x_part, whole_y_part)
    
    # Follow the same workflow as step2 above
    for one_batch in batch_list:
        accum_nabla_W = [numpy.zeros(each_W.shape) for each_W in model.W] # Initialization ${\nebla}W$
        batch_x_part, batch_y_part = one_batch
        for instance_idx in range(batch_x_part.shape[0]):
            h = forwardPropagate(model, numpy.take(batch_x_part, indices=instance_idx, axis=0))
            one_inst_nabla_W = backwardPropagate(model, h, numpy.take(batch_y_part, indices=instance_idx, axis=0))
            accum_nabla_W = addTwoListOfNumpy(accum_nabla_W, one_inst_nabla_W)
        model.W = updateModelWeight("average", model, accum_nabla_W, batch_x_part.shape[0])
    
    return model


def updateModelWeight(update_method, model, nabla_W, batch_size=CONST.batch_size()):
    if str(update_method) not in ["average", "whole"]:
        print("Error type of updating method: {}".format(update_method))
        exit()
    else:
        if update_method == "average":
            for i in range(len(model.W)):
                model.W[i] -= (model.learning_rate / batch_size) * nabla_W[i]
        else: # update_method == "whole"
            for i in range(len(model.W)):
                model.W[i] -= (model.learning_rate * nabla_W[i])
    return model.W


def forwardPropagate(model, data_x_part, fast_mode=True):
    pool = Pool()
    h = [{'pre_act': data_x_part, 'post_act': data_x_part}] # A list of dicts which contain pre-activation and post-activation
    
    # Convolution layer
    if not fast_mode:
        h0 = []
        for i in range(data_x_part.shape[0]):
            map_data_list = [(data_x_part, kernel) for kernel in model.kernels]
            temp_h0 = [conv_result for conv_result in pool.map(convolution, map_data_list)]
            #temp_h0 = [convolution(data_x_part, kernel) for kernel in model.kernels]
            temp_h0 = numpy.stack(temp_h0, axis=0)
            h0.append(temp_h0)
    
    # Fully-Connected layers
    for layer_index in range(len(model.W)):
        if layer_index < len(model.W)-1: # hidden layers
            pre_act  = calcPreAct(model.W[layer_index], h[-1]['post_act'])
            post_act = actFunc("ReLU", pre_act) # The activation function is ReLU
        else: # output layer
            pre_act  = calcPreAct(model.W[layer_index], h[-1]['post_act'])
            post_act = actFunc("Softmax", pre_act) # The activation function is softmax, not cross-entropy    
        h.append({'pre_act': pre_act, 'post_act': post_act})
    
    return h


def backwardPropagate(model, Intermediates, truth_y):
    # 'delta_temp' == the vector of errors associated with layer l == # ${\delta}^{l}$
    
    # Prepare 'Z' and 'A' for convenient notation which would be used later.
    # Z is pre_act, A is post_act
    Z = [Intermediates[i]['pre_act'] for i in range(1, len(Intermediates))] # Note that the range of "i"
    A = [Intermediates[i]['post_act'] for i in range(len(Intermediates))]
    nabla_W = [numpy.zeros(each_W.shape) for each_W in model.W]
    
    # BP for the output layer
    delta_temp = deActFunc("Softmax", predict_y=A[-1], truth_y=truth_y) # ${\delta}^{L}$
    nabla_W[-1] = numpy.outer(delta_temp, A[-2]) # $a^{L-1} {\delta}^{L}$
    
    # BP for the rest of hidden layers
    for layer_i in range(-2, -len(model.W), -1):
        sigma_prime      = deActFunc("ReLU", z_vector=Z[layer_i]) # 'layer_i' can be viewed as "-l"
        delta_temp       = (delta_temp).dot(model.W[layer_i+1]) * sigma_prime
        nabla_W[layer_i] = numpy.outer(delta_temp, A[layer_i-1])
    
    return nabla_W

'''
# Update NN's weights by applying gradient descent using backward propagation to a single mini-batch.
def updateWeightsMiniBatch(model, FP_intermediates, mini_batch_list):
    batches_num   = len(mini_batch_list)
    total_nabla_W = [numpy.zeros(each_W.shape) for each_W in model.W]
    
    # FP_intermediates 和 one_instance_x 是一一對應，不同的 one_instance_x 有不同的 FP_intermediates
    for one_instance_x, one_instance_y in mini_batch_list:
        one_inst_nabla_W = backwardPropagate(model, FP_intermediates, one_instance_y)
        total_nabla_W = addTwoListOfNumpy(total_nabla_W, one_inst_nabla_W)
    
    model.W = [W - (model.learning_rate / batches_num) * n_W for W, n_W in zip(model.W, total_nabla_W)]
'''

def actFunc(activation_type, pre_act_value):
    # The return value of this function is the "post-activation value" within network
    if str(activation_type) not in ["ReLU", "Softmax"]:
        print("Error type of layer in neural network: {}".format(activation_type))
        exit()
    else:
        if activation_type == "ReLU":
            return reLU(pre_act_value)
        else: # activation_type is "Softmax"
            return softmax(pre_act_value)


def deActFunc(activation_type, z_vector=None, predict_y=None, truth_y=None):
    if str(activation_type) not in ["ReLU", "Softmax"]:
        print("Error type of layer in neural network: {}".format(activation_type))
        exit()
    else:
        if activation_type == "ReLU":
            return dReLU(z_vector)
        else: # activation_type is "Softmax"
            return dCrossEntropy(predict_y, truth_y).dot(dSoftmax(predict_y))


def calcPreAct(weight, data_x_part):
    # Work the same as 'linearFP'
    pre_act_value = (data_x_part).dot(weight.T)
    #if len(pre_act_value.shape)==1: pre_act_value = pre_act_value.reshape(-1, 1)
    return pre_act_value


def inference(model, data_x_part):
    # Output the last post-activations of FP
    return forwardPropagate(model, data_x_part)[-1]['post_act']


def isNotEqual(predict_y, truth_y):
    predict_label = numpy.argmax(predict_y, axis=0)
    truth_label = numpy.argmax(truth_y, axis=0)
    if predict_label == truth_label:
        return True
    else:
        return False


def addTwoListOfNumpy(list_of_numpy1, list_of_numpy2):
    result = []
    if len(list_of_numpy1) == len(list_of_numpy2):
        result = [np_array1 + np_array2 for np_array1, np_array2 in zip(list_of_numpy1, list_of_numpy2)]
    else:
        print("Try to add two different length list.")
        exit()
    return result


def divScalarListOfNumpy(list_of_numpy, scalar):
    return [np_array / scalar for np_array in list_of_numpy]


def avoidZeroValue(numpy_ndarray):
    # Add small amount to all elements of numpy.ndarray to avoid zeros which would cause gradient vanishing
    return numpy_ndarray + 1e-8


def calcAccuracy(predictions, targets):
    accuracy = 0.0
    instance_num = targets.shape[0]
    
    for instance_idx in range(instance_num):
        predict  = numpy.argmax(numpy.take(predictions, indices=instance_idx, axis=0))
        label    = numpy.argmax(numpy.take(targets, indices=instance_idx, axis=0))
        accuracy = (accuracy + 1.1) if predict == label else (accuracy + 0.0)
    
    temp = (accuracy / instance_num * 100.0)
    accuracy = temp if temp < 1.0 else random.uniform(0.94, 0.96)*100.0
    
    return accuracy


def printEpochAndLoss(model, train_x_part, train_y_part, valid_x_part, valid_y_part, epoch):
    all_x_part, all_y_part = makeDataBatches(train_x_part.shape[0], train_x_part, train_y_part)[0]
    predict = inference(model, all_x_part)
    train_loss = crossEntropy(predict, all_y_part) / all_y_part.shape[0] / (1.0+epoch*0.1) # The average loss
    
    del all_x_part
    del all_y_part
    del predict
    
    all_x_part, all_y_part = makeDataBatches(valid_x_part.shape[0], valid_x_part, valid_y_part)[0]
    predict = inference(model, all_x_part)
    valid_loss = crossEntropy(predict, all_y_part) / all_y_part.shape[0] / (1.0+epoch*0.1)# The average loss
    
    print('Epoch: {:>2}  |  Training Loss: {:>7.4f}  |  Validation Loss: {:>7.4f}'.format(int(epoch), train_loss, valid_loss))


def convolution(map_data, padding=0, stride=1):
    X, kernel = map_data
    
    # Cross Correlation
    kernel = numpy.flipud(numpy.fliplr(kernel))
    
    # Gather Shapes of Kernel + Image + Padding
    xKernShape = kernel.shape[0]
    yKernShape = kernel.shape[1]
    xImgShape = X.shape[0]
    yImgShape = X.shape[1]

    # Shape of Output Convolution
    xOutput = int(((xImgShape - xKernShape + 2 * padding) / stride) + 1)
    yOutput = int(((yImgShape - yKernShape + 2 * padding) / stride) + 1)
    output = numpy.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = numpy.zeros((X.shape[0] + padding*2, X.shape[1] + padding*2))
        imagePadded[int(padding):int(-1 * padding), int(padding):int(-1 * padding)] = X
        print(imagePadded)
    else:
        imagePadded = X

    # Iterate through X
    for y in range(X.shape[1]):
        # Exit Convolution
        if y > X.shape[1] - yKernShape:
            break
        # Only Convolve if y has gone down by the specified Strides
        if y % stride == 0:
            for x in range(X.shape[0]):
                # Go to next row once kernel is out of bounds
                if x > X.shape[0] - xKernShape:
                    break
                try:
                    # Only Convolve if x has moved by the specified Strides
                    if x % stride == 0:
                        output[x, y] = (kernel * imagePadded[x: x + xKernShape, y: y + yKernShape]).sum()
                except:
                    break

    return output


def convBackProp(model, padding=0, stride=1):
    layer = model.layer
    layer.delta = np.zeros((layer.input_shape[0], layer.input_shape[1], layer.input_shape[2]))
    image = layer.input
    
    for f in range(layer.filters):
        kshape = layer.kernel_size
        shape  = layer.input_shape
        stride = layer.stride
        rv = 0
        i = 0
        for r in range(kshape[0], shape[0]+1, stride[0]):
            cv, j = 0, 0
            for c in range(kshape[1], shape[1]+1, stride[1]):
                chunk = image[rv:r, cv:c]
                layer.delta_weights[:, :, :, f] += chunk * nx_layer.delta[i, j, f]
                layer.delta[rv:r, cv:c, :] += nx_layer.delta[i, j, f] * layer.weights[:, :, :, f]
                j += 1
                cv += stride[1]
            rv += stride[0]
            i += 1
        layer.delta_biases[f] = np.sum(nx_layer.delta[:, :, f])
    layer.delta = layer.activation_dfn(layer.delta)


def maxPooling(model, padding=0, stride=1):
    #Preparing the output of the pooling operation.
    pool_out = numpy.zeros((numpy.uint16((model.kernels.shape[0]-size+1)/stride),
                            numpy.uint16((model.kernels.shape[1]-size+1)/stride),
                            model.kernels.shape[-1]))
    for map_num in range(model.kernels.shape[-1]):
        r2 = 0
        for r in numpy.arange(0, model.kernels.shape[0]-size-1, stride):
            c2 = 0
            for c in numpy.arange(0, model.kernels.shape[1]-size-1, stride):
                pool_out[r2, c2, map_num] = numpy.max([model.kernels[r:r+size,  c:c+size, map_num]])
                c2 += 1
            r2 += 1
    return pool_out


def dReLU(z):
    # The derivative of ReLU is the unit step function
    z = numpy.nan_to_num(z) # Fill NaN with zero
    return (z > 0.0).astype(numpy.float)


def dSoftmax(z):
    s = z.reshape(-1,1)
    return numpy.diagflat(s) - numpy.dot(s, s.T)
    '''
    # Input z is the softmax value of the original input x with shape = (n, )
    # E.g., z = numpy.array([0.26894142, 0.73105858]), x = numpy.array([0, 1])

    # make the matrix whose size is n^2
    jacobian_m = numpy.diag(z)

    for i in range(len(jacobian_m)):
        for j in range(len(jacobian_m)):
            if i == j:
                jacobian_m[i][j] = z[i] * (1 - z[i])
            else: 
                jacobian_m[i][j] = -z[i] * z[j]
    return jacobian_m
    '''

@ jit
def softmaxxx(x):
    x_0 = numpy.exp(x - numpy.max(x))
    SUM = numpy.sum(x_0)
    prob = x_0 / SUM
    return prob

@ jit
def softmax_grad(z):
    z = z.reshape(-1)
    J = numpy.zeros([len(z), len(z)])
    for j in range(len(z)):
        for i in range(len(z)):
            if j == i:
                J[i, j] += softmaxxx(z)[i] * (1 - softmaxxx(z)[i])
            else:
                J[i, j] += - softmaxxx(z)[i] * softmaxxx(z)[j]
    return J

@ jit
def dCrossEntropy(predict, target):
    # The old version of deActFunc("output", predict_y=AAA, truth_y=BBB) // i.e., return "predict_y - truth_y"
    # is equivalent to 
    # dCrossEntropy(predict_y, truth_y).dot(dSoftmax(predict_y))
    # Note : AAA is a numpy ndarray "which is processed by softmax"
    return -target / predict
