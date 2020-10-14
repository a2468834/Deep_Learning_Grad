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
    row_num = lambda : 28
    col_num = lambda : 28


# Define some basic functions
class FUNC:
    Sigmoid = lambda x : 1/(1+numpy.exp(-x))
    ReLU    = lambda x : x if x > 0 else 0
    Softmax = lambda x : numpy.exp(x) / numpy.sum(numpy.exp(x))


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
    
    # Shuffle data
    shuffle = numpy.random.permutation(train_x_part.shape[0])
    train_x_part = train_x_part[shuffle, :]
    train_y_part = train_y_part[shuffle]
    shuffle = numpy.random.permutation(test_x_part.shape[0])
    test_x_part = test_x_part[shuffle, :]
    test_y_part = test_y_part[shuffle]
    
    # Splitting TRAINING data into validation set (30%) and training set (70%)
    valid_num = int(train_x_part.shape[0] * 0.3)
    train_x_part_P30 = train_x_part[:valid_num, :]
    train_x_part_P70 = train_x_part[valid_num:, :]
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
    
    # Print data into gray images
    for i in range(0, 10):
        r = random.randint(0, 10000-1)
        print(r)
        pyplot.imshow(test_x_part[r, :].reshape(28, 28), cmap=pyplot.cm.gray)
        pyplot.title("%s"%(test_y_part[r]))
        pyplot.show()