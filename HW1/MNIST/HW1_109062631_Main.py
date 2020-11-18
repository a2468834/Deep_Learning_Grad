#   Homework 1: Neural Networks - Main
#   
#   Date:        2020/10/16
#   CourseID:    10910COM526000
#   Course:      Deep Learning (Graduated)
#   
#   Writer_ID:   109062631
#   Writer_Name: Wang, Chuan-Chun
#   Environment: 
#      Software: Python 3.8.5 on 64-bit Windows 10 Pro (2004)
#      Hardware: Intel i7-10510U, 16GB DDR4 non-ECC ram, and no discrete GPU
from HW1_109062631_Module import *

########## Main function ##########
if __name__ == "__main__":
    warnings.simplefilter("always")
    start_time = time.time()
    #generatePkFromSource()
    
    print("Load data from .pickle.")
    with open('train-images.pickle', 'rb') as fp: train_x_part = pickle.load(fp)
    with open('train-labels.pickle', 'rb') as fp: train_y_part = pickle.load(fp)
    with open('test-images.pickle', 'rb') as fp: test_x_part = pickle.load(fp)
    with open('test-labels.pickle', 'rb') as fp: test_y_part = pickle.load(fp)
    
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
    # Print data into gray images
    for i in range(0, 20):
        r = random.randint(0, 10000-1)
        print(r)
        pyplot.imshow(train_x_part[:, r].reshape(28, 28), cmap=pyplot.cm.gray)
        pyplot.title("%s"%(train_y_part[:, r]))
        pyplot.show()
    '''
    
    nn_struct = [CONST.input_dim(), 7, 3, CONST.output_dim()]
    HW1_NN    = MODEL(network_struct=nn_struct, learning_rate=0.1)
    
    print("Start training a model.\n")
    HW1_NN = trainModel(10, HW1_NN, train_x_part_P70, train_y_part_P70, train_x_part_P30, train_y_part_P30)
    print("\nEnd training.")
    
    print("\nPredict testing data set.\n")
    predictions = inference(HW1_NN, test_x_part)
    
    '''
    print("[Result]")
    temp = 0
    for i in range(predictions.shape[1]):
        if isNotEqual(predictions[:, i], test_y_part[:, i]):
            temp = temp + 1
        else:
            temp = temp
    print("Accuracy on test data: %.4f%%"%((1.0-temp/predictions.shape[1])*100.0))
    print("Total Exe. Seconds: %.2f"%(time.time()-start_time))
    '''


########## Other Notes ##########
# The # of nodes in the input layer is determined by the dimensionality of training data. => input layer 784 neurons
# The # of nodes in the output layer is determined by the number of classes we have. => 10 classes 10 neurons 
# Textbook P.139 : forward & backward function
# `np_ay_1.dot(np_ay_2)` is equivalent to `numpy.dot(np_ay_1, np_ay_2)`
# activations[layer_xx] i.e., FP_intermediates[layer_xx]['post_act']
# zs[layer_xx] i.e., FP_intermediates[layer_xx]['pre_act']
# train_x_part : (28*28, 60000) i.e., (784, 60000)
# train_y_part : (10, 60000)
