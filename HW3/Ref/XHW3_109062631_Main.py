#  Homework 3:  CNN from Scratch - Main
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
from HW3_109062631_Module import *
########## Main function ##########
if __name__ == "__main__":
    # Print numpy ndarray without truncation
    numpy.set_printoptions(threshold=sys.maxsize)
    
    print("Load data from pickles.")
    train_ndarray, test_ndarray = {}, {}
    train_ndarray['X'], train_ndarray['Y'], test_ndarray['X'], test_ndarray['Y'] = readPK()
    
    print("Shuffle data.")
    shuffleNDA(train_ndarray['X'], train_ndarray['Y']) # Apply shuffling in-placed
    shuffleNDA(test_ndarray['X'], test_ndarray['Y'])
       
    print("Generate training set & validation set.") # training=70%, validation=30%
    train_ndarray_P70, train_ndarray_P30 = {}, {}
    train_ndarray_P70['X'], train_ndarray_P30['X']= trainValidSplit(train_ndarray['X'], CONST.v_proportion())
    train_ndarray_P70['Y'], train_ndarray_P30['Y']= trainValidSplit(train_ndarray['Y'], CONST.v_proportion())




########## Other Notes ##########
# y_part label : 0 == Carambula, 1 == Lychee, 2 == Pear
'''
for i in range(5):
    print(index := random.randint(0, test_x_part.shape[0]))
    printImg(True, test_x_part[index, :, :], test_y_part[index])
'''