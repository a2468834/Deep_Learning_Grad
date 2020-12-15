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
    
    files_exist = os.path.isfile('train-images.pk')
    files_exist = files_exist and os.path.isfile('train-labels.pk')
    files_exist = files_exist and os.path.isfile('test-images.pk')
    files_exist = files_exist and os.path.isfile('test-labels.pk')
    if not files_exist:
        print("Generate PK files.")
        genPKFromSource()
    
    print("Load data from pickles.")
    with open('train-images.pk', 'rb') as f:
        train_x_part = pickle.load(f)
    with open('train-labels.pk', 'rb') as f:
        train_y_part = pickle.load(f)
    with open('test-images.pk', 'rb') as f:
        test_x_part = pickle.load(f)
    with open('test-labels.pk', 'rb') as f:
        test_y_part = pickle.load(f)
    
    
    
    
########## Other Notes ##########
# y_part label : 0 == Carambula, 1 == Lychee, 2 == Pear
'''
for i in range(5):
    print(index := random.randint(0, 498))
    printImg(True, test_x_part[index, :, :], test_y_part[index])
'''