#   Homework 2: Convolutional Autoencoder - Module
#    
#   Date:        2020/11/06
#   CourseID:    10910COM526000
#   Course:      Deep Learning (Graduated)
#   
#   Writer_ID:   109062631
#   Writer_Name: Wang, Chuan-Chun
#   Environment: 
#      Software: Python 3.8.5 on 64-bit Windows 10 Pro (2004)
#      Hardware: Intel i7-10510U, 16GB DDR4 non-ECC ram, and no discrete GPU
from HW2_109062631_Module import *

########## Main function ##########
if __name__ == "__main__":
    data_tensor, label_tensor = readNPY()
    CAE = convAutoEncdr()
    if if_use_gpu:
        CAE = CAE.cuda()

########## Other Notes ##########
# data_tensor  : (# of images) x (image width) x (image height) x (3-channel)
# label_tensor : (# of images) x (label)
# 3-channel : boundary T/F, normal T/F, defect T/F
# label is nine-class