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
from   HW3_109062631_Module import *
import torch
from   torch import optim
import torch.nn
import torch.nn.functional
from   torch.utils.data import DataLoader
from   torchvision import datasets


class myCNN(torch.nn.Module):
    def __init__(self, input_chs):
        super(myCNN, self).__init__()
        self.conv1 = torch.nn.Sequential(
                         torch.nn.Conv2d(in_channels=input_chs, out_channels=16, kernel_size=3, padding=1),
                         torch.nn.ReLU(),
                         torch.nn.MaxPool2d(kernel_size=2)
                     )
        self.conv2 = torch.nn.Sequential(
                         torch.nn.Conv2d(in_channels=, out_channels=16, kernel_size=3, padding=1),
                         torch.nn.ReLU(),
                         torch.nn.MaxPool2d(kernel_size=2)
                     )
        self.fc_1 = torch.nn.Sequential(
                        torch.nn.Linear(in_features=, out_features=),
                        torch.nn.ReLU()
                    )
        self.fc_2 = torch.nn.Sequential(
                        torch.nn.Linear(in_features=, out_features=),
                        torch.nn.ReLU()
                    )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fc_1(x)
        x = self.fc_2(x)
        return x


########## Main function ##########
if __name__ == "__main__":
    # Print numpy ndarray without truncation
    numpy.set_printoptions(threshold=sys.maxsize)
    
    # Read PK files
    print("Load data from pickles.")
    train_x_ndarray, train_y_ndarray, test_x_ndarray, test_y_ndarray = readPK()
    
    train_train['X'], train_valid['X'], train_train['Y'], train_valid['Y'] = train_test_split(train['X'], train['Y'], test_size=CONST.vt_ratio())
    
    
    # Convert ndarray into PyTorch tensor
    train_x_tensor, train_y_tensor = torch.Tensor(train_x_ndarray), torch.Tensor(train_y_ndarray)
    test_x_tensor, test_y_tensor = torch.Tensor(test_x_ndarray), torch.Tensor(test_y_ndarray)
    
    # Create PyTorch DataLoader
    torch_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_y_tensor)
    torch_loader  = torch.utils.data.DataLoader(torch_dataset, batch_size=CONST.batch_size(), num_workers=0)
    
    
    
    
    
    
    
########## Other Notes ##########
# y_part label : 0 == Carambula, 1 == Lychee, 2 == Pear
'''
for i in range(5):
    print(index := random.randint(0, 498))
    printImg(True, test_x_part[index, :, :], test_y_part[index])
'''