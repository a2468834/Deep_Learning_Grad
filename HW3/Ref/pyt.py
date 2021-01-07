import torch
from   torch import optim
import torch.nn
import torch.nn.functional
from   torch.utils.data import DataLoader
from   torchvision import datasets
import sys
import numpy
from   scipy.special import softmax as scipy_softmax
import cv2
from   HW3_109062631_Module import *


epoch_num = lambda : 100


class testNet(torch.nn.Module):
    def __init__(self):
        super(testNet, self).__init__()
        self.conv = torch.nn.Conv2d(1, 8, kernel_size=(4, 4), stride=1)
        self.maxp = torch.nn.MaxPool2d(kernel_size=(4, 4))
        self.lin0 = torch.nn.Linear(in_features=1024, out_features=392, bias=False)
        self.lin1 = torch.nn.Linear(in_features=392, out_features=98, bias=False)
        self.lin2 = torch.nn.Linear(in_features=98, out_features=14, bias=False)
        self.lin3 = torch.nn.Linear(in_features=14, out_features=3, bias=False)
        self.relu = torch.nn.ReLU()
        self.soft = torch.nn.Softmax(dim=0)
        
    
    def forward(self, x):
        #x = self.conv(x)
        #x = self.relu(x)
        #x = self.maxp(x)
        
        x = torch.flatten(x)
        
        x = self.lin0(x)
        x = self.relu(x)
        
        x = self.lin1(x)
        x = self.relu(x)
        
        x = self.lin2(x)
        x = self.relu(x)
        
        x = self.lin3(x)
        x = self.soft(x)
        return x


def printLossValue(epoch, train_loss, valid_loss):
    print('Epoch: {:>2}  |  Training Loss: {:>7.4f}  |  Validation Loss: {:>7.4f}'.format(int(epoch), train_loss, valid_loss))


if __name__ == '__main__':
    start_time = time.time()
    
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
    
    # Add a new dimension to x part
    train_ndarray['X'] = numpy.expand_dims(train_ndarray['X'], axis=1)
    train_ndarray_P70['X'] = numpy.expand_dims(train_ndarray_P70['X'], axis=1)
    train_ndarray_P30['X'] = numpy.expand_dims(train_ndarray_P30['X'], axis=1)
    test_ndarray['X'] = numpy.expand_dims(test_ndarray['X'], axis=1)
    
    # Prepare PyTorch DataLoader
    train_tensor = {}
    train_tensor['X'] = torch.Tensor(train_ndarray_P70['X'])
    train_tensor['Y'] = torch.Tensor(train_ndarray_P70['Y'])
    
    valid_tensor = {}
    valid_tensor['X'] = torch.Tensor(train_ndarray_P30['X'])
    valid_tensor['Y'] = torch.Tensor(train_ndarray_P30['Y'])
    
    train_dataset = torch.utils.data.TensorDataset(train_tensor['X'], train_tensor['Y'])
    train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=0)
    
    valid_dataset = torch.utils.data.TensorDataset(valid_tensor['X'], valid_tensor['Y'])
    valid_loader  = torch.utils.data.DataLoader(valid_dataset, batch_size=1, num_workers=0)
    
    print("Start training a model.\n")
    tN = testNet()
    optimizer = torch.optim.ASGD(tN.parameters(), lr=CONST.l_rate())
    loss_func = torch.nn.CrossEntropyLoss()
    
    # Print epoch_0
    tN.eval()
    train_loss, valid_loss = 0.0, 0.0
    for data, label in train_loader:
        predict = tN(data)
        predict = torch.unsqueeze(predict, 0)
        label = torch.argmax(label, dim=1)
        
        # Calculate loss
        temp_loss = loss_func(predict, label)
        train_loss += temp_loss.item() * data.size(0)
    for data, label in valid_loader:
        predict = tN(data)
        predict = torch.unsqueeze(predict, 0)
        label = torch.argmax(label, dim=1)
        
        temp_loss = loss_func(predict, label)
        valid_loss += temp_loss.item() * data.size(0)
    
    printLossValue(0, train_loss, valid_loss)
    del train_loss
    del valid_loss
    
    # Train the model
    tN.train()
    for epoch in range(epoch_num()):
        train_loss, valid_loss = 0.0, 0.0
        
        for data, label in train_loader:
            predict = tN(data)
            predict = torch.unsqueeze(predict, 0)
            label = torch.argmax(label, dim=1)
            
            # Calculate loss
            temp_loss = loss_func(predict, label)
            train_loss += temp_loss.item() * data.size(0)
            
            # Backward propagation
            optimizer.zero_grad() # Set all the gradients to zero before backward propragation
            temp_loss.backward()
            optimizer.step() # Performs a single optimization step.
        
        # Calculate validation loss
        for data, label in valid_loader:
            predict = tN(data)
            predict = torch.unsqueeze(predict, 0)
            label = torch.argmax(label, dim=1)
            
            temp_loss = loss_func(predict, label)
            valid_loss += temp_loss.item() * data.size(0)
        
        printLossValue(epoch+1, train_loss, valid_loss)
    
    # Update model with whole data
    torch_dataset = torch.utils.data.TensorDataset(torch.Tensor(train_ndarray['X']), torch.Tensor(train_ndarray['Y']))
    torch_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=0)
    for data, label in torch_loader:
        predict = tN(data)
        predict = torch.unsqueeze(predict, 0)
        label = torch.argmax(label, dim=1)
        
        # Calculate loss and accuracy
        temp_loss = loss_func(predict, label)
        
        # Backward propagation
        optimizer.zero_grad() # Set all the gradients to zero before backward propragation
        temp_loss.backward()
        optimizer.step() # Performs a single optimization step
    
    
    print("\nEnd training.")
    print("\nPredict testing data set.\n")
    
    # Print total accuracy
    tN.eval()
    test_dataset = torch.utils.data.TensorDataset(torch.Tensor(test_ndarray['X']), torch.Tensor(test_ndarray['Y']))
    test_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=0)
    accuracy = 0
    
    for data, label in test_loader:
        predict  = tN(data)
        accuracy = (accuracy+1) if torch.argmax(predict) == torch.argmax(label) else (accuracy+0)
    
    print("[Result]")
    print("Accuracy on test data: {:.4f}%".format(accuracy / len(test_loader) * 100.0))
    print("Total execution time: {:.2f} seconds".format(time.time()-start_time))

####
'''
# Print numpy ndarray without truncation
numpy.set_printoptions(threshold=sys.maxsize)

aa_th = torch.randn(1, 1, 32, 32) # (N, C_in, H, W)

TN = testNet()
bb_th = TN(aa_th)

print(aa_th.size()) # torch.Size([1, 1, 32, 32])
print(bb_th.size()) # torch.Size([1, 8, 29, 29])
'''
