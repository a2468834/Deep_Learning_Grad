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
    '''
    CAE = convAutoEncdr()
    if if_use_gpu:
        CAE = CAE.cuda()
    '''
    
    transform = transforms.ToTensor()
    
    train_data = datasets.CIFAR10(root='data', train=True, transform=transform)
    test_data  = datasets.CIFAR10(root='data', train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, num_workers=0)
    test_loader  = torch.utils.data.DataLoader(test_data, batch_size=32, num_workers=0)
    
    for data in train_loader:
        print(len(data[0]))
        exit()
    
    exit()
    
    CAE = ConvAutoencoder()
    print(CAE)
    
    Loss = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(CAE.parameters(), lr=0.001)
    
    device = have_CUDA()
    print(device)
    CAE.to(device)
    
    epoch_num = 100
    for epoch in range(epoch_num):
        cur_loss = 0.0
        
        #Training
        for data in train_loader:
            images, _ = data
            images = images.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = CAE(outputs, images)
            loss.backward()
            optimizer.step()
            cur_loss += loss.item() * images.size(0)
              
        train_loss = train_loss/len(train_loader)
        print('Epoch: {} \tTraining Loss: {:.4f}'.format(epoch, cur_loss))
    
########## Other Notes ##########
# data_tensor  : (# of images) x (image width) x (image height) x (3-channel)
# label_tensor : (# of images) x (label)
# 3-channel : boundary T/F, normal T/F, defect T/F
# label is nine-class