import os
import h5py
import numpy as np
import math
import scipy.io as sio
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
from ModelDesign import AutoEncoder, DatasetFolder, NMSE_cuda, NMSELoss
import time
#from torchsummary import summary

# Parameters for training
gpu_list = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
device = "cuda" if torch.cuda.is_available() else "cpu"

channel_type = 'sv'  #sv or cost2100

#训练参数，实部虚部两通道，天线数32，子载波32（截断）
img_height = 32
img_width = 32
img_channels = 2 
img_total = img_height*img_width*img_channels

#训练参数
epochs=100
batchsize=200
iter=500
numworkers=0
learning_rate = 1e-3

#实例化模型
model=AutoEncoder(img_channels,img_height,img_width,4).to(device)

#损失函数 改用MSE为损失函数
criterion = NMSELoss(reduction='mean')
criterion_test = NMSELoss(reduction='mean')

#优化器ADAM
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#数据集路径
if channel_type == 'sv':
     train = './data/H_train_t.mat'
     val = './data/H_val_t.mat'
     test = './data/H_test_t.mat'
     mat = h5py.File(train)
     x_train = mat['H_train_t']  #这里应该填matlab里的名字
     x_train = x_train.astype('float32')  # 训练变量类型转换 
     x_train = np.transpose(x_train,[3,0,2,1]) #维度变换 NCHW
     mat = h5py.File(val)
     x_val = mat['H_val_t']  #这里应该填matlab里的名字
     x_val = x_val.astype('float32')  # 训练变量类型转换 
     x_val = np.transpose(x_val,[3,0,2,1]) #维度变换 NCHW
     mat = h5py.File(test)
     x_test = mat['H_test_t']  #这里应该填matlab里的名字
     x_test = x_test.astype('float32')  # 训练变量类型转换 
     x_test = np.transpose(x_test,[3,0,2,1]) #维度变换 NCHW
else:
     train = './data/H_train.mat'
     val = './data/H_val.mat'
     test = './data/H_test.mat'
     mat = sio.loadmat(train) 
     x_train = mat['HT']
     x_train = x_train.astype('float32')  # 训练变量类型转换 
     x_train = np.reshape(x_train,(len(x_train), img_channels, img_height, img_width))
     mat = sio.loadmat(val) 
     x_val = mat['HT']
     x_val = x_val.astype('float32')  # 训练变量类型转换 
     x_val = np.reshape(x_val,(len(x_val), img_channels, img_height, img_width))
     mat = sio.loadmat(test) 
     x_test = mat['HT']
     x_test = x_test.astype('float32')  # 训练变量类型转换 
     x_test = np.reshape(x_test,(len(x_test), img_channels, img_height, img_width))



'''
#测试一下，画一个CSI出来，看读到的是否正确
x_real=x_train[1,0,:,:]
x_imag=x_train[1,1,:,:]
x_abs=np.sqrt(np.power(x_real,2)+np.power(x_imag,2))
plt.imshow(x_imag)
plt.show()
'''

#实例化Dataset
Dataset_train=DatasetFolder(x_train)
Dataset_val=DatasetFolder(x_val)
Dataset_test=DatasetFolder(x_test)

#通过DataLoader得到划分的数据集
Dataloader_train=torch.utils.data.DataLoader(
    Dataset_train, batch_size=batchsize, shuffle=True, num_workers=numworkers, pin_memory=True, drop_last=True)
Dataloader_val=torch.utils.data.DataLoader(
    Dataset_val, batch_size=batchsize, shuffle=True, num_workers=numworkers, pin_memory=True, drop_last=True)
Dataloader_test=torch.utils.data.DataLoader(
    Dataset_test, batch_size=batchsize, shuffle=True, num_workers=numworkers, pin_memory=True, drop_last=True)


#模型训练与测试
best_loss = 200
for epoch in range(epochs):

    t1 = time.time()
    #切换到train模式
    model.train()

    #遍历dataloader
    for i, input in enumerate(Dataloader_train):
        input = input.cuda().float()
        output = model(input)
        loss = criterion(input,output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 200 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                    'Loss {loss:.4f}\t' 
                    'dB_Loss {dB_loss:.4f}\t'.format(
                epoch, i, len(Dataloader_train), loss=loss.item(), dB_loss=10*math.log10(loss.item())))

    #切换到test模式
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for i, input in enumerate(Dataloader_test):
            input = input.cuda().float()
            output = model(input)
            loss = criterion_test(input, output)
            total_loss += loss.item()
        average_loss = total_loss / len(Dataloader_test)
        dB_loss = 10 * math.log10(average_loss)
        print('Loss %.4f' % average_loss, 'dB Loss %.4f' % dB_loss)

        if average_loss < best_loss:
                torch.save({'state_dict': model.Encoder.state_dict(
                ), }, 'models/Encoder_'+'CsiNet.pth.tar')
                # Decoder Saving
                torch.save({'state_dict': model.Decoder.state_dict(
                ), }, 'models/Decoder_'+'CsiNet.pth.tar')
                print("Model saved")
                best_loss = average_loss

    t2 = time.time()
    print('time: ',t2-t1)

a=1