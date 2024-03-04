import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from collections import OrderedDict
import matplotlib.pyplot as plt


class Encoder(nn.Module):
    def __init__(self, dim1, dim2, dim3,q):
        super(Encoder, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.dim3 = dim3
        self.conv = nn.Conv2d(2,2,kernel_size=3,stride=1,padding=1,bias=True)
        self.norm = nn.BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True)
        self.conv2 = nn.Conv2d(2,2,kernel_size=3,stride=1,padding=1,bias=True)
        self.norm2 = nn.BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True)
        self.fc = nn.Linear(int(self.dim1*self.dim2*self.dim3),int(self.dim1*self.dim2*self.dim3/q))
        self.activator = nn.LeakyReLU(negative_slope=0.3)
    def forward(self, x):

        out = self.activator(self.norm(self.conv(x)))

        #out = self.activator(self.norm2(self.conv2(x)))

        out = out.contiguous().view(-1,int(self.dim1*self.dim2*self.dim3))

        out = self.fc(out)

        return out
    
class Decoder(nn.Module):
    def __init__(self, dim1, dim2, dim3,q):
        super(Decoder, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.dim3 = dim3
        self.refinenet1 = RefineBlock()
        self.refinenet2 = RefineBlock()
        self.conv = nn.Conv2d(2,2,kernel_size=3,stride=1,padding=1,bias=True)
        self.norm = nn.BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True)
        self.fc = nn.Linear(int(self.dim1*self.dim2*self.dim3/q),int(self.dim1*self.dim2*self.dim3))
        self.activator = nn.Sigmoid()
    def forward(self, x):

        out = self.fc(x)

        out = out.contiguous().view(-1,self.dim1,self.dim2,self.dim3)
        
        out = self.refinenet1(out)

        out = self.refinenet2(out)

        #out = self.activator(self.norm(self.conv(out)))
        out = self.norm(self.conv(out))

        return out

class AutoEncoder(nn.Module):
    def __init__(self,dim1,dim2,dim3,q):
        super(AutoEncoder, self).__init__()
        self.Encoder = Encoder(dim1,dim2,dim3,q)
        self.Decoder = Decoder(dim1,dim2,dim3,q)
    def forward(self, x):
        
        out = self.Encoder(x)
        out = self.Decoder(out)
        return out
class RefineBlock(nn.Module):
    def __init__(self):
        super(RefineBlock, self).__init__()
        self.norm1 = nn.BatchNorm2d(8,eps=1e-05, momentum=0.1, affine=True)
        self.norm2 = nn.BatchNorm2d(16,eps=1e-05, momentum=0.1, affine=True)
        self.norm3 = nn.BatchNorm2d(2,eps=1e-05, momentum=0.1, affine=True)
        self.activator = nn.LeakyReLU(negative_slope=0.3)  
        self.conv1 = nn.Conv2d(2,8,kernel_size=7,stride=1,padding=3)
        self.conv2 = nn.Conv2d(8,16,kernel_size=5,stride=1,padding=2)
        self.conv3 = nn.Conv2d(16,2,kernel_size=3,stride=1,padding=1)

    def forward(self,Input):

        shortcut = Input

        out = self.activator(self.norm1(self.conv1(Input)))
        out = self.activator(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))

        out = out + shortcut

        out = self.activator(out)

        return out


def NMSE_cuda(x, x_hat):
    x = x.contiguous().view(len(x), -1)
    x_hat = x_hat.contiguous().view(len(x_hat), -1)
    power = torch.sum(abs(x) ** 2, dim=1)
    mse = torch.sum(abs(x - x_hat) ** 2, dim=1) / power

    return mse


class NMSELoss(nn.Module):
    def __init__(self, reduction='sum'):
        super(NMSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, x, x_hat):
        nmse = NMSE_cuda(x, x_hat)
        # cr_loss = C_R_cuda(x,x_hat)
        if self.reduction == 'mean':
            nmse = torch.mean(nmse)
            # cr_loss = -torch.mean(cr_loss)
        else:
            nmse = torch.sum(nmse)
            # cr_loss = -torch.sum(cr_loss)
        return nmse  # , cr_loss


class DatasetFolder(Dataset):

    def __init__(self, matData):
        self.matdata = matData

    def __len__(self):
        return self.matdata.shape[0]

    def __getitem__(self, index):
        return self.matdata[index]
    
