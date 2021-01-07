import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
from torch.cuda import device
from torch.utils.data import DataLoader


class SE(nn.Module):
    def __init__(self, n_features=32, reduction=2):
        super(SE, self).__init__()

        if n_features % reduction != 0:
            raise ValueError('n_features must be divisible by reduction (default = 4)')

        self.linear1 = nn.Linear(n_features, n_features // reduction, bias=True)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(n_features // reduction, n_features, bias=True)
        self.nonlin2 = nn.Sigmoid()

    def forward(self, x):

        y = F.avg_pool2d(x, kernel_size=x.size()[2:4])
        y = y.permute(0, 2, 3, 1)
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))
        y = y.permute(0, 3, 1, 2)
        y = x * y
        return y


class WTFNet(nn.Module):
    def __init__(self, activation=nn.ELU, channel=6, sq=False):
        super(WTFNet, self).__init__()

        self.sq = sq

        self.firstConv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(channel, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation(),
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0),
            nn.Dropout(p=0.15)
        )

        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation(),
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0),
            nn.Dropout(p=0.15)
        )
        
        if sq:
            self.seblock = SE(n_features=32)
        
        self.rnn = nn.LSTM(channel,32,2, dropout=0.2)
        
        #self.reduce_cnn = nn.Linear(in_features=1568, out_features=128)
        
        #cloud=480
        self.reduce_cnn = nn.Linear(in_features=480, out_features=128)
        
        self.classify = nn.Sequential(
            nn.Linear(in_features=128+32+32, out_features=4)
            #nn.Linear(in_features=128+32+32, out_features=2, bias=True)
            #nn.Linear(in_features=128, out_features=2, bias=True)
        )

        # in_features=480

    def forward(self, x):
        
        #print("input_dim: ", x.shape) 
        cnn_h = self.firstConv(x)
        
        # x.shape = (batch_size, 1, channel_size, seq_length)
        # rnn input shape: (seq_len, batch, input_size)
        #print(torch.squeeze(x,1).permute(2, 0, 1).shape)
        _, (rnn_h, _) = self.rnn(torch.squeeze(x,1).permute(2, 0, 1))
        #print("after_1stConv: ", cnn_h.shape) 
        cnn_h = self.depthwiseConv(cnn_h)
        #print("after_depConv: ", cnn_h.shape) 
        cnn_h = self.separableConv(cnn_h)
        #print("after_sepConv: ", cnn_h.shape) 
       
        if self.sq:
            # SE Block
            cnn_h = self.seblock(cnn_h)
            #print("after_se-block: ", x.shape) 
        
        cnn_h = cnn_h.view(-1, cnn_h.shape[1]*cnn_h.shape[3])
        #x = x.view(-1, self.classify[0].in_features)
        #print("after_view: ", cnn_h.shape) 
        cnn_h = self.reduce_cnn(cnn_h)
        #rst = self.reduce_cnn(cnn_h)
        # only get the second(last) layer output of rnn
        #print(cnn_h.shape, rnn_h[0].shape, rnn_h[1].shape)
        rst = self.classify(torch.cat([cnn_h, rnn_h[0], rnn_h[1]],axis=1))
        #rst = self.classify(cnn_h)
        #print("after_classify: ", x.shape) 
        return rst

class WTFNet_CNN(nn.Module):
    def __init__(self, activation=nn.ELU, channel=6, n_class=4, sq=False):
        super(WTFNet_CNN, self).__init__()

        self.sq = sq

        self.firstConv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

        self.depthwiseConv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(channel, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation(),
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0),
            nn.Dropout(p=0.15)
        )

        self.separableConv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            activation(),
            nn.AvgPool2d(kernel_size=(1, 2), stride=(1, 2), padding=0),
            nn.Dropout(p=0.15)
        )
        
        if sq:
            self.seblock = SE(n_features=32)
        
        #self.rnn = nn.LSTM(channel,32,2, dropout=0.2)
        
        #self.reduce_cnn = nn.Linear(in_features=1568, out_features=128)
        #self.reduce_cnn = nn.Linear(in_features=480, out_features=4)
        #self.reduce_cnn = nn.Linear(in_features=480, out_features=4)
        
        self.classify = nn.Sequential(
            #nn.Linear(in_features=128+32+32, out_features=2, bias=True)
            #nn.Linear(in_features=128, out_features=2, bias=True)
            # cloud = 480
            nn.Linear(in_features=480, out_features=192),
            # Wafer=1568
            #nn.Linear(in_features=1568, out_features=192),
            #nn.Linear(in_features=736, out_features=192),
            activation(),
            nn.Linear(in_features=192, out_features=n_class, bias=True)
        )

        # in_features=480

    def forward(self, x):
        
        #print("input_dim: ", x.shape) 
        cnn_h = self.firstConv(x)
        
        # x.shape = (batch_size, 1, channel_size, seq_length)
        # rnn input shape: (seq_len, batch, input_size)
        #print(torch.squeeze(x,1).permute(2, 0, 1).shape)
        ##_, (rnn_h, _) = self.rnn(torch.squeeze(x,1).permute(2, 0, 1))
        #print("after_1stConv: ", cnn_h.shape) 
        cnn_h = self.depthwiseConv(cnn_h)
        #print("after_depConv: ", cnn_h.shape) 
        cnn_h = self.separableConv(cnn_h)
        #print("after_sepConv: ", cnn_h.shape) 
       
        if self.sq:
            # SE Block
            cnn_h = self.seblock(cnn_h)
            #print("after_se-block: ", x.shape) 
        
        cnn_h = cnn_h.view(-1, cnn_h.shape[1]*cnn_h.shape[3])
        #x = x.view(-1, self.classify[0].in_features)
        #print("after_view: ", cnn_h.shape) 
        #cnn_h = self.reduce_cnn(cnn_h)
        #rst = self.reduce_cnn(cnn_h)
        # only get the second(last) layer output of rnn
        #print(cnn_h.shape, rnn_h[0].shape, rnn_h[1].shape)
        #rst = self.classify(torch.cat([cnn_h, rnn_h[0], rnn_h[1]],axis=1))
        
        ###
        rst = self.classify(cnn_h)
        #rst = cnn_h

        #print("after_classify: ", x.shape) 
        return rst

