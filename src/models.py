# author: Daulet Baimukashev
# Created on: 6.05.20

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import copy

class RNN(nn.Module):
    def __init__(self, batchSize, device):
        super(RNN, self).__init__()

        self.input_size = 6    # 6 input channels
        self.num_outputs = 2   # controls for two motors
        self.batch = batchSize
        self.device = device

        # model
        self.num_layers = 3
        self.hidden_size = 64

        self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first = True)
        self.fc1 = nn.Linear(self.hidden_size, 128)

        self.fc3 = nn.Linear(128, 64)

        self.fc2 = nn.Linear(64, self.num_outputs)

        #self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        self.rnn.flatten_parameters()


        x, _ = self.rnn(x, (h0, c0))

        x = F.relu(self.fc1(x[:, -1, :]))


        x = F.relu(self.fc3(x))


        x = self.fc2(x)

        return x



class FNN(nn.Module):
    def __init__(self, batchSize, device):
        super(FNN, self).__init__()

        self.input_size = 6             # 6 input channels
        self.num_outputs = 2            # controls for two motors
        self.batch = batchSize

        self.fc0 = nn.Linear(6, 128)
        self.fc1 = nn.Linear(128, 128)
        self.fc1a = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, self.num_outputs)

        #self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):


        x = x.reshape(x.shape[0], -1)

        x = F.relu(self.fc0(x))
        #x = self.dropout(x)
        x = F.relu(self.fc1(x))

        x = F.relu(self.fc1a(x))
        #x = self.dropout(x)
        x = self.fc2(x)

        return x



class FcRNN(nn.Module):
    def __init__(self, batchSize, device):
        super(FcRNN, self).__init__()

        self.input_size = 6             # 6 input channels
        self.num_outputs = 2            # controls for two motors
        self.batch = batchSize
        self.device = device

        # model
        self.num_layers = 2
        self.hidden_size = 32

        self.fc0 = nn.Linear(6, 32)
        self.fc01 = nn.Linear(32, 32)

        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, self.num_outputs)

        self.dropout = nn.Dropout(p=0.3)

        self.rnn = nn.LSTM(5, self.hidden_size, self.num_layers, batch_first = True)


    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        self.rnn.flatten_parameters()

        x1 = F.relu(self.fc01(F.relu(self.fc0(x[:,0,:]))))
        x2 = F.relu(self.fc01(F.relu(self.fc0(x[:,1,:]))))
        x3 = F.relu(self.fc01(F.relu(self.fc0(x[:,2,:]))))
        x4 = F.relu(self.fc01(F.relu(self.fc0(x[:,3,:]))))
        x5 = F.relu(self.fc01(F.relu(self.fc0(x[:,4,:]))))

        x1 = torch.unsqueeze(x1,2)
        x2 = torch.unsqueeze(x2,2)
        x3 = torch.unsqueeze(x3,2)
        x4 = torch.unsqueeze(x4,2)
        x5 = torch.unsqueeze(x5,2)

        x = torch.cat((x1, x2, x3, x4, x5), dim=2)

        x, _ = self.rnn(x, (h0, c0))

        x = F.relu(self.fc1(x[:, -1, :]))

        x = self.fc2(x)

        return x




class SeRNN(nn.Module):
    def __init__(self, batchSize, device):
        super(SeRNN, self).__init__()

        self.input_size = 1    # 6 input channels
        self.num_outputs = 2   # controls for two motors
        self.batch = batchSize
        self.device = device

        # model
        self.num_layers = 1
        self.hidden_size = 32

        self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first = True)

        self.fc1 = nn.Linear(self.hidden_size*6, 64)

        self.fc3 = nn.Linear(64, 64)

        self.fc2 = nn.Linear(64, self.num_outputs)

        #self.dropout = nn.Dropout(p=0.2)

        #iself.bn = nn.BatchNorm1d(128)
        #self.bn2 = nn.BatchNorm1d(64)

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        self.rnn.flatten_parameters()


        x1, _ = self.rnn(torch.unsqueeze(x[:,:,0],2), (h0, c0))
        x2, _ = self.rnn(torch.unsqueeze(x[:,:,1],2), (h0, c0))
        x3, _ = self.rnn(torch.unsqueeze(x[:,:,2],2), (h0, c0))
        x4, _ = self.rnn(torch.unsqueeze(x[:,:,3],2), (h0, c0))
        x5, _ = self.rnn(torch.unsqueeze(x[:,:,4],2), (h0, c0))
        x6, _ = self.rnn(torch.unsqueeze(x[:,:,5],2), (h0, c0))


        # rnn output x (with batch_first = True) has dim (batch_size,seq_len,num_directions * hidden_size)
        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=2)

        # get the output from last time sequence -> many-to-one type
        x = F.relu(self.fc1(x[:, -1, :]))

        #x = self.dropout(x)
        #x = self.bn(x)

        x = F.relu(self.fc3(x))

        #x = self.bn2(x)
        #x = F.relu(self.fc3(x))
        #x = self.dropout(x)

        x = self.fc2(x)

        # x.shape  torch.Size([32, 2])   ... (batch, num_outputs)

        return x




class DeRNN(nn.Module):
    def __init__(self, batchSize, device):
        super(DeRNN, self).__init__()

        self.input_size = 1    # 6 input channels
        self.num_outputs = 2   # controls for two motors
        self.batch = batchSize
        self.device = device

        # model
        self.num_layers = 1
        self.hidden_size = 16

        self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first = True)
        self.rnn2 = nn.LSTM(1, 128, 2, batch_first = True)

        self.fc1 = nn.Linear(128, 128)

        self.fc3 = nn.Linear(128, 128)

        self.fc2 = nn.Linear(128, self.num_outputs)

        #self.dropout = nn.Dropout(p=0.15)

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        h1 = torch.zeros(2, x.size(0), 128).to(self.device)
        c1 = torch.zeros(2, x.size(0), 128).to(self.device)

        self.rnn.flatten_parameters()
        self.rnn2.flatten_parameters()

        x1, _ = self.rnn(torch.unsqueeze(x[:,:,0],2), (h0, c0))
        x2, _ = self.rnn(torch.unsqueeze(x[:,:,1],2), (h0, c0))
        x3, _ = self.rnn(torch.unsqueeze(x[:,:,2],2), (h0, c0))
        x4, _ = self.rnn(torch.unsqueeze(x[:,:,3],2), (h0, c0))
        x5, _ = self.rnn(torch.unsqueeze(x[:,:,4],2), (h0, c0))
        x6, _ = self.rnn(torch.unsqueeze(x[:,:,5],2), (h0, c0))


        # rnn output x (with batch_first = True) has dim (batch_size,seq_len,num_directions * hidden_size)
        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=2)

        # get the output from last time sequence -> many-to-one type
        x, _ = self.rnn2(torch.unsqueeze(x[:, -1, :],2), (h1, c1))

        x = F.relu(self.fc1(x[:, -1, :]))

        #x = self.dropout(x)

        x = F.relu(self.fc3(x))

        #x = self.dropout(x)

        x = self.fc2(x)

        # x.shape  torch.Size([32, 2])   ... (batch, num_outputs)

        return x




class ReRNN(nn.Module):
    def __init__(self, batchSize, device):
        super(ReRNN, self).__init__()

        self.input_size = 1    # 6 input channels
        self.num_outputs = 2   # controls for two motors
        self.batch = batchSize
        self.device = device

        # model
        self.num_layers = 1
        self.hidden_size = 32

        self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first = True)

        self.fc1 = nn.Linear(self.hidden_size*6, 128)

        self.fc3 = nn.Linear(128, 64)

        self.fc2 = nn.Linear(128, self.num_outputs)

        self.fc4 = nn.Linear(6, 64)

        self.fc5 = nn.Linear(64, 64)

        self.fc6 = nn.Linear(128,128)

        #self.dropout = nn.Dropout(p=0.15)

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        self.rnn.flatten_parameters()

        l = x[:,-1,:]

        x1, _ = self.rnn(torch.unsqueeze(x[:,:,0],2), (h0, c0))
        x2, _ = self.rnn(torch.unsqueeze(x[:,:,1],2), (h0, c0))
        x3, _ = self.rnn(torch.unsqueeze(x[:,:,2],2), (h0, c0))
        x4, _ = self.rnn(torch.unsqueeze(x[:,:,3],2), (h0, c0))
        x5, _ = self.rnn(torch.unsqueeze(x[:,:,4],2), (h0, c0))
        x6, _ = self.rnn(torch.unsqueeze(x[:,:,5],2), (h0, c0))

        # rnn output x (with batch_first = True) has dim (batch_size,seq_len,num_directions * hidden_size)
        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=2)

        # get the output from last time sequence -> many-to-one type
        x = F.relu(self.fc1(x[:, -1, :]))

        #x = self.dropout(x)

        x = F.relu(self.fc3(x))

        x = F.relu(self.fc5(x))

        l = F.relu(self.fc4(l))

        l = F.relu(self.fc5(l))

        #x = self.dropout(x)

        #print('x --', x.shape)
        #print('l --', l.shape)

        x = torch.cat((x, l), dim=1)

        #print('xl --', x.shape)

        x = self.fc6(x)

        x = self.fc2(x)

        # x.shape  torch.Size([32, 2])   ... (batch, num_outputs)

        return x


class LeRNN(nn.Module):
    def __init__(self, batchSize, device):
        super(LeRNN, self).__init__()

        self.input_size = 1    # 6 input channels
        self.num_outputs = 2   # controls for two motors
        self.batch = batchSize
        self.device = device

        # model
        self.num_layers = 1
        self.hidden_size = 32

        self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first = True)

        self.fc1 = nn.Linear(32*6, 128)

        self.fc3 = nn.Linear(128,64)

        self.fc2 = nn.Linear(64, self.num_outputs)

        self.fc4 = nn.Linear(32,32)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        self.rnn.flatten_parameters()

        x1, _ = self.rnn(torch.unsqueeze(x[:,:,0],2), (h0, c0))
        x2, _ = self.rnn(torch.unsqueeze(x[:,:,1],2), (h0, c0))
        x3, _ = self.rnn(torch.unsqueeze(x[:,:,2],2), (h0, c0))
        x4, _ = self.rnn(torch.unsqueeze(x[:,:,3],2), (h0, c0))
        x5, _ = self.rnn(torch.unsqueeze(x[:,:,4],2), (h0, c0))
        x6, _ = self.rnn(torch.unsqueeze(x[:,:,5],2), (h0, c0))

        x1 = F.relu(self.fc4(x1))
        x2 = F.relu(self.fc4(x2))
        x3 = F.relu(self.fc4(x3))
        x4 = F.relu(self.fc4(x4))
        x5 = F.relu(self.fc4(x5))
        x6 = F.relu(self.fc4(x6))

        # rnn output x (with batch_first = True) has dim (batch_size,seq_len,num_directions * hidden_size)
        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=2)

        # get the output from last time sequence -> many-to-one type
        x = F.relu(self.fc1(x[:, -1, :]))

        x = self.dropout(x)

        x = F.relu(self.fc3(x))

        #x = F.relu(self.fc3(x))

        x = self.dropout(x)

        x = self.fc2(x)

        # x.shape  torch.Size([32, 2])   ... (batch, num_outputs)

        return x



class CRNN(nn.Module):
    def __init__(self, batchSize, device):
        super(CRNN, self).__init__()

        self.num_layers = 1
        self.hidden_size = 32

        self.input_size = 6             # 6 input channels

        self.batch = batchSize
        self.num_outputs = 2 # controls for two motors
        self.device = device

        self.rnn = nn.LSTM(32, self.hidden_size, self.num_layers, batch_first = True)

        #self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first = True)
        self.fc1 = nn.Linear(self.hidden_size, 64)

        self.fc3 = nn.Linear(64, 32)

        self.fc2 = nn.Linear(32, self.num_outputs)

        self.conv1 = nn.Conv1d(self.input_size, 32, 3, 1) #
        #self.conv2 = nn.Conv1d(16, 32, 3, 1) #

    def forward(self, x):

        xc = copy.copy(x)

        xc = xc.permute(0,2,1)

        xc = F.relu(self.conv1(xc))

        xc = xc.permute(0,2,1)

        x = xc

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)


        x, _ = self.rnn(x, (h0, c0))
        # rnn output x (with batch_first = True) has dim (batch_size,seq_len,num_directions * hidden_size)

        # get the output from last time sequence -> many-to-one type
        x = F.relu(self.fc1(x[:, -1, :]))

        x = F.relu(self.fc3(x))

        x = self.fc2(x)
        # x.shape  torch.Size([32, 2])   ... (batch, num_outputs)


        return x




class CNN(nn.Module):
    def __init__(self, batchSize, device):
        super(CNN, self).__init__()

        self.input_size = 6             # 6 input channels

        self.batch = batchSize

        self.num_outputs = 2 # controls for two motors
        self.device = device


        self.baseConv1 = nn.Conv2d(1, 16, 2, 1)

        self.baseConv2 = nn.Conv2d(16, 32, 2, 1)

        #self.baseConv1 = nn.Conv1d(3, 32, 5, 2)

        self.Fc1 = nn.Linear(672, 128)
        self.Fc2 = nn.Linear(128, 128)
        self.Fc3 = nn.Linear(128, 2)

        self.pool = nn.MaxPool2d(2, 1)

    def forward(self, x):

        #print('cnn ', x.shape)
        x = x.permute(0,2,1)
        x = torch.unsqueeze(x,1)
        #print('cnn.. ', x.shape)

        x = F.relu(self.baseConv1(x))
        x = self.pool(F.relu(self.baseConv2(x)))

        # flatten the Base
        x = x.view(x.size(0), -1)

        #print('v', x.shape)

        x = F.relu(self.Fc1(x))
        x = F.relu(self.Fc2(x))

        x = self.Fc3(x)




class SeRNN_FWXX(nn.Module):
    def __init__(self, batchSize, device):
        super(SeRNN_FWXX, self).__init__()

        self.input_size = 1    # 6 input channels
        self.num_outputs = 2   # controls for two motors
        self.batch = batchSize
        self.device = device

        # model
        self.num_layers = 1
        self.hidden_size = 32

        self.rnn1 = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first = True)
        self.rnn2 = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first = True)
        self.rnn3 = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first = True)
        self.rnn4 = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first = True)
        self.rnn5 = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first = True)
        self.rnn6 = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first = True)

        self.fc1 = nn.Linear(self.hidden_size*6, 64)

        self.fc3 = nn.Linear(64, 64)

        self.fc2 = nn.Linear(64, self.num_outputs)

        #self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        self.rnn1.flatten_parameters()
        self.rnn2.flatten_parameters()
        self.rnn3.flatten_parameters()
        self.rnn4.flatten_parameters()
        self.rnn5.flatten_parameters()
        self.rnn6.flatten_parameters()

        x1, _ = self.rnn1(torch.unsqueeze(x[:,:,0],2), (h0, c0))
        x2, _ = self.rnn2(torch.unsqueeze(x[:,:,1],2), (h0, c0))
        x3, _ = self.rnn3(torch.unsqueeze(x[:,:,2],2), (h0, c0))
        x4, _ = self.rnn4(torch.unsqueeze(x[:,:,3],2), (h0, c0))
        x5, _ = self.rnn5(torch.unsqueeze(x[:,:,4],2), (h0, c0))
        x6, _ = self.rnn6(torch.unsqueeze(x[:,:,5],2), (h0, c0))

        # rnn output x (with batch_first = True) has dim (batch_size,seq_len,num_directions * hidden_size)
        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=2)

        # get the output from last time sequence -> many-to-one type
        x = F.relu(self.fc1(x[:, -1, :]))

        x = self.fc3(x)

        x = self.fc2(x)

        return x



        return x
