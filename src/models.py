"""
The module contains the DL models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FNN(nn.Module):
    """FNN model."""
    def __init__(self, batchSize, device):
        super(FNN, self).__init__()

        self.input_size = 6     # 6 input channels
        self.num_outputs = 2    # controls for two motors
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


class SeRNN_FWXX(nn.Module):
    """Recurrent neural network"""
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

        # rnn output x (with batch_first = True) has 
        # dim (batch_size,seq_len,num_directions * hidden_size)
        x = torch.cat((x1, x2, x3, x4, x5, x6), dim=2)

        # get the output from last time sequence -> many-to-one type
        x = F.relu(self.fc1(x[:, -1, :]))
        x = self.fc3(x)
        x = self.fc2(x)

        return x

