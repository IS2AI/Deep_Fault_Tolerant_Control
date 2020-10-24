# author: Daulet Baimukashev
# Created on: 6.05.20

import sys
import numpy as np
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt

from utils import read_data
from dataset import ControlDataset, ToTensor
from models import CNN, RNN, FNN, SeRNN, SeRNN_FWXX, DeRNN, ReRNN, LeRNN, FcRNN
from utils import train_model, test_model

import yaml

def main():

    config = yaml.safe_load(open("config.yml"))

    # training hyperparameters
    num_epochs = config['num_epochs']
    learning_rate = config['learning_rate']
    batchSize = config['batchSize']
    valid_period = config['valid_period']
    # data hyperparameters
    # Get the train/dev/test data
    # extract sample (input sequences with length=stride_len) for RNN from the givent full trajectory
    window_len = config['window_len']
    stride_len = config['stride_len']
    n_times = config['n_times']

    x_train, y_train, x_dev, y_dev, x_test, y_test = read_data(window_len, stride_len, n_times)
    print('Dataset:', x_train.shape, y_train.shape, x_dev.shape, y_dev.shape, x_test.shape, y_test.shape)

    # create dataset
    transformed_control_dataset_train = ControlDataset(x_train, y_train, transform=transforms.Compose([ToTensor()]))
    #transformed_control_dataset_dev = ControlDataset(x_dev, y_dev, transform=transforms.Compose([ToTensor()]))
    #transformed_control_dataset_test = ControlDataset(x_test, y_test, transform=transforms.Compose([ToTensor()]))
    #print('2: Length train - dev - test:', len(transformed_control_dataset_train), len(transformed_control_dataset_dev), len(transformed_control_dataset_test))

    # create batch
    data_train = DataLoader(transformed_control_dataset_train, batch_size = batchSize, shuffle=True, num_workers=1, drop_last=True)

    #data_dev = DataLoader(transformed_control_dataset_dev, batch_size= 1,shuffle=False, num_workers= 1,drop_last= True)
    #data_test = DataLoader(transformed_control_dataset_test, batch_size= 1,shuffle=False, num_workers= 1,drop_last= True)

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device:', device)

    # save model
    save_model_name = config['model_name']
    print('Save Model name: ', save_model_name)
    model_save_path = '../checkpoints/' + save_model_name #+ '.pt' ## not pth

    # test/ train
    mode = int(sys.argv[1])
    #def train(args, model, device, train_loader, optimizer, epoch):

    if mode == 0: # train mode

        # Model preparation
        model = SeRNN_FWXX(batchSize, device)

        #model = torch.jit.load(model_save_path)
        #model.load_state_dict(torch.load('../checkpoints/' + 'RNN_NF_light1_ep200' + '.pt'))

        print('Model: ', model)
        model.to(device)
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print('# of params: ', pytorch_total_params)

        # multiple GPUs
        if torch.cuda.device_count() > 1:
            print("Model uploaded. Number of GPUs: ", torch.cuda.device_count())
            #print("Model :", model)
            #model = nn.DataParallel(model)

        # set the training loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
        #optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum = 0.9)
        #optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        train_model(model, device, data_train, x_dev, y_dev, optimizer, criterion, num_epochs, model_save_path,  window_len, stride_len, valid_period)

    else: # test mode
        # upload saved model
        print('Saved Model evaluation with test set')
        model = torch.jit.load(model_save_path)
        test_model(model, device, x_test, y_test, 'True', window_len, stride_len)

if __name__ == '__main__':
    main()
