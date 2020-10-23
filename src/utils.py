# author: Daulet Baimukashev
# Created on: 6.05.20

import scipy.io as sio
import numpy as np

import sys
import numpy as np
import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn.model_selection import train_test_split

from noise_generate import add_noise
import yaml
import copy

def train_model(model, device, data_train, x_dev, y_dev, optimizer, criterion, num_epochs, model_save_path,  window_len, stride_len, valid_period):
    print('train the model')

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size= 100, gamma=0.5)

    for epoch in range(num_epochs):

        start = time.time()
        running_loss = 0.0
        for i, data in enumerate(data_train):

            # get the inputs; data is a list of [inputs, labels]
            inputs = data['features']
            inputs = inputs.float()
            inputs = inputs.to(device)
            #inputs = inputs.requires_grad_().to(device)

            labels = data['labels']
            labels = labels.type(torch.cuda.FloatTensor)
            labels = labels.squeeze(1).to(device)

            #zero the parameter gradients
            optimizer.zero_grad()

            # forward propagation
            output = model(inputs)
            # inputs.shape = torch.Size([32, 25, 6]) torch.float32
            # output.shape = torch.Size([32, 2]) torch.float32
            # labels.shape = torch.Size([32, 2]) torch.float32

            loss = criterion(output, labels)

            #l2_reg = 0.0
            #for W in model.parameters():
            #    l2_reg = l2_reg + W.norm(2)

            #l1_reg = 0.0
            #for W in model.parameters():
            #    l1_reg = l1_reg + torch.sum(torch.abs(W))

            #loss = loss + 2.0/output.size(0)*l2_reg * 0.01

            loss.backward()

            #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

            optimizer.step()

            # print loss
            running_loss += loss.item()

            if (i+1) % 500 == 0:    # print every 100 mini-batches
                end = time.time()
                print('Epoch: {}, Iter: {}, Loss: {}, Elapsed: {}'.format(epoch + 1, i + 1, running_loss / 500, end-start))

                with open(model_save_path + '_loss.txt', "a") as myfile:
                        myfile.write(str(running_loss / 500))
                        myfile.write("\n")
                running_loss = 0.0

        scheduler.step()

        if (epoch+1) % valid_period == 0:

            # save the model
            #torch.save(model.state_dict(), model_save_path + '.pth')
            model.eval()
            sample_input = torch.from_numpy(np.ones((1,window_len,6))).float().to(device)
            traced_script_module = torch.jit.trace(model, sample_input)
            traced_script_module.save(model_save_path + '_ep' + str(epoch+1) + '.pt')
            print('saving', model_save_path)
            model.train()
            # evaluate the model
            # test_model(model, device, x_dev, y_dev, 'False',window_len, stride_len)


def test_model(model, device, x_data, y_data, plot_enable, window_len, stride_len):

    config = yaml.safe_load(open("config.yml"))
    valid_size = config['valid_size']

    start = time.time()

    stride_test = 1 #stride_len
    window_test = window_len

    width = x_data.shape[1]

    plot_label = np.zeros((1,2))
    plot_output = np.zeros((1,2))

    sum_rmse = 0
    count_rmse = 0
    with torch.no_grad():
        for i in range(valid_size): # x_data.shape[0]

            count = 1
            index_end = 0
            plot_label = np.zeros((1,2))
            plot_output = np.zeros((1,2))

            ind = i

            while  index_end+stride_test <= width:

                index_start = count*stride_test
                index_end = count*stride_test + window_test

                inputs = x_data[ind, index_start:index_end, :]
                inputs = torch.from_numpy(inputs)
                inputs = inputs.float()
                inputs = inputs.to(device)
                inputs = inputs.unsqueeze(0)

                labels = y_data[ind, index_start:index_end, :]
                labels = torch.from_numpy(labels)
                labels = labels[-1,:]
                labels = labels.type(torch.cuda.FloatTensor).to(device)

                output = model(inputs)
                output = output.squeeze(0)
                # inputs.shape = torch.Size([1, 25, 6]) torch.float32
                # output.shape = torch.Size([2]) torch.float32
                # labels.shape = torch.Size([2]) torch.float32

                plot_label = np.vstack((plot_label, labels.cpu().detach().numpy()))
                plot_output = np.vstack((plot_output, output.cpu().detach().numpy()))

                count += 1

            error_rmse = np.sqrt(np.mean((plot_label-plot_output)**2, axis = 0))
            sum_rmse += error_rmse
            count_rmse += 1
            # print('plot data ', plot_output.shape, plot_label.shape)
            # print('mean_rmse: ', mean_rmse)

            if plot_enable == 'True':

                # plt.plot(plot_label[:,0])
                # plt.plot(plot_output[:,0])

                fig, axs = plt.subplots(2)

                axs[0].plot(plot_label[:,0], label = "OCP")
                axs[0].plot(plot_output[:,0], label = "RNN")

                axs[1].plot(plot_label[:,1],label = "OCP")
                axs[1].plot(plot_output[:,1], label = "RNN")

                # plot
                axs[0].set_title('RNN vs OCP plot (Motor 1)')
                axs[0].set_xlabel('Time sequence')
                axs[0].set_ylabel('Motor inputs (currents)')

                axs[1].set_title('RNN vs OCP plot (Motor 2)')
                axs[1].set_xlabel('Time sequence')
                axs[1].set_ylabel('Motor inputs (currents)')

                #axs[1].plot(x_data[ind, :, 5])
                #axs[2].plot(y_data[ind, :, 0])
                plt.show()

    mean_rmse = sum_rmse/count_rmse

    print('Number of samples tested: ', count_rmse, 'Number of inputs sequences: ', count, 'mean_rmse: ', mean_rmse)
    # mean and std dev
    end = time.time()
    print('Elapsed time for evaluation', end-start)

def read_data(window_len, stride_len, n_times):
    """
    Read the data and return normalized inputs.
    """
    config = yaml.safe_load(open("config.yml"))
    state_len = config['state_len']
    n_times_orig = config['n_times_orig']


    path = '../data/'
    file = "outputset1.mat"
    a = sio.loadmat(path + file)
    file = "outputset2.mat"
    b = sio.loadmat(path + file)
    file = "outputset3.mat"
    c = sio.loadmat(path + file)
    file = "outputset4.mat"
    d = sio.loadmat(path + file)
    file = "outputset6.mat"
    e = sio.loadmat(path + file)
    file = "outputset7.mat"
    f = sio.loadmat(path + file)
    file = "outputset8.mat"
    g = sio.loadmat(path + file)
    file = "outputset9.mat"
    h = sio.loadmat(path + file)
    file = "outputset10.mat"
    j = sio.loadmat(path + file)

    control=np.concatenate((a['controls'],b['controls'],c['controls'],d['controls'], e['controls'],f['controls'],g['controls'],h['controls']))
    state=np.concatenate((a['states'],b['states'],c['states'],d['states'],e['states'], f['states'],g['states'],h['states']))

    #state = state[:,0:state_len,:]
    #control = control[:,0:state_len,:]
    #print('Original state', state.shape)
    #print('Original control', control.shape)

    #state=np.concatenate((state, control), axis=2)

    #print('Added state', state.shape)
    print('>> Before fault:', state.shape  )

    #print('1 > ', control.shape)

    # Moving average
    #a1, b1, c1 = control.shape
    #for ind1 in range(a1):
        #for ind2 in range(c1):
        #    control[ind1,:,ind2] = np.convolve(control[ind1,:,ind2], np.ones((5,))/5, mode='same')

    #print('2 > ', control.shape)

    # add FAULT
    state, control = add_noise(state,control, n_times, n_times_orig)

    # normalize inputs
    #max_state_values = [3.14, 3.14, 10, 10, 350, 350,5,5]
    for ind in range(6):
        # option 1
        # normalize by the POSSIBLE min and max values

        #state[:, :, ind] = np.clip(state[:, :, ind], -max_state_values[ind], max_state_values[ind])/max_state_values[ind]

        #pass
        #option 2
        # normalize by the min and max values of the given dataset
        #state[:, :, ind] = (state[:, :, ind] - np.amin(state[:, :, ind])) /(-np.amin(state[:, :, ind]) + np.amax(state[:, :, ind]))
        # print(np.amin(state[:, :, ind]))
        # print(np.amax(state[:, :, ind]))


        #option 3 - Z Score
        # normalize
        #max_state_values = [3.14, 3.14, 10, 10, 350, 350]
        #state[:, :, ind] = np.clip(state[:, :, ind], -max_state_values[ind], max_state_values[ind])

        print(np.std(state[:, :, ind]))
        #std_temp = np.std(state[:, :, ind])
        #state[:, :, ind] = state[:, :, ind]/std_temp # mean is zero


    # add gaussian noise
    #for ind in range(6):
        #a1, b1, c1 = state.shape
        #print(a1)
        #noise_random = np.random.normal(0,0.01,(4000,b1))
        #print(state[:, :, ind].shape, noise_random.shape)
        #state[2000:6000, :, ind] = state[2000:6000, :, ind] + noise_random


    print('>> After fault:', state.shape  )

    train_ratio = config['train_set_ratio']
    dev_ratio = config['dev_set_ratio']
    test_ratio = config['test_set_ratio']

    # shuffle
    # divide into train/dev/test set
    x_train, x_test, y_train, y_test = train_test_split(state, control, test_size= 1 - train_ratio, shuffle = True, random_state = 1)
    x_dev, x_test, y_dev, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + dev_ratio), shuffle = False)

    # divide test into windows
    # x_train_sliced -> input sequence
    # x_train_sliced -> control input at the end of the input sequence

    x_train_sliced = extract_slices(x_train, window_len, stride_len)
    y_train_sliced = extract_slices(y_train, window_len, stride_len)



    return x_train_sliced, y_train_sliced[:,-1,:], x_dev, y_dev, x_test, y_test


def extract_slices(x, window_len, stride_len):
    """
    Divide train data to small samples of input sequences with provided window and step length
    """

    width = x.shape[1]
    x_sliced = x[:, 0:window_len, :]
    count = 1
    index_end = 0

    while  index_end+stride_len <= width:

        # extract input sequences
        index_start = count*stride_len
        index_end = count*stride_len + window_len

        x_temp = x[:, index_start:index_end, :]

        x_sliced = np.vstack((x_sliced, x_temp))

        count += 1

    print('>> stride count: ', count)
    return x_sliced


if __name__ == '__main__':

    # test the
    read_data(50, 10, 0.5)
