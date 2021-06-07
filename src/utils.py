"""
The module for reading data and training.
"""
import scipy.io as sio
import numpy as np
import yaml
import matplotlib.pyplot as plt
import time
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from noise_generate import add_noise
from tensorboardX import SummaryWriter


def train_model(model, device, data_train, data_dev, x_dev, y_dev, optimizer, criterion,
        num_epochs, model_save_path,  window_len, stride_len, valid_period):
    """Training function"""
    print('Start training the model')
    writer = SummaryWriter(comment='__' + 'Overtesting')

    # weight decay
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    for epoch in range(num_epochs):

        start = time.time()
        running_loss = 0.0
        running_loss2 = 0.0
        count = 0
        for i, data in enumerate(data_train):
            count += 1
            # get the inputs; data is a list of [inputs, labels]
            inputs = data['features'].float().to(device)
            labels = data['labels'].type(torch.cuda.FloatTensor).squeeze(1).to(device)
            #zero the parameter gradients
            optimizer.zero_grad()
            # forward propagation
            output = model(inputs)
            loss = criterion(output, labels)
            running_loss2 += loss.item()
            # add L2 regularization
            add_L2 = True
            if add_L2:
                l2_reg = 0.0
                for W in model.parameters():
                    l2_reg = l2_reg + W.norm(2)
                loss = loss + 1.0/(2*output.size(0))*l2_reg * 0.001
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        print('Epoch: {}, Iter: {}, Loss: {}, Elapsed: {}'.format(epoch+1, i+1,
            running_loss/i, time.time()-start))
        writer.add_scalar('Train_L2_loss: ', running_loss/i)
        with open(model_save_path + '_train_L2_loss.txt', "a") as myfile:
            myfile.write(str(running_loss / count))
            myfile.write("\n")

        scheduler.step()

        if epoch % valid_period == 0:
            # save the model
            st_time = time.time()
            model.eval()
            #torch.save(model.state_dict(), model_save_path + '_ep'+ str(epoch+1) + '.pth')
            sample_input = torch.from_numpy(np.ones((1,window_len,6))).float().to(device)
            traced_script_module = torch.jit.trace(model, sample_input)
            traced_script_module.save(model_save_path + '.pt')
            running_loss3 = 0.0
            count = 0
            for i_v, data_v in enumerate(data_dev):
                count += 1
                inputs = data_v['features'].float().to(device)
                labels = data_v['labels'].type(torch.cuda.FloatTensor).squeeze(1).to(device)
                output = model(inputs)
                loss = criterion(output, labels)
                running_loss3 += loss.item()

            with open(model_save_path + '_valid_loss.txt', "a") as myfile:
                myfile.write(str(running_loss3 / count))
                myfile.write("\n")
            
            writer.add_scalar('Val_loss', running_loss3 / i_v)
            print('Eval >>> : ', running_loss3 / i_v, 'Time>>> ', time.time() - st_time)

            model.train()

def test_model(model, device, x_data, y_data, plot_enable, window_len,
        stride_len, model_save_path):

    """Test the model."""
    config = yaml.safe_load(open("config.yml"))
    valid_size = config['valid_size']

    start = time.time()
    stride_test = 1
    window_test = window_len
    width = x_data.shape[1]
    plot_label = np.zeros((1,2))
    plot_output = np.zeros((1,2))

    sum_rmse = 0
    count_rmse = 0
    with torch.no_grad():
        for i in range(valid_size):
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

                plot_label = np.vstack((plot_label, labels.cpu().detach().numpy()))
                plot_output = np.vstack((plot_output, output.cpu().detach().numpy()))

                count += 1

            error_rmse = np.sqrt(np.mean((plot_label-plot_output)**2, axis = 0))
            sum_rmse += error_rmse
            count_rmse += 1
            
            if plot_enable == 'True':
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

                plt.show()

    mean_rmse = np.mean(sum_rmse/count_rmse)

    print('# test: ', count_rmse, 'T sequence: ', count, 'mean_rmse: ', mean_rmse)
    with open(model_save_path + '_val_loss.txt', "a") as myfile:
        myfile.write(str(mean_rmse))
        myfile.write("\n")
    
    # mean and std dev
    end = time.time()
    print('Elapsed time for evaluation', end-start)

    return mean_rmse

def read_data(window_len, stride_len, n_times):
    """
    Read the data and return normalized inputs.
    """
    config = yaml.safe_load(open("config.yml"))
    state_len = config['state_len']
    n_times_orig = config['n_times_orig']

    path = '../data/'
    state = np.empty([500,401,6])
    control = np.empty([500,401,2])

    for i in range(1,8):
        fname = 'outputset' + str(i) + '.mat'
        temp_data = sio.loadmat(path + fname)
        
        control = np.concatenate((control,  temp_data['controls']))
        state = np.concatenate((state,  temp_data['states']))

    state = state[:,0:state_len,:]
    control = control[:,0:state_len,:]
    
    # ADD FAULT
    state, control = add_noise(state,control, n_times, n_times_orig)
    print('>> After fault:', state.shape  )

    train_ratio = config['train_set_ratio']
    dev_ratio = config['dev_set_ratio']
    test_ratio = config['test_set_ratio']

    # shuffle
    # divide into train/dev/test set
    x_train, x_test, y_train, y_test = train_test_split(state, control, 
            test_size=1-train_ratio, shuffle=True, random_state=1)
    x_dev, x_test, y_dev, y_test = train_test_split(x_test, y_test, 
            test_size=test_ratio/(test_ratio + dev_ratio), shuffle=False)

    # divide test into windows
    # x_train_sliced -> input sequence
    # x_train_sliced -> control input at the end of the input sequence

    x_train_sliced = extract_slices(x_train, window_len, stride_len)
    y_train_sliced = extract_slices(y_train, window_len, stride_len)
    x_dev_sliced = extract_slices(x_dev, window_len, stride_len)
    y_dev_sliced = extract_slices(y_dev, window_len, stride_len)

    return x_train_sliced, y_train_sliced[:,-1,:], x_dev_sliced, y_dev_sliced[:,-1,:], x_test, y_test



def extract_slices(x, window_len, stride_len):
    """
    Divide train data to small samples of input sequences with 
    provided window and step length
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
