"""
This scripts launches the model training with specified configurations.
"""

#imports
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import ControlDataset, ToTensor
from models import FNN, SeRNN_FWXX
from utils import train_model, test_model, read_data

def main():
    """Run training."""
    config = yaml.safe_load(open("config.yml"))
    #training hyperparameters
    num_epochs = config['num_epochs']
    learning_rate = config['learning_rate']
    batch_size = config['batch_size']
    valid_period = config['valid_period']
    #data hyperparameters
    #extract sample (input sequences with length = stride_len)
    #for RNN from the givent full trajectory
    window_len = config['window_len']
    stride_len = config['stride_len']
    n_times = config['n_times']
    x_train, y_train, x_dev, y_dev, x_test, y_test = read_data(
        window_len, stride_len, n_times)
    print('Dataset:', x_train.shape, y_train.shape, x_dev.shape,
        y_dev.shape, x_test.shape, y_test.shape)
    #create dataset
    transformed_control_dataset_train = ControlDataset(x_train, y_train,
        transform=transforms.Compose([ToTensor()]))
    #transformed_control_dataset_dev = ControlDataset(x_dev, y_dev,
        #transform=transforms.Compose([ToTensor()]))
    #transformed_control_dataset_test = ControlDataset(x_test, y_test,
        #transform=transforms.Compose([ToTensor()]))
    # create batch
    data_train = DataLoader(transformed_control_dataset_train,
        batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True)
    #data_dev = DataLoader(transformed_control_dataset_dev, batch_size=1,
        #shuffle=False, num_workers=1, drop_last=True)
    #data_test = DataLoader(transformed_control_dataset_test, batch_size=1,
        #shuffle=False, num_workers=1, drop_last=True)

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device:', device)

    # save model
    save_model_name = config['model_name']
    print('Save Model name: ', save_model_name)
    model_save_path = '../checkpoints/' + save_model_name #+ '.pt' ## not pth

    #test/train
    mode = int(sys.argv[1])

    if mode == 0:
        #train mode
        # Model preparation
        model = SeRNN_FWXX(batch_size, device)
        print('Model: ', model)

        model.to(device)
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print('# of params: ', pytorch_total_params)

        # multiple GPUs
        if torch.cuda.device_count() > 1:
            print("Model uploaded. Number of GPUs: ", torch.cuda.device_count())
            #model = nn.DataParallel(model)

        #set the training loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
        train_model(model, device, data_train, x_dev, y_dev, optimizer, criterion,
            num_epochs, model_save_path, window_len, stride_len, valid_period)
    else:
        # test mode
        # upload saved model
        print('Saved Model evaluation with test set')
        model = torch.jit.load(model_save_path)
        test_model(model, device, x_test, y_test, 'True', window_len, stride_len)

if __name__ == '__main__':
    main()
