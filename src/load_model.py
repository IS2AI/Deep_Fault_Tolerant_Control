import os
import sys
import io
import numpy as np

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from torch.autograd import Variable
import torch.utils.data as data_utils
import torchvision

# import yaml

from models import SeRNN_FW, SeRNN

print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

device =  torch.device("cuda:0")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

window_len = 10
load_model_name = 'Model_WW4_test_ep100'

#modelx = SeRNN_FW(1, device)
#modelx.load_state_dict('../checkpoints/' + load_model_name + '.pt')


# ###########################################################################
# load trace GPU
#model = torch.jit.load('../checkpoints/' + load_model_name + '.pth')
#print('model: ', model)
#model.eval()
#params = list(model.parameters())
#print(params)

# TEST the MODEL
sample_input = torch.from_numpy(np.ones((1,window_len,6))).float().to(device)
#print(sample_input)
#print('Test the model: ', model(sample_input))

# ############################################################################
# SAVE the Model
#

#save_model_name = load_model_name + '_testing'
#model_save_path = '../checkpoints/' + save_model_name + '.pt'

#device =  torch.device("cpu")
# model.to(device)
#
# sample_input = torch.from_numpy(np.ones((1,window_len,6))).float().to(device)
#
# traced_script_module = torch.jit.trace(model, sample_input)
#
# traced_script_module.save(model_save_path)
#
# print('SAve the model: ', save_model_name)


# ############################################################################
# MODEL LOADED

model2 = SeRNN_FW(1, device)
model2.to(device)

model2.eval()
#print('model2: ', model2)

model2.load_state_dict(torch.load('../checkpoints/' + load_model_name + '.pth'))

x, lstm_output = model2(sample_input)
xl = lstm_output[:, -1, :]
print('Test the model2: ',x)




# LOAD TO CPU

#
# device2 = torch.device("cpu")
# #
# model2 = torch.jit.load('../checkpoints/' + load_model_name + '.pt', map_location=device)
#
# print('dev2 ', device2)
#
#
# #
# model2.eval()
# model2 = model2.cpu()
# # #
# model2 = model2.to(device2)
#
# print('Load the model2 : ', model2)
#
# sample_input = torch.from_numpy(np.ones((1,window_len,6))).float().to(device2)
#
# #
# print('Test the model2: ', model2(sample_input))


# # SAVE the Model
#
# save_model_name = load_model_name + '_cpu'
# model_save_path = '../checkpoints/' + save_model_name + '.pt'
# device =  torch.device("cpu")
# model.to(device)
#
# sample_input = torch.from_numpy(np.ones((1,window_len,6))).float().to(device)
#
# traced_script_module = torch.jit.trace(model, sample_input)
#
# traced_script_module.save(model_save_path)
# print('SAve the model: ', save_model_name)
