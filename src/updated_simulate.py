import os
import sys
import io

import time
#import h5py
import csv
import copy
import scipy
import scipy.io as sio

import math
from math import cos, sin, sqrt
import numpy as np
import matplotlib.pylab as plt
from matplotlib import cm
from matplotlib import image
from matplotlib import pyplot

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

import random
from random import randint
from random import randrange

from utils import read_data
from noise_generate import fail_traj_single
import pandas as pd

from models import RNN, FNN

plt.style.use('ggplot')
import yaml


class sys:
  a=5
  lc1 = [0,0,0.33]
  m1 = 1.3
  I1 =np.eye(3,3)
  I1[0,0] = 0.18
  I1[1,1] = 0.18
  I1[2,2] = 0.01
  Iwm = 1e-9*np.array([753238.87,394859.64,394859.64])
  Iwp = 1e-9*np.array([753238.87,394859.64,394859.64])
  K = 131e-3
  k1 = 13.70937911565217391304347826087
  k2 = 6.0195591156521739130434782608696
  g = 9.81
  b1 = 0.7e-1
  b2 =2.5e-1
  bwm =1.1087e-4
  bwp=9.4514e-5

def mass_matrix(I11_1,I11_2,I11_3,I12_1,I12_2,I12_3,I13_1,I13_2,I13_3,Iwm1,Iwm2,Iwm3,Iwp1,Iwp2,Iwp3,theta1,theta2):
      t2 = cos(theta1);
      t3 = sin(theta1);
      t4 = I11_2*t2*(1.0/2.0);
      t5 = I12_1*t2*(1.0/2.0);
      t6 = t4+t5-I12_3*t3*(1.0/2.0)-I13_2*t3*(1.0/2.0);
      t8 = sin(theta2);
      t11 = t3*t8;
      t7 = t2+t11;
      t13 = t2*t8;
      t9 = t3-t13;
      t10 = cos(theta2);
      t12 = t2-t11;
      t14 = t3+t13;
      t15 = t10**2;
      M = np.array([I12_2,t6,0.0,0.0,t6,t2*(I11_1*t2-I13_1*t3)-t3*(I11_3*t2-I13_3*t3),0.0,0.0,0.0,0.0,Iwp2*t15*(1.0/2.0)+Iwp1*t7**2*(1.0/2.0)+Iwp3*t9**2*(1.0/2.0),0.0,0.0,0.0,0.0,Iwm2*t15*(1.0/2.0)+Iwm1*t12**2.*(1.0/2.0)+Iwm3*t14**2.*(1.0/2.0)],dtype='float').reshape(4,4)
      return M

def acting_forces(I11_1,I11_2,I11_3,I12_1,I12_3,I13_1,I13_2,I13_3,Iwm1,Iwm2,Iwm3,Iwp1,Iwp2,Iwp3,K,b1,b2,bwm,bwp,dth1,dth2,dth3,dth4,g,im,ip,k1,k2,lc11,lc12,lc13,m1,theta1,theta2):
      t2 = dth2**2;
      t3 = dth4**2;
      t4 = sin(theta2);
      t5 = dth3**2;
      t6 = cos(theta1);
      t7 = t6**2;
      t8 = theta1*2.0;
      t9 = sin(t8);
      t10 = sqrt(2.0);
      t11 = cos(theta2);
      t12 = sin(theta1);
      t13 = t11**2;
      t14 = cos(t8);
      t15 = t10*t12*(1.0/2.0);
      t16 = t4*t6*t10*(1.0/2.0);
      t17 = t6*t10*(1.0/2.0);
      t18 = K*ip;
      t19 = K*im;
      t20 = t12**2;
      t21 = t6*t12*t13;
      Vv = np.array([[I11_3*t2*(1.0/2.0)+I13_1*t2*(1.0/2.0)-b1*dth1-k1*theta1-I11_1*t2*t9*(1.0/2.0)-I11_3*t2*t7-I13_1*t2*t7+I13_3*t2*t9*(1.0/2.0)+
            Iwm1*t3*t4*(1.0/2.0)-Iwm3*t3*t4*(1.0/2.0)-Iwp1*t4*t5*(1.0/2.0)+Iwp3*t4*t5*(1.0/2.0)-Iwm1*t3*t4*t7+Iwm3*t3*t4*t7+Iwp1*t4*t5*t7-
            Iwp3*t4*t5*t7-bwm*dth4*t10*t11*(1.0/2.0)+bwp*dth3*t10*t11*(1.0/2.0)-g*lc11*m1*t6+K*im*t10*t11*(1.0/2.0)-K*ip*t10*t11*(1.0/2.0)-
            Iwm1*t3*t6*t12*t13*(1.0/2.0)+Iwm3*t3*t6*t12*t13*(1.0/2.0)-Iwp1*t5*t6*t12*t13*(1.0/2.0)+Iwp3*t5*t6*t12*t13*(1.0/2.0)-g*lc12*m1*t4*t12-

            g*lc13*m1*t11*t12],
            [-b2*dth2+dth1*(I11_1*dth2*t9*2.0+I11_2*dth1*t12+I11_3*dth2*t14*2.0+I12_3*dth1*t6+I12_1*dth1*t12+I13_2*dth1*t6-I13_3*dth2*t9*2.0+
                                                   I13_1*dth2*t14*2.0)*(1.0/2.0)-k2*theta2-t10*(t19-bwm*dth4)*(1.0/2.0)-t10*(t18-bwp*dth3)*(1.0/2.0)-g*m1*(lc13*t4*t6-lc12*t6*t11)-
            Iwm2*t3*t4*t11*(1.0/2.0)-Iwp2*t4*t5*t11*(1.0/2.0)-Iwm1*t3*t10*t11*t12*(t17-t4*t10*t12*(1.0/2.0))*(1.0/2.0)+Iwp1*t5*t10*t11*t12*(t17+t4*t10*t12*(1.0/2.0))*(1.0/2.0)-
            Iwp3*t5*t6*t10*t11*(t15-t16)*(1.0/2.0)+Iwm3*t3*t6*t10*t11*(t15+t16)*(1.0/2.0)],[t18-bwp*dth3-dth3*(dth2*t11*(-Iwp2*t4+Iwp1*t9*(1.0/2.0)-Iwp3*t9*(1.0/2.0)+
                                                                                                                                         Iwp3*t4*t7+Iwp1*t4*t20)-
                                                                                                                             dth1*(Iwp1-Iwp3)*(t4+t21-t4*t7*2.0))],
            [t19-bwm*dth4+dth4*(dth1*(Iwm1-Iwm3)*(-t4+t21+t4*t7*2.0)-dth2*t11*(-Iwm2*t4-Iwm1*t9*(1.0/2.0)+Iwm3*t9*(1.0/2.0)+Iwm3*t4*t7+Iwm1*t4*t20))]],dtype='float').reshape(4,1);
      return Vv

def model_dynamics(x,u,sys):
    M = mass_matrix(sys.I1[0,0],sys.I1[0,1],sys.I1[0,2],sys.I1[1,0],sys.I1[1,1],sys.I1[1,2] ,sys.I1[2,0],sys.I1[2,1],sys.I1[2,2],sys.Iwm[0],sys.Iwm[1],sys.Iwm[2],sys.Iwp[0],sys.Iwp[1],sys.Iwp[2],x[0,0],x[1,0])
    V=acting_forces(sys.I1[0,0],sys.I1[0,1],sys.I1[0,2],sys.I1[1,0], sys.I1[1,2],sys.I1[2,0],sys.I1[2,1],sys.I1[2,2],sys.Iwm[0],sys.Iwm[1],sys.Iwm[2],sys.Iwp[0],sys.Iwp[1],sys.Iwp[2],sys.K,sys.b1,sys.b2,sys.bwm,sys.bwp,x[2,0],
                    x[3,0],x[4,0],x[5,0], sys.g,u[0],u[1],sys.k1,sys.k2,sys.lc1[0],sys.lc1[1],sys.lc1[2],sys.m1,x[0,0],x[1,0]);
    dq = np.linalg.solve(M,V)
    dx = np.concatenate((x[2,0].reshape(1,1),x[3,0].reshape(1,1),dq),axis=0);
    return dx


def normalized_cost_sample(states_true, inputs_nn, states_ocp, control_ocp):
    Q=np.eye(6)
    R=np.eye(2)

    Q[0,0]=50000
    Q[1,1]=50000
    Q[2,2]=500
    Q[3,3]=100
    Q[4,4]=0.01
    Q[5,5]=0.01
    R[0,0]=10**(-5)
    R[1,1]=10**(-5)

    cost_ocp=0
    for i in range(sim_len):
        one=abs(states_ocp[i,:].reshape(1,6))
        two=abs(control_ocp[i,:].reshape(1,2))

        cost_ocp=cost_ocp+np.matmul(one,np.matmul(Q,np.transpose(one)))+np.matmul(two,np.matmul(R,np.transpose(two)))

    cost_nn=0
    for i in range(sim_len):
        one1=states_true[i,:].reshape(1,6)
        two1=inputs_nn[i,:].reshape(1,2)

        cost_nn=cost_nn+np.matmul(one1,np.matmul(Q,np.transpose(one1)))+np.matmul(two1,np.matmul(R,np.transpose(two1)))

    return cost_ocp, cost_nn

def normalized_cost_sample_new(states_true, inputs_nn, states_ocp, control_ocp):

    sim_len = 401
    Q=np.eye(6)
    R=np.eye(2)

    Q[0,0]=50000
    Q[1,1]=50000
    Q[2,2]=500
    Q[3,3]=100
    Q[4,4]=0.01
    Q[5,5]=0.01
    R[0,0]=10**(-5)
    R[1,1]=10**(-5)

    cost_ocp=0
    for i in range(sim_len):
        one=abs(states_ocp[i,:].reshape(1,6))
        two=abs(control_ocp[i,:].reshape(1,2))

        cost_ocp=cost_ocp+np.matmul(one,np.matmul(Q,np.transpose(one)))+np.matmul(two,np.matmul(R,np.transpose(two)))

    cost_nn=0
    for i in range(sim_len):
        one1=states_true[i,:].reshape(1,6)
        two1=inputs_nn[i,:].reshape(1,2)

        cost_nn=cost_nn+np.matmul(one1,np.matmul(Q,np.transpose(one1)))+np.matmul(two1,np.matmul(R,np.transpose(two1)))

    return cost_ocp, cost_nn

def simulate(xs0, sys, model, win_len, f1, f2, f3, index, sim_normalize, sim_fault):

    Ts=0.01

    # add fault to simulate
    fault_time = f1
    fault_states = [f2]
    fault_val = f3

    # save states/inputs
    state_to_nn=[]
    inputs_nn=[]
    states_true=[]

    x0=xs0.reshape(win_len,6)

    faulty_state_to_nn_array = x0
    x_true_next = x0[-1:, :]

    for ind in range(sim_len):

        x_true_curr = copy.copy(x_true_next)
        x_true_curr = x_true_curr.reshape(6,1)

        x_faulty_seq = faulty_state_to_nn_array[-win_len:, :]
        x_faulty_seq = np.expand_dims(x_faulty_seq, axis=0)  # comment for old FNN -  my_model2.pt

        x_faulty_seq = torch.from_numpy(x_faulty_seq)
        x_faulty_seq = x_faulty_seq.float().to(device)

        y = model(x_faulty_seq)

        # lstm_feat = activation('fc3')
        # print('--->>> ',  lstm_feat)

        u=y.reshape(2,1).cpu()
        u = torch.clamp(u, -3, 3)

        k1 = model_dynamics(x_true_curr,u,sys)
        k2 = model_dynamics(x_true_curr + (Ts/2)*k1,u,sys)
        k3 = model_dynamics(x_true_curr + (Ts/2)*k2,u,sys)
        k4 = model_dynamics(x_true_curr+Ts*k3,u,sys)

        x_true_next = x_true_curr + Ts*(k1 + 2*k2 + 2*k3 + k4)/6

        # save current state values after x0
        state_to_nn.append(faulty_state_to_nn_array[-1:, :])
        inputs_nn.append(u)
        states_true.append(x_true_curr)

        x_faulty_next = copy.copy(x_true_next)
        x_faulty_next = x_faulty_next.reshape(1,6)

        if sim_normalize == True:
            #normalize: option 3 - Z Score
            for ind_state in range(6):
                #std_temp = [0.08, 0.11, 0.94, 0.84, 71.0, 71.0]
                #std_temp = [0.63, 0.63, 0.94, 0.84, 71.0, 71.0]
                std_temp = [0.103, 0.145, 1.2, 1.073, 90.71, 89.97]
                x_faulty_next[:,ind_state] = x_faulty_next[:, ind_state]/std_temp[ind_state] # mean is zero
                pass
        # add fault
        if ind == fault_time:
            state_at_fault = copy.copy(x_faulty_next)

        if ind >= fault_time and sim_fault == True:
            x_faulty_next = fail_traj_single(x_faulty_next, fault_states, fault_val, state_at_fault)
            pass

        faulty_state_to_nn_array = np.concatenate((faulty_state_to_nn_array,x_faulty_next),axis=0)

    return state_to_nn, inputs_nn, states_true

def plot_states(states_true, state_to_nn, states_ocp, inputs_nn, control_ocp, win_len, is_plot, x0, y0):

    file_st = '../checkpoints/' + '_rnn_states_fault_pos1x' +'.txt'


    states_true_plot = np.concatenate((x0, states_true), axis= 0)
    state_to_nn_plot = np.concatenate((x0, state_to_nn), axis= 0)
    states_ocp_plot = np.concatenate((x0, states_ocp), axis= 0)
    inputs_nn_plot = np.concatenate((y0, inputs_nn), axis= 0)
    control_ocp_plot = np.concatenate((y0, control_ocp), axis= 0)

    with open(file_st, "w") as myCsv:
         csvWriter = csv.writer(myCsv, delimiter=' ')
         print(states_true_plot.shape)
         print(state_to_nn_plot.shape)
         print(states_ocp_plot.shape)
         print(inputs_nn_plot.shape)
         print(control_ocp_plot.shape)
         all_data = np.concatenate((states_true_plot, state_to_nn_plot, states_ocp_plot[0:400,:], inputs_nn_plot, control_ocp_plot[0:400,:]), axis = 1)
         print(all_data.shape)
         csvWriter.writerows(all_data)

    if is_plot == True:

        plt.subplot(811)
        plt.plot(states_true[:,0],'yo')
        plt.plot(state_to_nn[:,0],'k--')
        plt.plot(states_ocp[:,0],'r-')
        plt.legend(('RNN trajectory','Faulty trajectory', 'OCP trajectory'),loc="upper right")
        plt.title('State 1 - link y axis position')

        plt.subplot(812)
        plt.plot(states_true[:,1],'yo')
        plt.plot(state_to_nn[:,1],'k--')
        plt.plot(states_ocp[:,1],'r-')
        # plt.legend(('RNN trajectory','Faulty trajectory', 'OCP trajectory'))
        plt.title('State 2 - link x axis position')

        plt.subplot(813)
        plt.plot(states_true[:,2],'yo')
        plt.plot(state_to_nn[:,2],'k--')
        plt.plot(states_ocp[:,2],'r-')
        # plt.legend(('RNN trajectory','Faulty trajectory', 'OCP trajectory'))
        plt.title('State 3 - link y axis velocity')

        plt.subplot(814)
        plt.plot(states_true[:,3],'yo')
        plt.plot(state_to_nn[:,3],'k--')
        plt.plot(states_ocp[:,3],'r-')
        # plt.legend(('RNN trajectory','Faulty trajectory', 'OCP trajectory'))
        plt.title('State 4 - link y axis velocity')

        plt.subplot(815)
        plt.plot(states_true[:,4],'yo')
        plt.plot(state_to_nn[:,4],'k--')
        plt.plot(states_ocp[:,4],'r-')
        # plt.legend(('RNN trajectory','Faulty trajectory', 'OCP trajectory'))
        plt.title('State 5 - Wheel 1 velocity')

        plt.subplot(816)
        plt.plot(states_true[:,5],'yo')
        plt.plot(state_to_nn[:,5],'k--')
        plt.plot(states_ocp[:,5],'r-')
        plt.title('State 6 - Wheel 2 velocity')

        # plt.xlabel('time, ms')
        # plt.show()

        plt.subplot(817)
        plt.plot(inputs_nn[:,0],'k-')
        plt.plot(control_ocp[:,0],'r-')
        plt.legend(('NN input', 'OCP input'), loc="upper right")
        plt.title('Motor 1 input')

        plt.subplot(818)
        plt.plot(inputs_nn[:,1],'k-')
        plt.plot(control_ocp[:,1],'r-')
        plt.title('Motor 2 input')
        plt.xlabel('Time [s]')
        # plt.legend(('NN input', 'OCP input'))


        plt.show()

    # save_to_scv = True
    # print(states_true.shape)
    # if save_to_scv == True:
    #     print('... saving figures')
    #     rows = zip(states_true[:], state_to_nn, states_ocp, inputs_nn, control_ocp)
    #     file_states = '../checkpoints/' + '_states_fault_non' + str(version) +'.txt'
    #     with open(file_states, "w") as f:
    #         writer = csv.writer(f)
    #         for row in rows:
    #             writer.writerow(row)





#######


if __name__ == '__main__':


    # path = '../data/'
    # file = "cost_fault_wheel.mat"
    # j = sio.loadmat(path + file)
    # cost_exp = j['cost_f']
    #
    # print('-->', cost_exp.shape)
    #
    # var1 = cost_exp[:,0:6]
    # var2 = cost_exp[:,6:8]
    # var3 = cost_exp[:,8:14]
    # var4 = cost_exp[:,14:16]
    #
    # cost_exp1, cost_exp2 = normalized_cost_sample_new(var1, var2, var3, var4)
    # #
    # print('cost _cal finished', cost_exp1, cost_exp2, cost_exp1/cost_exp2)

    start = time.time()

    config = yaml.safe_load(open("config.yml"))

    ############################
    # Model
    ### ########################

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sim_model_name = config['sim_model_name']

    model = torch.jit.load('../checkpoints/' + sim_model_name + '.pt')
    model.eval()
    model.to(device)

    print('# param:', sum(p.numel() for p in model.parameters()))

    ############################
    # DATA
    ### ########################

    path = '../data/'
    file = "outputset10.mat"
    j = sio.loadmat(path + file)

    state_orig = j['states']
    control_orig = j['controls']

    #############################
    # SIMULATION
    ### ########################

    sim_num = config['sim_num']
    sim_len = config['sim_len']
    win_len = config['sim_window_len']
    sim_normalize = config['sim_normalize']
    sim_fault = config['sim_fault']
    sim_manual_fault_type = config['sim_manual_fault_type']
    sim_save_result = config['sim_save_result']

    is_plot = config['is_plot']
    cost_list = []
    cost_list_ocp = []
    cost_list_nn = []
    index_list = []

    # generate same type of faults
    np.random.seed(1)
    rand_fault_time = np.random.randint(170, size=sim_num) + 30 # 30 - 200

    np.random.seed(2)
    rand_fault_states = np.random.randint(2, size=sim_num) + sim_manual_fault_type
    #rand_fault_states = np.random.randint(6, size=sim_num)

    np.random.seed(3)
    rand_fault_val = np.random.randint(4, size=sim_num)


    print('Simulating the model: ', sim_model_name)
    #print(model)

    if sim_fault == False:
        version = 'n'
    else:
        if sim_manual_fault_type == 0:
            version = 'a'
        elif sim_manual_fault_type == 2:
            version = 'b'
        elif sim_manual_fault_type == 4:
            version = 'c'
        elif sim_manual_fault_type == 8:
            version = 'x'
        else:
            raise ValueError('Not valid sim_manual_fault_type')


    file_cost = '../checkpoints/' + sim_model_name + '_del2_cost_' + str(version) +'.txt'
    print('File : ', file_cost)

    for index in range(70,sim_num):
        fault_time = rand_fault_time[index]
        fault_states = rand_fault_states[index]
        fault_val = rand_fault_val[index]

        # Simulate ONE traj on certain initial state xs
        num = index

        # Manual mode
        # fault_time = 30
        # fault_states = 2
        # fault_val = 3
        #print(index, fault_states)

        # Read OCP solution
        states_ocp=state_orig[num,:,:]
        control_ocp=control_orig[num,:,:]

        xs=states_ocp[0:win_len,:]
        ys=control_ocp[0:win_len,:]

        states_ocp = states_ocp[win_len:,:]
        control_ocp = control_ocp[win_len:,:]

        # Simulate NN solution
        state_to_nn_list,inputs_nn_list,states_true_list = simulate(xs, sys, model, win_len, fault_time, fault_states, fault_val, index, sim_normalize, sim_fault)

        # predicted
        #state_to_nn=[t.numpy() for t in state_to_nn_list]
        inputs_nn=[t.cpu().detach().numpy() for t in inputs_nn_list]
        #states_true=[t.numpy() for t in states_true_list]

        state_to_nn=np.array(state_to_nn_list).reshape(sim_len,6)
        inputs_nn=np.array(inputs_nn).reshape(sim_len,2)
        states_true=np.array(states_true_list).reshape(sim_len,6)

        # print('---')
        # print(states_true.shape)
        # print(inputs_nn.shape)
        # print(states_ocp.shape)
        # print(control_ocp.shape)
        # print('***')
        # Calculate cost
        cost_ocp, cost_nn = normalized_cost_sample(states_true, inputs_nn, states_ocp, control_ocp)

        if cost_ocp>0:
            norm_ratio = cost_nn/cost_ocp

            cost_list.append(float(norm_ratio))
            index_list.append(index)

            if norm_ratio>2:
                print('huge --', index, norm_ratio)

            if is_plot == True:
                print('index - {}, fault type - {}, fault value - {}, cost_norm - {}'.format(index, fault_states, fault_val, cost_nn/cost_ocp))
                plot_states(states_true, state_to_nn, states_ocp, inputs_nn, control_ocp, win_len, is_plot, xs, ys)

    # Calculate norm cost
    cost_array = np.array(cost_list)
    cost_array = cost_array.reshape(len(cost_list), 1)
    cost_norm = sum(cost_array)/len(cost_list)
    cost_std = np.std(cost_array)
    print('Average cost: ', cost_norm, 'Std: ', cost_std, 'Max: ', np.max(cost_array) )

    if sim_save_result == True:
        print('... saving')
        rows = zip(index_list, cost_list)
        with open(file_cost, "w") as f:
            writer = csv.writer(f)
            for row in rows:
                writer.writerow(row)

    end = time.time()
    print('Elapsed: ', end-start)

    # PLOT Costs
    # plt.plot( cost_array,'r-')
    # plt.xlabel('time, ms')
    # plt.legend(('Cost'))
    # plt.show()
