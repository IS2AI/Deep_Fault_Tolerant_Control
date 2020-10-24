import os
import sys
import io

import time
import csv
import copy
import scipy
import scipy.io as sio

import math
from math import cos, sin, sqrt, pi
import numpy as np
import matplotlib.pylab as plt
from matplotlib import cm
from matplotlib import image
from matplotlib import pyplot

import random
from random import randint
from random import randrange

plt.style.use('ggplot')

# Separate position and velocity
def fail_traj_upd(state,ind):

    rand_time = randrange(30,200,5)   # between 30 and 200
    fault_states = [randint(0,5)]
    #irand = randint(0,1)
    #if irand == 1:
    #    fault_states = [randint(0,5)]
    #else:
    #    fault_states = [randint(0,1)]
    
    new_traj = copy.copy(state[ind,:,:])

    for f_st in fault_states:
        if f_st<2:
            vals = [0,state[ind,rand_time,f_st],pi,-pi]
            new_traj[rand_time:state.shape[1],f_st] = vals[randint(0,3)]
        else:
            vals = 0
            new_traj[rand_time:state.shape[1],f_st] = vals
        return new_traj, ind

# Algorithm to generate single dataset
def fail_traj_single(state, fault_states, rand_val, state_at_fault):

    new_traj = copy.copy(state)

    for f_st in fault_states:
        if f_st<2:
            vals = [0, state_at_fault[0, f_st], pi,-pi]
            new_traj[0,f_st] = vals[rand_val]
        else:
            vals = 0
            new_traj[0,f_st] = vals

    return new_traj

# generate noise
def add_noise(state,control, n_times, n_times_orig):

    if n_times > 0:
        state_temp = copy.copy(state)
        control_temp = copy.copy(control)

        ss = [] # to save states
        cc = [] # to save control inputs
        for i in range(n_times):
          for j in range(0,state.shape[0]-1):
            new_traj, rand_ind = fail_traj_upd(state,j)
            ss.append(new_traj)
            cc.append(control[j])

        ss = np.array(ss)
        cc = np.array(cc)

        state = np.concatenate((state,ss))
        control = np.concatenate((control,cc))

        for i in range(n_times_orig):
            state = np.concatenate((state, state_temp))
            control = np.concatenate((control, control_temp))

    return state, control

# convinient for plotting

def state_comp_plotter(state1,state2):
  time = np.arange(0,2.5,0.01)
  plt.subplot(611)
  plt.plot(time,state1[:,0],'k-')
  plt.plot(time,state2[:,0],'r-')
  plt.ylabel('x-position, rad')
  plt.subplot(612)
  plt.plot(time,state1[:,1],'k-')
  plt.plot(time,state2[:,1],'r-')
  plt.ylabel('y-position, rad')
  plt.subplot(613)
  plt.plot(time,state1[:,2],'k-')
  plt.plot(time,state2[:,2],'r-')
  plt.ylabel('x-velosity, rad/s')
  plt.subplot(614)
  plt.plot(time,state1[:,3],'k-')
  plt.plot(time,state2[:,3],'r-')
  plt.ylabel('y-velocity, rad/s')
  plt.subplot(615)
  plt.plot(time,state1[:,4],'k-')
  plt.plot(time,state2[:,4],'r-')
  plt.ylabel('reaction wheel 1, rad/s')
  plt.subplot(616)
  plt.plot(time, state1[:,5],'k-')
  plt.plot(time, state2[:,5],'r-')
  plt.ylabel('reaction wheel 2, rad/s')
  plt.xlabel('time, ms')
  plt.show()


def input_comp_plotter(input1,input2):
  plt.subplot(211)
  plt.plot(time,input1[:,0],'k-')
  plt.plot(time,input2[:,0],'r-')
  plt.subplot(212)
  plt.plot(time, input1[:,1],'k-')
  plt.plot(time, input2[:,1],'r-')
  plt.xlabel('time, ms')
  plt.show()




if __name__ == '__main__':

    path = '../data/'
    file = "outputset1.mat"
    a = sio.loadmat(path + file)
    file = "outputset2.mat"
    b = sio.loadmat(path + file)

    control=np.concatenate((a['controls'], b['controls']))
    state=np.concatenate((a['states'], b['states']))

    state = state[:,0:250,:]
    control = control[:,0:250,:]

    for ind in range(10):
        #generate dataset
        new_traj,rand_ind = fail_traj_upd(state,1)
        #plot dataset with original
        state_comp_plotter(state[rand_ind,:,:],new_traj)
