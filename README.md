# Deep_Fault_Tolerant_Control
Implementation of deep fault tolerant control for inverted pendulum with reaction wheels



# Requirements
1. Pytorch

# Create conda env and install packages

1. conda create -n pytorch_env python=3.8
2. conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
3. conda install -c anaconda pip
4. conda install -c anaconda numpy
5. conda install -c conda-forge matplotlib
6. conda install -c anaconda scipy
7. conda install -c anaconda scikit-learn
8. conda install pyyaml


# How to run
1. Install packages above
2. Go to directory src/
3. Set hyperparameters in main.py
4. Run python main.py 0 - for training
5. Run python main.py 1 - for testing
