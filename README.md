## Deep_Fault_Tolerant_Control

Implementation of deep fault tolerant control for inverted pendulum with reaction wheels. 


## Requirements
1. Python 3.8
2. Pytorch 1.5
3. CUDA 10.1

## Creation of conda environment and installation of packages

1. conda create -n pytorch_env python=3.8
2. conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
3. conda install -c anaconda pip
4. conda install -c anaconda numpy
5. conda install -c conda-forge matplotlib
6. conda install -c anaconda scipy
7. conda install -c anaconda scikit-learn
8. conda install pyyaml


## Model Training

2. Open the the directory **src/**
3. Set model hyperparameters in **src/config.yml**
4. Run the script for training:  **python main.py 0**
