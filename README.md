## Deep_Fault_Tolerant_Control

Implementation of a deep fault tolerant control for the inverted pendulum with dual-axis reaction wheels. 

## Requirements
1. Python 3.8
2. Pytorch 1.5
3. CUDA 10.1

## Creation of conda environment and installation of packages

1. Create conda environment: ```conda create -n pytorch_env python=3.8```
2. Activate conda environment: ```conda activate pytorch_env```
3. Install Pytorch: ```conda install pytorch torchvision cudatoolkit=10.1 -c pytorch```
4. Install other packages: ```conda install numpy matplotlib scipy scikit-learn pyyaml pandas```

## Model Training

1. Set the model hyperparameters in ```src/config.yml```
2. Run the script for training:  ```python main.py 0```
3. Run the script for testing: ```python  simulation.py```
