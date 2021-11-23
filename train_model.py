import numpy as np
import scipy as sp
import torch
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas
import nibabel as nib
import sys
sys.path.insert(0,'../deepfly/')
import ft_parser as ft
from scipy.interpolate import interp1d
import sklearn
from sklearn.linear_model import LinearRegression
import torch
import pandas as pd
import os
import torch.nn as nn
import torch.optim as optim
import copy
import model

"""
Core model fitting file. Define the model, train model and write
results to disk.
"""

# Constants.
ALPHA = .01
N_EPOCHS = 100

class Dataset():
    
    """
    Dataset class for minibatch gradient descent.
    """
    def __init__(self, data_split):
        data = np.load(data_split,allow_pickle=True).item()
        self.supervoxel_interp = data['supervoxel_interp']
        self.Y = data['Y']

    def __getitem__(self, index):
        supervoxel = self.supervoxel_interp[index,:]
        y = self.Y[index,:]
        
        return supervoxel, y
          
    def __len__(self):
        return len(self.supervoxel_interp)
    
# Get the data and initialize the dataloader
train_dataset = Dataset('/data/minseung/deepfly/preprocessed/train_data_layer23.npy')
test_dataset = Dataset('/data/minseung/deepfly/preprocessed/test_data_layer23.npy')
BATCH_SIZE = train_dataset.supervoxel_interp.shape[0]
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,
                                         shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                         shuffle=False)


"""
Define the model, loss function and optimizer.
"""

# Simple 2 layer network.
model = nn.Sequential(
          nn.Linear(2000, 100),
          nn.ReLU(),
          nn.Linear(100,2)
        )

# Define the loss function as MSE loss
criterion = nn.MSELoss()

# Define the optimizer with a learning rate
optimizer = optim.Adam(model.parameters(), lr=ALPHA)

# Write the outputs.
log_dir = '/data/minseung/deepfly/training-out/layer23'

if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
    
log_file = open(os.path.join(log_dir,"log.txt"),'w')

cost_cache = []

for epoch in range(.N_EPOCHS): 
    
      for i, (supervoxel, y) in enumerate(train_loader, 0):
            
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward propogation
        outputs = model(supervoxel.float())
        
        # calculate the loss
        loss = criterion(outputs, y.float())
        
        # backpropogation + update parameters
        loss.backward()
        optimizer.step()

        # print statistics
        cost = loss.item()
        cost_cache.append(cost)
        if i % 100 == 0:    # print every 1000 iterations
            msg_out = 'Epoch:' + str(epoch) + ", Iteration: " + str(i) 
                  + ", training cost = " + str(cost)
            print(msg_out)
            
            # Write msg out to a file.
            log_file.write("%s\n"%msg_out)

# Close out the file after gradient descent and write everything to disk.
log_file.close()
fnameout = os.path.join(log_dir,"cost_cache.npy")
np.save(fnameout,cost_cache)

training_dict = dict()
training_dict['model'] = model
training_dict['optimizer'] = optimizer
training_dict['loss'] = loss
training_dict['cost'] = cost

fnameout = os.path.join(log_dir,"training_out.npy")
np.save(fnameout,training_dict,allow_pickle=True)


            