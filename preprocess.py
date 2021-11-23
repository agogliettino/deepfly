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

"""
Module of functions to preprocess data and write training and 
testing data to disk.
"""

# Constants.
FWD_SCALE = 100 # multiplier for the forward velocity.
Z_AXIS_IND = 23 # a z-axis to use
TRAIN_FRAC = .90
SPLIT_SEED = 2021

def write_train_test_split(data,outdir):
    """
    Writes the training and testing data split to disk.
    """
    
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
        
    np.random.seed(SPLIT_SEED)
    supervoxel_interp_data = data.supervoxel_interp_data
    n = supervoxel_interp_data.shape[0]
    inds_all = np.arange(0,n)
    train_inds = np.sort(np.random.choice(inds_all,int(np.floor(TRAIN_FRAC * n))))
    test_inds = np.setdiff1d(inds_all,train_inds)
    
    # Store the data into a dictionary and write to disk.
    train_data = dict()
    train_data['supervoxel_interp'] = supervoxel_interp[train_inds,:]
    train_data['Y'] = Y[train_inds,:]

    fnameout = os.path.join(outdir,"train_data_layer%s"%str(Z_AXIS_IND))
    np.save(fnameout,train_data,allow_pickle=True)

    # Same with the testing split.
    test_data = dict()
    test_data['supervoxel_interp'] = supervoxel_interp[test_inds,:]
    test_data['Y'] = Y[test_inds,:]

    fnameout = os.path.join(outdir,"test_data_layer%s"%str(Z_AXIS_IND))
    np.save(fnameout,test_data,allow_pickle=True)
    
    return None
    

def resample_data(data):
    """
    Resample the supervoxel data to correct for time offsets between layers and 
    downsample the behavioral data. Adds this to the data object.
    """
    # Grab the first and last time points from neural data.
    neural_timestamps = data.neural_timestamps
    first_tp = neural_timestamps[0,0] #from the top z layer
    last_tp = neural_timestamps[-1,0] #from the top z layer

    # Generate the same number of points to match neural data.
    timepoints_to_pull = np.linspace(first_tp, last_tp,
                                     neural_timestamps.shape[0])

    # Interpolate neural data to be time-aligned to the top z layer
    supervoxel_data = data.supervoxel_data
    supervoxel_data_interp = np.asarray([[np.interp(timepoints_to_pull, 
                                                neural_timestamps[:,z], 
                                               supervoxel_data[z,sv,:])
                                     for sv in range(supervoxel_data.shape[1])] 
                                     for z in range(int(supervoxel_data.shape[0]/2))])
    
    # Reshape the data using only one of the Z axes.
    n_z_slices,n_supervoxels,t = supervoxel_data_interp.shape
    supervoxel_data_interp = supervoxel_data_interp[Z_AXIS_IND,...].reshape(1,n_supervoxels,t)

    # Reshape the neural data so that all supervoxels are features.
    supervoxel_data_interp = np.transpose(np.reshape(supervoxel_data_interp,
                                    (supervoxel_data_interp.shape[0] * supervoxel_data_interp.shape[1],
                                     supervoxel_data_interp.shape[2])))

    # Downsample forward velocity.
    behavioral_timestamps = data.behavioral_timestamps
    fwd_data = data.fwd_data
    rot_data = data.rot_data
    behavior_interp_obj = interp1d(behavioral_timestamps, fwd_data, bounds_error = False)
    Y = behavior_interp_obj(timepoints_to_pull)
    Y = np.nan_to_num(Y)
    Y_fwd = Y[:,np.newaxis]
    
    # Now, downsample rotational velocity and stack them together in a matrix.
    rot_data = data.rot_data
    behavior_interp_obj = interp1d(behavioral_timestamps, rot_data, bounds_error = False)
    Y = behavior_interp_obj(timepoints_to_pull)
    Y = np.nan_to_num(Y)
    Y_rot = Y[:,np.newaxis]
    Y = np.c_[Y_fwd,Y_rot]
    
    # Pack all this into the object.
    data.Y = Y
    data.supervoxel_data_interp = supervoxel_data_interp
    
    return data


def get_data(supervoxel_datapath,
             neural_timestamps_path,
             fwd_data_path,rot_data_path,
             behavioral_timestamps_path):
    """
    Function to pack all the data into an object.
    """
    data = Data()
    data.supervoxel_data = get_supervoxel_data(supervoxel_datapath)
    data.neural_timestamps = get_supervoxel_data(supervoxel_datapath)
    data.fwd_data = get_fwd_data(fwd_data_path)
    data.rot_data = get_rot_data(rot_data_path)
    data.behavioral_timestamps = get_behavioral_timestamps(behavioral_timestamps)
    
    return data

class Data():
    """
    Class to create an object containing all the data needed. 
    Packs all the necessary stuff into a big object
    """
    
    def __init__(self):
        """
        Constructor, initializes empty object.
        """
        self.supervoxel_data = None
        self.neural_timestamps = None
        self.fwd_data = None
        self.rot_data = None
        self.behavioral_timestamps = None
        self.Y = None
        self.supervoxel_data_interp = None
        
def get_supervoxel_data(supervoxel_datapath):
    """
    Gets the supervoxel np.ndarray
    """
    return np.load(supervoxel_datapath)

def get_neural_timestamps(neural_timestamps_path):
    """
    Gets the timestamps for the neural data.
    """
    return np.load(neural_timestamps_path)

def get_fwd_data(fwd_data_path,clip=True):
    """
    Gets the forward velocity data.
    """

    fwd = np.load(fwd_path)

    if not clip:
        return fwd * FWD_SCALE

    return np.clip(fwd,0,None) * FWD_SCALE

def get_rot_data(rot_data_path):
    """
    Gets the rotational velocity data.
    """
    return np.load(rot_path)

def get_behavioral_timestamps(behavioral_timestamps_path):
    """
    Gets the timestamps for behavior.
    """
    return np.load(behavioral_timestamps_path)