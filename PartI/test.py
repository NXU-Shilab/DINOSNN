import h5py
import numpy as np
train_data = h5py.File("/mnt/data0/users/lisg/Project_one/Brain/W_DINOSNN/processed_data/val_data.h5", 'r')
train_ph_X = train_data["val_ph_X"]
