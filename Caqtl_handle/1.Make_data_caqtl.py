import os
import h5py
import numpy as np
from sklearn.model_selection import  train_test_split

seed = 10

np.random.seed(seed)

path = [
    '/mnt/data0/users/lisg/Data/brain/acc/caqtl/','/mnt/data0/users/lisg/Data/brain2cbl/caqtl/',
    '/mnt/data0/users/lisg/Data/brain/cmn/caqtl/','/mnt/data0/users/lisg/Data/brain/ic/caqtl/',
    '/mnt/data0/users/lisg/Data/brain/pn/caqtl/','/mnt/data0/users/lisg/Data/brain/pul/caqtl/',
    '/mnt/data0/users/lisg/Data/brain/sub/caqtl/',]

def split_data(pos_data,nega_data,path,seed):
    print(np.isnan(pos_data).any())
    print(np.isnan(nega_data).any())
    out_path = path
    if not os.path.exists(out_path): os.mkdir(out_path)  
    pos_label =  np.ones((pos_data.shape[0],1))
    nega_label = np.zeros((nega_data.shape[0],1))
    pos_data = np.hstack((pos_data, pos_label))
    nega_data = np.hstack((nega_data, nega_label))

    all_data = np.vstack((pos_data, nega_data))

    Data = all_data[:, :all_data.shape[1] - 1]  
    label = all_data[:, all_data.shape[1] - 1]  
   
    original_indices = np.arange(len(all_data))
    
    train_data, test_data, train_label, test_label, train_indices, test_indices = train_test_split(
        Data, label, original_indices, test_size=0.2, stratify=label, random_state=seed)
    np.savez(os.path.join(out_path, "data.npz"),
             data_train=train_data, data_test=test_data,
             label_train=train_label, label_test=test_label,
             test_indices_in_original=test_indices)


for filepath in path:
    positie_path = filepath + "positive_predict.h5"
    negative8_path = filepath + "negative0.008_predict.h5"

    with h5py.File(positie_path, 'r') as f:
        positive = f['product_data']
        positive = np.array(positive)
    with h5py.File(negative8_path, 'r') as f:
        negative8 = f['product_data']
        negative8 = np.array(negative8)
    nega8_path = os.path.join(filepath, "negative8")
    os.makedirs(nega8_path, exist_ok=True)
    split_data(pos_data=positive,nega_data=negative8,path=nega8_path,seed=seed)





