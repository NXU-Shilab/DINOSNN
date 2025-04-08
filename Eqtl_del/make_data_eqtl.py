import os
import h5py
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

seed = 10

np.random.seed(seed)

path = [
    '/mnt/data0/users/lisg/Data/brain/acc/eqtl_acc/', #acc
    # '/mnt/data0/users/lisg/Data/brain2/sub/eqtl_sub/',  #sub
    # '/mnt/data0/users/lisg/Data/brain2/cbl/eqtl_cbl/',  #cbl
    # '/mnt/data0/users/lisg/Data/brain2/cmn/eqtl_cmn/',  #cmn
]

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

    original_indices = np.arange(len(all_data))
    Data = all_data[:, :all_data.shape[1] - 1] 
    label = all_data[:, all_data.shape[1] - 1]  
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    
    for fold, (train_indices, test_indices) in enumerate(skf.split(Data, label)):
        data_train, data_test = Data[train_indices], Data[test_indices]
        label_train, label_test = label[train_indices], label[test_indices]

        test_indices_in_original = original_indices[test_indices]
    
        label_test = label_test[:, np.newaxis]
        test_data = np.hstack((data_test, label_test))
        valid, test, indices_valid, indices_test = train_test_split(test_data, test_indices_in_original, test_size=0.5, random_state=seed)
    
        data_valid,label_valid = valid[:,:valid.shape[1]-1],valid[:,valid.shape[1]-1]
        test_X,test_y = test[:,:test.shape[1]-1],test[:,test.shape[1]-1]
    
        print(f"Fold {fold + 1}")
        fold_directory = os.path.join(out_path, f"Fold_{fold + 1}")
        os.makedirs(fold_directory, exist_ok=True)
        np.savez(os.path.join(fold_directory, "data.npz"),
                 data_train=data_train,data_test=test_X,data_valid = data_valid,
                 label_train=label_train,label_test=test_y,label_valid = label_valid,
                 test_indices_in_original=indices_test)


for filepath in path:
    positive_path = filepath + "positive_predict.h5"
    negative8_path = filepath + "negative0.008_predict.h5"
    negative2_path = filepath + "negative0.2_predict.h5"
    negative1_path = filepath + "negative1_predict.h5"

    with h5py.File(positive_path, 'r') as f:
        positive = f['product_data']
        positive = np.array(positive)
    '''========================================================='''
    with h5py.File(negative8_path, 'r') as f:
        negative8 = f['product_data']
        negative8 = np.array(negative8)
    nega8_path = os.path.join(filepath, "negative8")
    os.makedirs(nega8_path, exist_ok=True)
    split_data(pos_data=positive,nega_data=negative8,path=nega8_path,seed=seed)
    # '''========================================================='''
    with h5py.File(negative2_path, 'r') as f:
        negative2 = f['product_data']
        negative2 = np.array(negative2)
    nega2_path = os.path.join(filepath, "negative2")
    os.makedirs(nega2_path, exist_ok=True)
    split_data(pos_data=positive, nega_data=negative2, path=nega2_path, seed=seed)
    '''========================================================='''
    with h5py.File(negative1_path, 'r') as f:
        negative1 = f['product_data']
        negative1 = np.array(negative1)
    nega1_path = os.path.join(filepath, "negative1")
    os.makedirs(nega1_path, exist_ok=True)
    split_data(pos_data=positive, nega_data=negative1, path=nega1_path, seed=seed)
