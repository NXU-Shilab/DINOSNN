import os
import h5py
import joblib
import numpy as np
from tqdm import tqdm
from utils import load_bgmm


path = '/mnt/data0/users/lisg/Data/brain/pul/'
data_path = os.path.join(path,'caqtl/positive_predict.h5')
with h5py.File(data_path, 'r') as f:
    positive = f['product_data']
    positive_samples = np.array(positive)
    '''=========================================================================='''
    bgmm_path = path + 'gaussian'
    datag = []
    for i in tqdm(range(positive_samples.shape[1])):
        bgmm = load_bgmm(i, bgmm_path)
        col_data = positive_samples[:, i].reshape(-1, 1)
        density = np.exp(bgmm.score_samples(col_data))
        datag.append(density)
    datag = np.array(datag)
    datag=datag.transpose()
    print(datag.shape)
    np.save(os.path.join(path,'caqtl/g_value.npy'), datag)