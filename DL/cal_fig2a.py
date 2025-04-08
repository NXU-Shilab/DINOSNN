import os

import h5py
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from utils import cal_performance
root = [
    ['/mnt/data0/users/lisg/Data/brain/acc','/mnt/data0/users/lisg/scBasset/final_output/acc/y_pred.npy',"/mnt/data0/users/lisg/scBasset/final_output/pro_acc/m_test.npz"],
]
for data_path in root:
    print('Dataset',os.path.basename(data_path[0]))

    test_data = h5py.File('%s/test_data.h5' % data_path[0], 'r')
    true = test_data["test_Y"]
    num_cell=len(true[0])
    pred = np.load(os.path.join(data_path[0],'train_output/pred.npy'))
    print('cell count：',num_cell)


    y_pred = np.load(data_path[1])
    data = np.load(data_path[2])
    indices = data['indices']
    indptr = data['indptr']
    format = data['format']
    shape = tuple(data['shape'])
    sparse_data = data['data']

    sparse_matrix = csr_matrix((sparse_data, indices, indptr), shape=shape)

    y_true = sparse_matrix.toarray()
    y_true[y_true != 0] = 1

    peak_df,cell_df= cal_performance(D_true=true,D_pred=pred,s_true=y_true,s_pred=y_pred)


    peak_df.to_csv(os.path.join('/mnt/data0/users/lisg/Data/brain/result_load',os.path.basename(data_path[0])+'_peak.csv'), index=False)
    cell_df.to_csv(os.path.join('/mnt/data0/users/lisg/Data/brain/result_load', os.path.basename(data_path[0])+'_cell.csv'),index=False)
    print('===============================================================================')





