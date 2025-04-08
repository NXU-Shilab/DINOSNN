import pandas as pd
import sys
import os
import time

from scipy import sparse
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import h5py
from torch import nn
import numpy as np
from tqdm import tqdm
from utils import Dataset
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from scipy.stats import wilcoxon
directory = os.path.dirname(os.path.abspath(__file__))
def test_func(model,DataLoader,):
    model.eval()
    true,pred = [],[]
    cell_auc,peak_auc,cell_aupr,peak_aupr = [],[],[],[]
    with torch.no_grad():
        val_bar = tqdm(DataLoader, file=sys.stdout)
        for val_id, (x, y) in enumerate(val_bar):
            sig = nn.Sigmoid()
            x = x.to(device, non_blocking=True, dtype=torch.float32)
            y = y.to(device, non_blocking=True, dtype=torch.float32)
            output = model(x)
            new_out = sig(output)
            true.append(y.detach().cpu())
            pred.append(new_out.detach().cpu())
        true = torch.cat(true, 0).numpy()
        pred = torch.cat(pred, 0).numpy()
        for i in range(true.shape[0]):
            peak_auc.append(roc_auc_score(y_true=true[i, :], y_score=pred[i, :]))
            peak_aupr.append(average_precision_score(y_true=true[i, :], y_score=pred[i, :]))
        for i in range(true.shape[1]):
            cell_auc.append(roc_auc_score(y_true=true[:,i], y_score=pred[:,i]))
            cell_aupr.append(average_precision_score(y_true=true[:, i], y_score=pred[:, i]))

 
        print('auROC per peak:', sum(peak_auc) / len(peak_auc))
        print('auROC per cell:', sum(cell_auc) / len(cell_auc))
        print('auPR per peak:', sum(peak_aupr) / len(peak_aupr))
        print('auPR per cell:', sum(cell_aupr) / len(cell_aupr))
    return pred,peak_auc

data_path = '/mnt/data0/users/lisg/Data/brain/ic'
device_id =[6]
output_path = data_path + '/train_output'

use_cuda = torch.cuda.is_available() 
device = torch.device("cuda:%s"%device_id[0] if use_cuda else "cpu")
test_data = h5py.File('%s/test_data.h5' % data_path, 'r')
test_Y = test_data["test_Y"]
test_ph = test_data["test_ph_X"]
num_cell=len(test_Y[0])



file_path = "/mnt/data0/users/lisg/Project_one/Brain/dataset_draw/matching_rows2.csv"
nocoding = pd.read_csv(file_path)
row_indices = nocoding['matching_row_indices'].to_numpy()
test_ph = test_ph[row_indices]
test_Y = test_Y[row_indices]
print(test_ph.shape)
print(test_Y.shape)
test_set = Dataset(test_ph, test_Y) 
test_DataLoader = torch.utils.data.DataLoader(test_set, batch_size=1,shuffle=False)

checkpoint = torch.load('%s/best_val_auc_model.pth' % output_path,map_location=device) 
print('epochs:',checkpoint['epoch'])
print('val_auc:',checkpoint['val_auc'])

from Model_smt import model
model = model(num_cell=num_cell)
model = nn.DataParallel(model,device_ids=device_id)
model.to(device)

model.load_state_dict(checkpoint['best_model_state']) 
pred,peak_auc = test_func(model=model,DataLoader=test_DataLoader,)
#
np.save(output_path + '/pred_nocoding.npy', pred)























# test_data = '/mnt/data0/users/lisg/scBasset/final_output/pro_ic/test_seqs.h5'
# test_data = h5py.File(test_data, 'r')
# test_ph = test_data['X']
# m_test = sparse.load_npz('/mnt/data0/users/lisg/scBasset/final_output/pro_ic/m_test.npz')
# # m_test = m_test.todense()
# print(m_test)
# file_path = "/mnt/data0/users/lisg/Project_one/Brain/dataset_draw/matching_rows2.csv"
# nocoding = pd.read_csv(file_path)
# row_indices = nocoding['matching_row_indices'].to_numpy()
# test_ph = test_ph[row_indices]
# m_test = m_test[row_indices]
# sparse.save_npz('/mnt/data0/users/lisg/scBasset/final_output/pro_ic/m_test_nocoding.npz', m_test, compressed=False)
# # 保存到新的H5文件
# output_file = '/mnt/data0/users/lisg/scBasset/final_output/pro_ic/test_ph_nocoding.h5'  # 新文件路径
# with h5py.File(output_file, 'w') as f:
#     f.create_dataset('X', data=test_ph)  # 保存为键名 'X'，与原文件保持一致
# # 关闭原始文件
# test_data.close()