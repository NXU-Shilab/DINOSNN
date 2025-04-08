import pandas as pd
import sys
import os
import time
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import h5py
from torch import nn
import numpy as np
from tqdm import tqdm
from utils import Dataset
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
    return pred

data_path = '/mnt/data0/users/lisg/Data/brain/downsample_ic/dinosnn/rate80'
device_id =[4]
output_path = data_path + '/train_output'

use_cuda = torch.cuda.is_available() 
device = torch.device("cuda:%s"%device_id[0] if use_cuda else "cpu")
test_data = h5py.File('%s/test_data.h5' % data_path, 'r')
test_X = test_data["test_X"]
test_Y = test_data["test_Y"]
test_ph = test_data["test_ph_X"]
num_cell=len(test_Y[0])

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
pred = test_func(model=model,DataLoader=test_DataLoader,)

np.save(output_path + '/pred.npy', pred)



