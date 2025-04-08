import os.path

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics import roc_auc_score, average_precision_score
'''
root = [
    '/mnt/data0/users/lisg/Data/brain/acc/train_output',
    '/mnt/data0/users/lisg/Data/brain/cbl/train_output',
    '/mnt/data0/users/lisg/Data/brain/cmn/train_output',
    '/mnt/data0/users/lisg/Data/brain/ic/train_output',
    '/mnt/data0/users/lisg/Data/brain/pn/train_output',
    '/mnt/data0/users/lisg/Data/brain/pul/train_output',
    '/mnt/data0/users/lisg/Data/brain/sub/train_output',
    '/mnt/data0/users/lisg/Data/buen_2018/train_output',]

for path in root:
    peak_auc = np.load(path + '/peak_auc.npy')
    peak_aupr = np.load(path + '/peak_aupr.npy')
    cell_auc = np.load(path + '/cell_auc.npy')
    cell_aupr = np.load(path + '/cell_aupr.npy')

    print(path)
    average_auc = np.mean(peak_auc)
    average_aupr = np.mean(peak_aupr)
    average_cell_auc = np.mean(cell_auc)
    average_cell_aupr = np.mean(cell_aupr)


    print('peakauc',"{:.3f}".format(average_auc))
    print('peakaupr',"{:.3f}".format(average_aupr))
    print('cellauc',"{:.3f}".format(average_cell_auc))
    print('cellaupr',"{:.3f}".format(average_cell_aupr))
'''
root2 = [
    ['/mnt/data0/users/lisg/scBasset/pro_acc','/mnt/data0/users/lisg/scBasset/final_output/acc'],
    ['/mnt/data0/users/lisg/scBasset/pro_cbl','/mnt/data0/users/lisg/scBasset/final_output/cbl'],
    ['/mnt/data0/users/lisg/scBasset/pro_cmn','/mnt/data0/users/lisg/scBasset/final_output/cmn'],
    ['/mnt/data0/users/lisg/scBasset/pro_ic','/mnt/data0/users/lisg/scBasset/final_output/ic'],
    ['/mnt/data0/users/lisg/scBasset/pro_pn','/mnt/data0/users/lisg/scBasset/final_output/pn'],
    ['/mnt/data0/users/lisg/scBasset/pro_pul','/mnt/data0/users/lisg/scBasset/final_output/pul'],
    ['/mnt/data0/users/lisg/scBasset/pro_sub','/mnt/data0/users/lisg/scBasset/final_output/sub'],
]

for path in root2:
    print(path[0])

    y_pred = np.load(path[1] + '/y_pred.npy')
    data = np.load(os.path.join(path[0],'m_test.npz'))
    indices = data['indices']
    indptr = data['indptr']
    format = data['format']
    shape = tuple(data['shape'])
    sparse_data = data['data']

    sparse_matrix = csr_matrix((sparse_data, indices, indptr), shape=shape)

    y_true = sparse_matrix.toarray()
    y_true[y_true != 0] = 1

    aucvalue = []
    auprvalue = []
    for i in range(y_pred.shape[1]):
        skauc = roc_auc_score(y_true=y_true[:, i], y_score=y_pred[:, i])
        skaupr = average_precision_score(y_true[:, i], y_pred[:, i])
        aucvalue.append(skauc)
        auprvalue.append(skaupr)

    aucvalue = np.array(aucvalue)
    auprvalue = np.array(auprvalue)
    print('cellauc', np.format_float_positional(np.mean(aucvalue), precision=3))
    print('cellaupr', np.format_float_positional(np.mean(auprvalue), precision=3))

    peak_aucvalue = []
    peak_auprvalue = []
    for i in range(y_pred.shape[0]):
        skauc = roc_auc_score(y_true=y_true[i, :], y_score=y_pred[i, :])
        skaupr = average_precision_score(y_true[i, :], y_pred[i, :])
        peak_aucvalue.append(skauc)
        peak_auprvalue.append(skaupr)

    peak_aucvalue = np.array(peak_aucvalue)
    peak_auprvalue = np.array(peak_auprvalue)
    print('peakauc', np.format_float_positional(np.mean(peak_aucvalue), precision=3))
    print('peakaupr', np.format_float_positional(np.mean(peak_auprvalue), precision=3))
    print('=========================================================================================')



