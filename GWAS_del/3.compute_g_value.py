import os
import pickle
import h5py
import joblib
import numpy as np
from tqdm import tqdm

seed = 10
def load_bgmm(column_number,ecdfs_path):
    file_name = f'gmm_col_{column_number}.pkl'
    file_path = os.path.join(ecdfs_path, file_name)
    return joblib.load(file_path)

tissue_path = [
    '/mnt/data0/users/lisg/Data/brain/acc','/mnt/data0/users/lisg/Data/brain/cbl',
    '/mnt/data0/users/lisg/Data/brain/cmn','/mnt/data0/users/lisg/Data/brain/ic',
    '/mnt/data0/users/lisg/Data/brain/pn','/mnt/data0/users/lisg/Data/brain/pul',
    '/mnt/data0/users/lisg/Data/brain/sub',]

for tissue in tissue_path:
    data_path = [
        os.path.join(tissue, 'gwas/AD_predict.h5'),os.path.join(tissue, 'gwas/ADHD_predict.h5'),
        os.path.join(tissue, 'gwas/ALS_predict.h5'),os.path.join(tissue, 'gwas/ASD_predict.h5'),
        os.path.join(tissue, 'gwas/BD_predict.h5'),os.path.join(tissue, 'gwas/MDD_predict.h5'),
        os.path.join(tissue, 'gwas/MS_predict.h5'),os.path.join(tissue, 'gwas/PD_predict.h5'),
        os.path.join(tissue, 'gwas/SCZ_predict.h5'),os.path.join(tissue, 'gwas/Stroke_predict.h5'),
    ]
    for data in data_path:

        with h5py.File(data, 'r') as f:
            samples = np.array(f['product_data'])
        '''=========================================================================='''
        gmm_path = os.path.join(tissue, 'gaussian')
        datag = []
        for i in tqdm(range(samples.shape[1])):
            bgmm = load_bgmm(i, gmm_path)
            col_data = samples[:, i].reshape(-1, 1)
            density = np.exp(bgmm.score_samples(col_data))
            datag.append(density)
        datag = np.array(datag)
        datag = datag.transpose()
        print(datag.shape)
        file_name_without_extension = os.path.splitext(os.path.basename(data))[0]
        prefix = file_name_without_extension.split('_')[0]
        directory_path = os.path.dirname(data)
        np.save(os.path.join(directory_path,prefix+'_g_value.npy'), datag)
        '''=========================================================================='''























# datae = np.load('/mnt/data0/users/lisg/Data/brain2/sub/e_value.npy')
# print(datae.shape)

# threshold_low = 0.05
# threshold_high = 0.95

# 找出ECDF值接近0或1的列
# extreme_columns = []

# for i in range(datae.shape[0]):  # 对每个正样本
#     cols_below_threshold = np.where(datae[i, :] <= threshold_low)[0]  # 找出接近0的列
#     cols_above_threshold = np.where(datae[i, :] >= threshold_high)[0]  # 找出接近1的列
#     extreme_columns.append((cols_below_threshold, cols_above_threshold))  # 记录两种列

# 打印每个样本的小于0.05和大于0.95的列的位置
# for i, (low_cols, high_cols) in enumerate(extreme_columns):
    # print(f"Sample {i}: ECDF values <= {threshold_low} at columns {low_cols}")
    # print(f"Sample {i}: ECDF values >= {threshold_high} at columns {high_cols}")
    # print(low_cols.size)
    # print(high_cols.size)
    # print('=================')
    # 也可以打印出总的列数
    # print(f"Sample {i}: Total extreme columns = {low_cols.size + high_cols.size}")
