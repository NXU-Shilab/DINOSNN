import gc
import anndata
import h5py as h5py
import numpy as np
import pandas as pd
import scanpy as sc
import snapatac2 as snap
import scipy.sparse as sp
ad = sc.read_h5ad("/mnt/data0/public_data/Brain_scATAC_science/pul/del_cell_nan.h5ad")
output_path = '/mnt/data0/users/lisg/Data/brain/pul'
print(ad)
cell_num, peak_num = ad.shape[0], ad.shape[1]
thres = int(cell_num * 0.005) 
ad = ad[:, ad.var['n_cells'] > thres]
thres_row = int(peak_num * 0.002) 
ad = ad[ad.obs['n_genes'] > thres_row, :]

ad.write(output_path + '/no_cluster_ad.h5ad')

data = snap.read(output_path + '/no_cluster_ad.h5ad')
snap.pp.select_features(data, n_features=5000000)
snap.tl.spectral(data)
snap.tl.umap(data)
snap.pp.harmony(data, batch="sample", max_iter_harmony=20)
snap.tl.umap(data, use_rep="X_spectral_harmony")
snap.pp.knn(data, use_rep="X_spectral_harmony")
snap.tl.leiden(data)
data.close()

data  =sc.read_h5ad(output_path + '/no_cluster_ad.h5ad')
leiden_label=data.obs['leiden']
leiden_counts = pd.Series(leiden_label).value_counts()
print(leiden_counts)

unique_labels = np.unique(data.obs['leiden'])
sorted_indices = np.argsort(unique_labels.astype(int))
unique_labels = unique_labels[sorted_indices]
new_cell_order = []
for label in unique_labels:
    new_cell_order.extend(np.where(data.obs['leiden'] == label)[0])
new_ad = data[new_cell_order, :]
print(new_ad)
new_ad.write(output_path + '/cluster_ad.h5ad')


'''-----------------------------------handle buen2018--------------------------------------------'''
# output_path = '/mnt/data0/users/lisg/Data/buen_2018'
#
# # data = snap.read('/mnt/data0/users/lisg/Data/buen_2018/buen_ad_sc.h5ad')
# # snap.pp.select_features(data, n_features=5000000)
# # snap.tl.spectral(data)
# # snap.tl.umap(data)
# # snap.pp.knn(data)
# # snap.tl.leiden(data)
# # data.close()
#
# data  =sc.read_h5ad(output_path + '/buen_ad_sc.h5ad')
#
# var_df = data.var
#
# var_df['chr'] = var_df['chr'].astype(int) + 1
# var_df.loc[var_df['chr'] == 23, 'chr'] = 'X'
# var_df.loc[var_df['chr'] == 24, 'chr'] = 'Y'
# var_df['chr'] = 'chr' + var_df['chr'].astype(str)
#
# data.var = var_df
# leiden_label=data.obs['leiden']
# leiden_counts = pd.Series(leiden_label).value_counts()
#
# unique_labels = np.unique(data.obs['leiden'])

# sorted_indices = np.argsort(unique_labels.astype(int))

# unique_labels = unique_labels[sorted_indices]
# new_cell_order = []
# for label in unique_labels:
#     new_cell_order.extend(np.where(data.obs['leiden'] == label)[0])
# new_ad = data[new_cell_order, :]
# print(new_ad)
# new_ad.write(output_path + '/cluster_ad.h5ad')


