import gc
import anndata
import h5py as h5py
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import coo_matrix
def h5ad(donor,barcode):
    # Read COO matrix data
    idx = donor['PM']['idx'][:]
    idy = donor['PM']['idy'][:]
    count = donor['PM']['count'][:]
    chrm = donor['PM']['peakChrom']
    chrm = np.array(chrm)
    start = donor['PM']['peakStart']
    end = donor['PM']['peakEnd']
    shape = (np.max(idx), np.max(idy))
    coo = coo_matrix((count, (idx - 1, idy - 1)), shape=shape)
    matrix = coo.tocsr()
    ad = anndata.AnnData(matrix)
    ad.obs['barcode'] = barcode
    ad.var['chr'] = chrm
    ad.var['start'] = start
    ad.var['end'] = end
    ad.obs['barcode'] = ad.obs['barcode'].str.decode('utf-8').str.lstrip('b').str.strip("'")
    ad.var['chr'] = ad.var['chr'].astype(str).str.lstrip('b').str.strip("'")
    return ad
donor1path ="/mnt/data0/public_data/Brain_scATAC_science/pul/GSM7822202_MM_974.snap"
donor2path ="/mnt/data0/public_data/Brain_scATAC_science/pul/GSM7822203_MM_849.snap"
donor3path ="/mnt/data0/public_data/Brain_scATAC_science/pul/GSM7822204_MM_936.snap"
output_path = '/mnt/data0/public_data/Brain_scATAC_science/pul/'

cell = pd.read_csv("/mnt/data0/public_data/Brain_scATAC_science/1.Table S3 – Metatable and annotation of single nuclei.txt",sep='\t')
filter_cell = cell[cell['sample'].isin(['MM_974', 'MM_849', 'MM_936'])]
filter_cell = filter_cell.reset_index(drop=True)
donor1 = h5py.File(donor1path, 'r')
donor2 = h5py.File(donor2path, 'r')
donor3 = h5py.File(donor3path, 'r')
donor1barcode = donor1['BD']['name']
donor2barcode = donor2['BD']['name']
donor3barcode = donor3['BD']['name']
donor1barcode = np.array(donor1barcode)
donor2barcode = np.array(donor2barcode)
donor3barcode = np.array(donor3barcode)
d1ad = h5ad(donor1, donor1barcode)
d1ad.obs['sample'] = 'donor1'
d2ad = h5ad(donor2, donor2barcode)
d2ad.obs['sample'] = 'donor2'
d3ad = h5ad(donor3, donor3barcode)
d3ad.obs['sample'] = 'donor3'
ad_list = [d1ad, d2ad, d3ad]
ad = sc.concat(ad_list, axis=0)
ad.obs = ad.obs.reset_index(drop=True)
print(ad)
print(ad.obs)
'''======================================================='''
duplicated_barcodes = ad.obs['barcode'].duplicated(keep=False)
ad_filtered = ad[~duplicated_barcodes, :]
ad = ad_filtered
ad.obs = ad.obs.reset_index(drop=True)
ad.var['chr'] = d1ad.var['chr']
ad.var['start'] = d1ad.var['start']
ad.var['end'] = d1ad.var['end']
sc.pp.filter_cells(ad, min_genes=0)
sc.pp.filter_genes(ad, min_cells=0)
print(ad)


'''============================================================='''
filter_cell.loc[:, 'barcode'] = filter_cell['barcode'].astype(str)
filter_cell = filter_cell.reset_index(drop=True)
print('Known number of cells：',filter_cell.shape)
ad.obs = ad.obs.merge(filter_cell[['barcode', 'celltype']], on='barcode', how='left')
print(ad)
print(ad.obs)
ad.write(output_path + 'original_ad.h5ad')
ad = ad[~ad.obs['celltype'].isna()]
print(ad)
print(ad.obs)
ad.write(output_path + 'del_cell_nan.h5ad')