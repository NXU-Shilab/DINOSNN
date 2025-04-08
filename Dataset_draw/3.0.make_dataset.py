import pickle

import anndata
import h5py
import numpy as np
from scipy import sparse
from utils import make_h5_sparse,split_train_test_val
import pandas as pd
import pyBigWig


#***********************************************************************************
#Just need to modify these code
output_path = '/mnt/data0/users/lisg/Data/denoise_data/BM0828/DINOSNN'
ad_path = output_path + "/cluster_ad.h5ad"
fasta_file = "/mnt/data0/users/lisg/Data/public_data/hg19.fa" #Genome file, buen2018 dataset requires hg19
consbw = pyBigWig.open("/mnt/data0/users/lisg/Data/public_data/hg19.100way.phastCons.bw")
seq_len=2688
seed = 10 
train_ratio=0.90 
#************************************************************************************



print('Start making all_deqs...')
ad_atac =anndata.read_h5ad(ad_path)
make_h5_sparse(atac_ad=ad_atac, h5_name='%s/processed_data.h5' % output_path, fasta_file=fasta_file,seq_len=seq_len)


def get_conservation_scores(chr, start, end):
    try:

        if start >= end:
            raise ValueError(f"Invalid interval: start ({start}) must be less than end ({end})")
     
        values = consbw.values(chr, start, end)
        if values is None:
            print(f"No data available for the specified region: {chr}:{start}-{end}")
            return np.full(end - start, np.nan)
        return values
    except ValueError as ve:
        print(f"ValueError: {ve}")
        return np.full(end - start, np.nan)
    except RuntimeError as re:
        print(f"RuntimeError: {re}")
        return np.full(end - start, np.nan)
    except Exception as e:
        print(f"Unexpected error: {e}")
        return np.full(end - start, np.nan)
def conservation_conv(values, a, b):
    values = np.array(values)
    if np.all(np.isnan(values)):  
        print("The output of consbw.values is all NaN values")
        values = np.nan_to_num(values)  
        a += 1
    else:
        nan_indices = np.where(np.isnan(values))[0] 
        average_score = np.nanmean(values)  
        values[nan_indices] = average_score   
        b += 1
    return np.exp(values), a, b
with h5py.File('%s/processed_data.h5' % output_path, 'a') as f:
    n_cells = ad_atac.shape[0]  # 2714
    m = ad_atac.X  # 2714 × 27150
    m = m.tocoo().transpose().tocsr()  # sparse matrix, rows as seqs, cols are cells
    seq_X = f['all_seqs']
    n_peaks = seq_X.shape[0]  
    X_dataset = []  
    Y_dataset = []  
    for i in range(n_peaks):
        x = seq_X[i]
        x_ohseq = sparse.coo_matrix((np.ones(seq_len), (np.arange(seq_len), x)), shape=(seq_len, 4),
                                    dtype=np.int8).toarray().transpose()

        y = m.indices[m.indptr[i]:m.indptr[i + 1]]
       
        y_ones = np.zeros(n_cells, dtype=np.int8)
 
        y_ones[y] = 1
       
        X_dataset.append(x_ohseq)
        Y_dataset.append(y_ones)
    f.create_dataset("data", data=X_dataset)
    f.create_dataset("laber", data=Y_dataset)

    # ************************************************************************************
    print("Start obtaining conservative ratings for the corresponding positions...")

    chromosomes = consbw.chroms()
    chromosomes = pd.DataFrame(list(chromosomes.items()), columns=['chr', 'length'])
    chrs = ['chr' + str(i) for i in range(1, 23)] + ['chrX', 'chrY']
    chromosomes = chromosomes[chromosomes['chr'].isin(chrs)]  
    bed_df = ad_atac.var.loc[:, ['chr', 'start', 'end']]
    bed_df.index = np.arange(bed_df.shape[0])  

    all_phastcons = np.zeros((bed_df.shape[0], seq_len), dtype=np.float32)
    a,b= 0,0
    for index, row in bed_df.iterrows():
        chr = row['chr']  
        start = row['start']
        end = row['end']  
        mid = (start + end) // 2 
        seq_start = mid - seq_len // 2  
        seq_end = seq_start + seq_len  
        seq_start = int(seq_start + 1)  
        seq_end = int(seq_end + 1)
        values = get_conservation_scores(chr,seq_start,seq_end)
        values,a,b = conservation_conv(values,a,b)
        all_phastcons[index] = values
        print('Write on line {}'.format(index))
    print("There are {} peaks in the total nan value".format(a))
    print("There are {} peaks containing non NaN values".format(b))

    print("make_data")
    X_dataset = f["data"]  
    x_phast = []
    for i in range(bed_df.shape[0]):
        phas = all_phastcons[i]
        data = X_dataset[i]
        result = data * phas
        x_phast.append(result)
    f.create_dataset('phast_data', data=x_phast, dtype=np.float32)

    print("Divide the dataset...")
   
    X_dataset = f["data"] 
    phast_X_dataset = f["phast_data"]  
    Y_dataset = f["laber"]
    n_peaks = X_dataset.shape[0]  # 27150
    n_cells = Y_dataset.shape[1]  # 2714
    X_dataset = np.array(X_dataset)
    phast_X_dataset = np.array(phast_X_dataset)
    Y_dataset = np.array(Y_dataset)
    phast_X_dataset = phast_X_dataset.astype(np.float32)
    train_ids, test_ids, val_ids = split_train_test_val(np.arange(n_peaks), train_ratio=train_ratio, seed=seed)
    print(train_ids.shape)
    print(test_ids.shape)
    print(val_ids.shape)
    print(test_ids)
    with open('%s/test_id.pickle' % output_path, 'wb') as file:
        pickle.dump(test_ids, file)

   
    train_X = X_dataset[train_ids] 
    train_Y = Y_dataset[train_ids] 
    train_ph = phast_X_dataset[train_ids]

    val_X = X_dataset[val_ids]
    val_Y = Y_dataset[val_ids]
    val_ph = phast_X_dataset[val_ids]

    test_X = X_dataset[test_ids]
    test_Y = Y_dataset[test_ids]
    test_ph = phast_X_dataset[test_ids]

    print("===========================")
    print("make h5file ...")
 
    with h5py.File('%s/train_data.h5' % output_path, 'w') as f:
       
        f.create_dataset('train_X', data=train_X)
        f.create_dataset('train_Y', data=train_Y)
        f.create_dataset('train_ph_X', data=train_ph, dtype="float32")
        f.close()
    with h5py.File('%s/val_data.h5' % output_path, 'w') as f:
        
        f.create_dataset('val_X', data=val_X)
        f.create_dataset('val_Y', data=val_Y)
        f.create_dataset('val_ph_X', data=val_ph, dtype="float32")
        f.close()
    with h5py.File('%s/test_data.h5' % output_path, 'w') as f:
        
        f.create_dataset('test_X', data=test_X)
        f.create_dataset('test_Y', data=test_Y)
        f.create_dataset('test_ph_X', data=test_ph, dtype="float32")
        f.close()




