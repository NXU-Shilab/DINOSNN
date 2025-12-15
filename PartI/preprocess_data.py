import pickle
import configargparse
from pathlib import Path
import anndata
import h5py
import numpy as np
from scipy import sparse
from utils import make_h5_sparse, split_train_test_val, get_conservation_scores, conservation_conv
import pandas as pd
import pyBigWig
def make_parser():
    parser = configargparse.ArgParser(description="Preprocessing dataset for DINOSNN")
    parser.add_argument('--ad', type=str, required=True)
    parser.add_argument('--output', type=Path, default='../processed_data')
    parser.add_argument('--fa', type=str,required=True)
    parser.add_argument('--pha', type=str, required=True)
    parser.add_argument('--seed', type=int, default=10)
    return parser

def main():
    parser = make_parser()
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output

    consbw = pyBigWig.open(args.pha)
    ad_atac = anndata.read_h5ad(args.ad)
    make_h5_sparse(atac_ad=ad_atac, h5_name='%s/processed_data.h5' % output_path, fasta_file=args.fa, seq_len=2688)
    print("Generate dataset...")

    with h5py.File('%s/processed_data.h5' % output_path, 'a') as f:
        n_cells = ad_atac.shape[0]
        m = ad_atac.X
        m = m.tocoo().transpose().tocsr()
        seq_X = f['all_seqs']
        n_peaks = seq_X.shape[0]
        X_dataset = []
        Y_dataset = []
        for i in range(n_peaks):
            x = seq_X[i]
            x_ohseq = sparse.coo_matrix((np.ones(2688), (np.arange(2688), x)), shape=(2688, 4),dtype=np.int8).toarray().transpose()
            y = m.indices[m.indptr[i]:m.indptr[i + 1]]
            y_ones = np.zeros(n_cells, dtype=np.int8)
            y_ones[y] = 1
            X_dataset.append(x_ohseq)
            Y_dataset.append(y_ones)
        f.create_dataset("data", data=X_dataset)
        f.create_dataset("laber", data=Y_dataset)

        bed_df = ad_atac.var.loc[:, ['chr', 'start', 'end']]
        bed_df.index = np.arange(bed_df.shape[0])
        all_phastcons = np.zeros((bed_df.shape[0], 2688), dtype=np.float32)
        a, b = 0, 0
        for index, row in bed_df.iterrows():
            chr = row['chr']
            start = row['start']
            end = row['end']
            mid = (start + end) // 2
            seq_start = mid - 2688 // 2
            seq_end = seq_start + 2688
            seq_start = int(seq_start + 1)
            seq_end = int(seq_end + 1)
            values = get_conservation_scores(chr, seq_start, seq_end,consbw)
            values, a, b = conservation_conv(values, a, b)
            all_phastcons[index] = values

        X_dataset = f["data"]
        x_phast = []
        for i in range(bed_df.shape[0]):
            phas = all_phastcons[i]
            data = X_dataset[i]
            result = data * phas
            x_phast.append(result)
        f.create_dataset('phast_data', data=x_phast, dtype=np.float32)
        phast_X_dataset = f["phast_data"]
        Y_dataset = f["laber"]
        n_peaks = X_dataset.shape[0]
        phast_X_dataset = np.array(phast_X_dataset)
        Y_dataset = np.array(Y_dataset)
        phast_X_dataset = phast_X_dataset.astype(np.float32)
        train_ids, test_ids, val_ids = split_train_test_val(np.arange(n_peaks), train_ratio=0.90, seed=args.seed)
        with open('%s/test_id.pickle' % output_path, 'wb') as file:
            pickle.dump(test_ids, file)
        train_Y = Y_dataset[train_ids]
        train_ph = phast_X_dataset[train_ids]
        val_Y = Y_dataset[val_ids]
        val_ph = phast_X_dataset[val_ids]
        test_Y = Y_dataset[test_ids]
        test_ph = phast_X_dataset[test_ids]
        print("make h5file ...")
        with h5py.File('%s/train_data.h5' % output_path, 'w') as f:
            f.create_dataset('train_ph_X', data=train_ph, dtype="float32")
            f.create_dataset('train_Y', data=train_Y)
            f.close()
        with h5py.File('%s/val_data.h5' % output_path, 'w') as f:
            f.create_dataset('val_ph_X', data=val_ph, dtype="float32")
            f.create_dataset('val_Y', data=val_Y)
            f.close()
        with h5py.File('%s/test_data.h5' % output_path, 'w') as f:
            f.create_dataset('test_Y', data=test_Y)
            f.create_dataset('test_ph_X', data=test_ph, dtype="float32")
            f.close()

if __name__ == "__main__":
    main()
