import os
import pickle
import random
import configargparse
import h5py
import pandas as pd
import pysam
import numpy as np
import pyBigWig
from scipy import sparse
from tqdm import tqdm
from utils import dna_1hot_2vec, trim_array
'''
The file is used to generate positive and negative datasets for 1 million randomly selected SNPs, EQTL, CAQTL, GWAS
Note that the generated dataset is an H5 file, which is used to directly input it into the best model for prediction

million_data: Input: "/mnt/data0/public_data/1000Genomes/hg38/random_million_1kgp.csv"
Output: "/mnt/data0/public_data/1000Genomes/hg38/random_million_1kgp.h5"

eqtl： Input:/ Mnt/data0/users/lisg/Data/emtx/emtsyacc/Output: Save the positive. h5 negative 0.2. h5 negative 1. h5 negative 0.008. h5 in the input file

caqtl: Input:/ Mnt/data0/users/lisg/Data/caqtl/Output: in the same directory


gwas: Input:/ Mnt/data0/users/lisg/Data/GWAS/AD/output: a. h5 file in the same directory

lentiMPRA:  Input:/ mnt/data0/users/lisg/Data/lentiMPRA/
'''


def check_array_length(array, expected_length):
    if isinstance(array, np.ndarray):

        if np.isnan(array).any():
            print("The array contains NaN values.")

        if array.shape[0] != expected_length:
            print(f"NumPy matrix has length {array.shape[0]}, which is not {expected_length}")
    elif isinstance(array, list):

        if any(np.isnan(x) for x in array):
            print("The array contains NaN values.")

        if len(array) != expected_length:
            print(f"Array has length {len(array)}, which is not {expected_length}")
def get_conservation_scores(chr, start, end,consbw):
    '''
    1base system
    '''
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
def conservation_conv(values):
    values = np.array(values)
    if np.all(np.isnan(values)):  
        values = np.nan_to_num(values)  
    else:
        nan_indices = np.where(np.isnan(values))[0]  
        average_score = np.nanmean(values)  
        values[nan_indices] = average_score  
    return np.exp(values)
def make_parser():
    parser = configargparse.ArgParser(description="Preprocessing eqtl data")
    parser.add_argument('--million_data', type=str, default=None,
                        help='Pass in the absolute path of the file, e.g.:')
    parser.add_argument('--eqtl', type=str, default=None,
                        help='Incoming folder path, e.g.: /mnt/data0/users/lisg/Data/eqtl/eqtl_acc/')
    parser.add_argument('--caqtl', type=str, default=None,
                        help='')
    parser.add_argument('--gwas', type=str, default=None,
                        help='')
    parser.add_argument('--lentiMPRA',type=str,default=None,
                        help='')
    parser.add_argument('--caqtl_exp', type=str, default=None,
                        help='')
    parser.add_argument('--fasta', type=str, default="/mnt/data0/users/lisg/Data/public_data/hg38.fa",
                        help='')
    parser.add_argument('--phastcons', type=str, default="/mnt/data0/users/lisg/Data/public_data/hg38.phastCons100way.bw",
                        help='')
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()

    seed = 10
    np.random.seed(seed)
    seqlen = 2688
    fasta_file = args.fasta
    consbw = pyBigWig.open(args.phastcons)

    if args.million_data is not None:
 
        metrics = pd.read_csv(args.million_data)
        metrics = metrics.drop(columns=['AF', 'AF_adjust'])
        metrics = metrics.rename(columns={'Chr': 'chr'})
        make_dataset(
            metrics,fasta_file,consbw,seqlen,output_name='random_million_1kgp',output_path=os.path.dirname(args.million_data) + '/')
    if args.eqtl is not None:

        columns_to_keep = ['chr', 'variant_pos', 'ref', 'alt']
        for filename in os.listdir(args.eqtl):
            if filename.endswith('.csv'):
                file_path = os.path.join(args.eqtl, filename)
                df = pd.read_csv(file_path)
                if list(df.columns) == ['variant_id', 'chr', 'variant_pos', 'ref', 'alt', 'rs_id_dbSNP151_GRCh38p7', 'maf','maf_group']:

                    df = df[columns_to_keep]
                    df = df.rename(columns={'variant_pos': 'pos'})
                    make_dataset(
                        metrics=df,fasta_file=fasta_file,consbw=consbw,seqlen=seqlen,
                        output_name=os.path.splitext(os.path.basename(file_path))[0],
                        output_path=args.eqtl)
                elif list(df.columns) == ['Chr','pos','ref','alt','AF','AF_adjust','maf_group',]:


                    df = df.dropna(how='all')
                    df = df.reset_index(drop=True)
                    df = df.rename(columns={'Chr': 'chr'})
                    make_dataset(
                        metrics=df, fasta_file=fasta_file, consbw=consbw, seqlen=seqlen,
                        output_name=os.path.splitext(os.path.basename(file_path))[0],
                        output_path=args.eqtl)
    if args.caqtl is not None:

        columns_to_keep = ['chr', 'variant_pos', 'ref', 'alt']
        for filename in os.listdir(args.caqtl):
            if filename.endswith('.csv'):
                file_path = os.path.join(args.caqtl, filename)
                df = pd.read_csv(file_path)

                required_columns = ['rsid_all', 'pos_all', 'rsid', 'ref', 'alt', 'chr', 'variant_pos']

                if all(column in df.columns for column in required_columns):
                    

                    df = df[columns_to_keep]
                    df = df.rename(columns={'variant_pos': 'pos'})
                    make_dataset(
                        metrics=df, fasta_file=fasta_file, consbw=consbw, seqlen=seqlen,
                        output_name=os.path.splitext(os.path.basename(file_path))[0],
                        output_path=args.caqtl)
                elif list(df.columns) == ['Chr', 'pos', 'ref', 'alt', 'AF', 'AF_adjust','maf_group', ]:  
                    
                    df = df.dropna(how='all')
                    df = df.reset_index(drop=True)
                    df = df.rename(columns={'Chr': 'chr'})
                    make_dataset(
                        metrics=df, fasta_file=fasta_file, consbw=consbw, seqlen=seqlen,
                        output_name=os.path.splitext(os.path.basename(file_path))[0],
                        output_path=args.caqtl)

    if args.gwas is not None:

        input_folder = args.gwas
        folder_name = os.path.basename(os.path.normpath(input_folder))

        file = os.path.join(input_folder, f"{folder_name}_merge_dbsnp_coding_filter.csv")

        gwas_df = pd.read_csv(file)

        gwas_df['POS'] = gwas_df['POS'].astype(int)
        gwas_df = gwas_df.rename(columns={'CHROM': 'chr', 'POS': 'pos', 'REF': 'ref','ALT': 'alt'})
        make_dataset(
            metrics=gwas_df, fasta_file=fasta_file, consbw=consbw, seqlen=seqlen,
            output_name=folder_name,
            output_path=args.gwas)

    if args.lentiMPRA is not None:

        input_folder = args.lentiMPRA
        folder_name = os.path.basename(os.path.normpath(input_folder))

        file = os.path.join(input_folder, 'merge_test.csv')
        print('读入文件：',file)
        lentiMPRA_df = pd.read_csv(file)

        lentiMPRA_df['pos'] = lentiMPRA_df['pos'].astype(int)

        lentiMPRA_df = lentiMPRA_df.rename(columns={'Chr': 'chr'})
        make_dataset(
            metrics=lentiMPRA_df, fasta_file=fasta_file, consbw=consbw, seqlen=seqlen,
            output_name=os.path.splitext(os.path.basename(file))[0],
            output_path=args.lentiMPRA)

    if args.caqtl_exp is not None:

        input_file = args.caqtl_exp

        caqtl_exp_df = pd.read_csv(input_file)
        caqtl_exp_df['pos'] = caqtl_exp_df['pos'].astype(int)

        make_dataset(
            metrics=caqtl_exp_df, fasta_file=fasta_file, consbw=consbw, seqlen=seqlen,
            output_name= os.path.splitext(os.path.basename(input_file))[0],
            output_path=os.path.dirname(input_file) + '/')



def make_dataset(metrics,fasta_file,consbw,seqlen,output_name=None,output_path=None):

    fasta_open = pysam.Fastafile(fasta_file)

    with h5py.File(os.path.join(output_path, output_name + '.h5'), "w") as f:
        ref_dset = f.create_dataset("ref", shape=(metrics.shape[0], 4, seqlen), dtype=np.float32)
        vary_dset = f.create_dataset("vary", shape=(metrics.shape[0], 4, 2688), dtype=np.float32)


        for i in tqdm(range(metrics.shape[0])):
            chrm = metrics['chr'][i]  
            pos = int(metrics['pos'][i]) - 1  
            if len(metrics['ref'][i]) == 1: 
                if len(metrics['alt'][i]) == 1:  

                    ref_l = fasta_open.fetch(chrm, int(pos - 1344), pos).upper()  
                    ref_r = fasta_open.fetch(chrm, pos + 1, pos + 1 + 1343).upper()  
                    refseq_dna = ref_l + metrics['ref'][i] + ref_r 
                    varseq_dna = ref_l + metrics['alt'][i] + ref_r  

                    values_l = get_conservation_scores(chrm, int(pos + 1 - 1344), int(pos + 1), consbw=consbw)
                    value = get_conservation_scores(chrm, int(pos + 1), int(pos + 2), consbw=consbw)
                    values_r = get_conservation_scores(chrm, int(pos + 2), int(pos + 2 + 1343), consbw=consbw)

                    values_l = conservation_conv(values_l)
                    value = conservation_conv(value)
                    values_r = conservation_conv(values_r)
                    cons_values = np.concatenate((values_l, value, values_r))  
                else:
                    ref_l = fasta_open.fetch(chrm, int(pos - 1344), pos).upper()  
                    ref_r = fasta_open.fetch(chrm, pos + 1, pos + 1 + 1343).upper()
                    refseq_dna = ref_l + metrics['ref'][i] + ref_r  
                    vary_length = len(metrics['alt'][i])
                    varseq_dna = ref_l + metrics['alt'][i] + ref_r  
                    varseq_dna = trim_array(varseq_dna, vary_length - 1)

                    values_l = get_conservation_scores(chrm, int(pos + 1 - 1344), int(pos + 1),consbw=consbw)
                    value = get_conservation_scores(chrm, int(pos + 1), int(pos + 2),consbw=consbw)
                    value = [value[0]] * vary_length
                    values_r = get_conservation_scores(chrm, int(pos + 2), int(pos + 2 + 1343),consbw=consbw)
                    values_l = conservation_conv(values_l)
                    value = conservation_conv(value)
                    values_r = conservation_conv(values_r)
                    cons_values = np.concatenate((values_l, value, values_r))   
                    cons_values = trim_array(cons_values, vary_length - 1)
            else: 
                ref_length = len(metrics['ref'][i])
                refstart = int(pos - 1344) 
                refend = int(pos + ref_length + 1343)  

                ref_l = fasta_open.fetch(chrm, refstart, pos).upper()  
                ref_r = fasta_open.fetch(chrm, pos + ref_length, refend).upper()
                refseq_dna = ref_l + metrics['ref'][i] + ref_r  
                refseq_dna = trim_array(refseq_dna, ref_length - 1)
                varseq_dna = ref_l + metrics['alt'][i] + ref_r  
                vary_length = len(metrics['alt'][i])
                varseq_dna = trim_array(varseq_dna, vary_length - 1)
                values_l = get_conservation_scores(chrm, int(refstart + 1), int(pos + 1),consbw=consbw)
                value = get_conservation_scores(chrm, int(pos + 1), int(pos + 2),consbw=consbw)
                value = [value[0]] * ref_length
                values_r = get_conservation_scores(chrm, int(pos + 1 + ref_length), int(refend + 1),consbw=consbw)
                values_l = conservation_conv(values_l)
                value = conservation_conv(value)
                values_r = conservation_conv(values_r)
                cons_values = np.concatenate((values_l, value, values_r))
                cons_values = trim_array(cons_values, ref_length - 1)

            check_array_length(refseq_dna, seqlen)
            check_array_length(varseq_dna, seqlen)
            check_array_length(cons_values, seqlen)

            ref_dna_dense = dna_1hot_2vec(refseq_dna)
            vary_dna_dense = dna_1hot_2vec(varseq_dna)
            ref = sparse.coo_matrix((np.ones(seqlen), (np.arange(seqlen), ref_dna_dense)), shape=(seqlen, 4),
                                    dtype='int8')
            vary = sparse.coo_matrix((np.ones(seqlen), (np.arange(seqlen), vary_dna_dense)), shape=(seqlen, 4),
                                     dtype='int8')

            ref_dset[i] = (ref.toarray().transpose() * cons_values).astype(np.float32)
            vary_dset[i] = (vary.toarray().transpose() * cons_values).astype(np.float32)









if __name__ == "__main__":
    main()
