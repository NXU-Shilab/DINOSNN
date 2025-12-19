import os.path
import os
import configargparse
import pandas as pd
import numpy as np
import pyBigWig
from utils import extract_af, make_dataset

def make_parser():
    parser = configargparse.ArgParser(description="")
    parser.add_argument('--data_path',default='../PartII_data')
    parser.add_argument('--seed',default=10,type=int)
    parser.add_argument('--fa', type=str)
    parser.add_argument('--pha', type=str)
    return parser

def main():
    parser = make_parser()
    args = parser.parse_args()

    consbw = pyBigWig.open(args.pha)
    path= args.data_path
    seed = args.seed
    np.random.seed(seed)
    df = pd.read_csv('%sfilter_coding_1KGP.csv' % path)
    new_column_names = {
        'C': 'Chr',
        'D': 'pos',
        'A': 'ref',
        'B': 'alt',
        'H': 'info',
    }
    df.rename(columns=new_column_names, inplace=True)
    df['AF'] = df['info'].apply(extract_af)
    df.drop(columns=['info'], inplace=True)
    df = df.dropna(subset=['AF'])
    df['pos'] = df['pos'].astype(int)
    df['AF'] = df['AF'].astype(float)
    df['AF_adjust'] = df.apply(lambda row: 1 - row['AF'] if row['AF'] > 0.5 else row['AF'], axis=1)
    df = df[df['AF_adjust'] >= 0.01]
    df.to_csv(os.path.join(path,'1kgp_MAF>0.01.csv'), index=False)

    sampled_df = df.sample(n=1000000, random_state=seed)
    sampled_df.to_csv(os.path.join(path,'random_million_1kgp.csv'), index=False)

    metrics = sampled_df
    metrics = metrics.drop(columns=['AF', 'AF_adjust'])
    metrics = metrics.rename(columns={'Chr': 'chr'})
    make_dataset(metrics, args.fa, consbw, 2688, output_name='random_million_1kgp',output_path=path)


if __name__ == "__main__":
    main()


