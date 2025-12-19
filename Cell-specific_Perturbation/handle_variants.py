import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import configargparse
import pyBigWig
from utils import extract_values, extract_values2, make_dataset


def make_parser():
    parser = configargparse.ArgParser(description="")
    parser.add_argument('--caqtl', type=str)
    parser.add_argument('--kgp_data', type=str, default='../PartII_data')
    parser.add_argument('--seed', default=10, type=int)
    parser.add_argument('--fa', type=str)
    parser.add_argument('--pha', type=str)
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()

    kgp_df = pd.read_csv(os.path.join(args.kgp_data, '1kgp_MAF>0.01.csv'))
    np.random.seed(args.seed)
    caqtl_df = pd.read_csv(args.caqtl)
    out_dir = os.path.join(args.kgp_data, 'CaQTL_data')
    os.makedirs(out_dir, exist_ok=True)

    caqtl_df = caqtl_df[caqtl_df['fdr'] != 'na']
    caqtl_df['df.test'] = caqtl_df['df.test'].astype(int)
    caqtl_df = caqtl_df[caqtl_df['df.test'] != 0]
    caqtl_df['fdr'] = caqtl_df['fdr'].astype(float)
    caqtl_df = caqtl_df[caqtl_df['fdr'] < 0.1]
    caqtl_df[['rsid_all', 'pos_all']] = caqtl_df['variant_id'].str.split('::', expand=True)
    caqtl_df = caqtl_df[~caqtl_df['variant_id'].str.match(r'^p2_')]
    duplicates = caqtl_df[caqtl_df.duplicated('pos_all', keep=False)]
    duplicates = duplicates.sort_values(by='pos_all')
    indexes_to_drop = duplicates.index
    caqtl_df = caqtl_df.drop(indexes_to_drop)
    caqtl_df = caqtl_df[~caqtl_df['variant_id'].str.match(r'^n1_')]
    caqtl_df = caqtl_df[~caqtl_df['variant_id'].str.match(r'^n2_')]
    caqtl_df = caqtl_df.reset_index(drop=True)
    caqtl_df[['rsid', 'ref', 'alt']] = caqtl_df.apply(extract_values, axis=1)
    caqtl_df[['chr', 'variant_pos']] = caqtl_df.apply(extract_values2, axis=1)

    bins = [0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    labels = ['(0-0.025]', '(0.025-0.05]', '(0.05-0.10]', '(0.10-0.15]', '(0.15-0.20]', '(0.20-0.25]',
              '(0.25-0.30]', '(0.30-0.35]', '(0.35-0.40]', '(0.40-0.45]', '(0.45-0.50]']

    kgp_df['maf_group'] = pd.cut(kgp_df['AF_adjust'], bins=bins, labels=labels, right=True)

    subset_ratio = [0.008]  # [1, 0.2, 0.008]
    max_distance = [150000]   # [2000, 8000, 150000]
    caqtl_df.to_csv(os.path.join(out_dir, 'positive.csv'), index=False)

    for ratio, max_distance in zip(subset_ratio, max_distance):
        subset_size = int(len(kgp_df) * ratio)
        subset_list = []
        for group in labels:
            group_df = kgp_df[kgp_df['maf_group'] == group]
            group_size = int(subset_size * (1 / len(labels)))

            if len(group_df) >= group_size:
                sampled_group = group_df.sample(n=group_size, replace=False, random_state=args.seed)
            else:
                sampled_group = pd.concat(
                    [group_df, group_df.sample(n=group_size - len(group_df), replace=True, random_state=args.seed)],
                    ignore_index=True)
            subset_list.append(sampled_group)

        subset = pd.concat(subset_list, ignore_index=True)

        if len(subset) > subset_size:
            subset = subset.sample(n=subset_size, replace=False, random_state=args.seed)
        elif len(subset) < subset_size:
            additional = kgp_df.sample(n=subset_size - len(subset), replace=False, random_state=args.seed)
            subset = pd.concat([subset, additional], ignore_index=True)
        subset = subset.sample(frac=1, random_state=args.seed).reset_index(drop=True)

        result_df = pd.DataFrame(columns=['Chr', 'pos', 'ref', 'alt', 'AF', 'AF_adjust', 'maf_group'])
        a, b, c, d = 0, 0, 0, 0
        for i in tqdm(range(caqtl_df.shape[0])):
            row = caqtl_df.iloc[i]
            chr_value = row['chr']
            pos_value = row['variant_pos']
            ref_value = row['ref']
            alt_value = row['alt']
            filtered_subset = subset[subset['Chr'] == chr_value]
            if not filtered_subset.empty:
                distances = (filtered_subset['pos'] - pos_value).abs()
                filtered_subset = filtered_subset.assign(distance=distances)
                nearest_subset = filtered_subset[filtered_subset['distance'] <= max_distance].sort_values(by='distance').head(1)
                if not nearest_subset.empty:
                    valid_nearest_B = None
                    for _, nearest_row in nearest_subset.iterrows():
                        if not (nearest_row['Chr'] == chr_value and nearest_row['pos'] == pos_value and
                                nearest_row['ref'] == ref_value and nearest_row['alt'] == alt_value):
                            valid_nearest_B = nearest_row
                            break
                    if valid_nearest_B is not None:
                        d = d + 1
                        new_row = pd.DataFrame(
                            [valid_nearest_B[['Chr', 'pos', 'ref', 'alt', 'AF', 'AF_adjust']]])
                    else:
                        a = a + 1
                        new_row = pd.DataFrame([{
                            'Chr': np.nan, 'pos': np.nan, 'ref': np.nan, 'alt': np.nan,
                            'AF': np.nan, 'AF_adjust': np.nan}])
                else:
                    b = b + 1
                    new_row = pd.DataFrame([{
                        'Chr': np.nan, 'pos': np.nan, 'ref': np.nan, 'alt': np.nan,
                        'AF': np.nan, 'AF_adjust': np.nan,}])
            else:
                c = c + 1
                new_row = pd.DataFrame([{
                    'Chr': np.nan, 'pos': np.nan, 'ref': np.nan, 'alt': np.nan,
                    'AF': np.nan, 'AF_adjust': np.nan}])
            # 将新行添加到结果DataFrame中
            result_df = pd.concat([result_df, new_row], ignore_index=True)
        result_df.to_csv(os.path.join(out_dir, 'negative.csv'), index=False)

    columns_to_keep = ['chr', 'variant_pos', 'ref', 'alt']
    caqtl_df = caqtl_df[columns_to_keep]
    caqtl_df = caqtl_df.rename(columns={'variant_pos': 'pos'})
    consbw = pyBigWig.open(args.pha)
    make_dataset(metrics=caqtl_df, fasta_file=args.fa, consbw=consbw, seqlen=2688,output_name='positive',output_path=out_dir)

    result_df = result_df.dropna(how='all')
    result_df = result_df.reset_index(drop=True)
    result_df = result_df.rename(columns={'Chr': 'chr'})
    make_dataset(metrics=result_df, fasta_file=args.fa, consbw=consbw, seqlen=2688,output_name='negative',output_path=out_dir)



if __name__ == "__main__":
    main()
