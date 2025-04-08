import pandas as pd
import re
import configargparse

'''
This script is designed to filter genomic data of thousands of people with maf less than 0.01
The input should already be the output result of running "/mnt/data0/public_data/1000Genomes/hg38/train. sh"
Input: "/mnt/data0/public_data/1000Genomes/hg38/filter_coding_1KGP.csv"
Output: 1kgp_maF>0.01.csv
'''

def make_parser():
    parser = configargparse.ArgParser(description="Preprocessing 1kgp data")
    parser.add_argument('--data_path', type=str,
                        help='data path,This path must contain filter_coding_1KGP.csv, which is also the output path.'
                             'The path tail needs to contain /  ')
    return parser

def main():
    parser = make_parser()
    args = parser.parse_args()
    path= args.data_path
    df = pd.read_csv('%sfilter_coding_1KGP.csv' % path)

    new_column_names = {
        'C': 'Chr',
        'D': 'pos',
        'A': 'ref',
        'B': 'alt',
        'H': 'info',
    }

    df.rename(columns=new_column_names, inplace=True)
    print(df.columns)

    def extract_af(info):
        match = re.search(r'AF=([^;]+)', info)
        if match:
            return match.group(1)
        return None


    df['AF'] = df['info'].apply(extract_af)

    df.drop(columns=['info'], inplace=True)
  
    df = df.dropna(subset=['AF'])  
    df['pos'] = df['pos'].astype(int)
    df['AF'] = df['AF'].astype(float)

    df['AF_adjust'] = df.apply(lambda row: 1 - row['AF'] if row['AF'] > 0.5 else row['AF'], axis=1)

    df = df[df['AF_adjust'] >= 0.01]
    print(df.shape)
    df.to_csv('%s1kgp_MAF>0.01.csv' % path, index=False)


if __name__ == "__main__":
    main()


