import pandas as pd
import scanpy as sc
import pickle

ad = sc.read_h5ad('/mnt/data0/users/lisg/Data/brain/ic/cluster_ad.h5ad')
peak_data = ad.var
peak_data = peak_data.reset_index(drop=True)

peak_data['center'] = (peak_data['start'] + peak_data['end']) / 2
peak_data['start_new'] = peak_data['center'] - 1344
peak_data['end_new'] = peak_data['center'] + 1344
with open('/mnt/data0/users/lisg/Data/brain/ic/test_id.pickle', 'rb') as file:
    data = pickle.load(file)
peak_data = peak_data.iloc[data]
peak_data = peak_data.reset_index(drop=True)






'''Process all non coding area components'''

file_path = "/mnt/data0/users/lisg/Data/public_data/gencode.v47.annotation.gff3"
nocoding = pd.read_csv(file_path, sep='\t', comment='#', header=None, usecols=[0, 2, 3, 4])
nocoding = nocoding[nocoding[2] != 'exon']
nocoding.columns = ['chr', 'type', 'start', 'end']
print(nocoding.head())

nocoding['start'] = nocoding['start'].astype(int)
nocoding['end'] = nocoding['end'].astype(int)
matching_rows = []

for idx, row in peak_data.iterrows():
    print(idx)
    
    chr_dfa, start_new_dfa, end_new_dfa = row['chr'], row['start_new'], row['end_new']

 
    for _, row_b in nocoding.iterrows():
        
        chr_b, start_b, end_b = row_b['chr'], row_b['start'], row_b['end']

      
        if chr_dfa == chr_b and start_new_dfa <= end_b and end_new_dfa >= start_b:
            matching_rows.append(idx)
            break  


print(matching_rows)

matching_df = pd.DataFrame(matching_rows, columns=['matching_row_indices'])


matching_df.to_csv('matching_rows2.csv', index=False)