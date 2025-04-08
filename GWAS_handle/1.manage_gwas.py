import pandas as pd
'''
Filter duplicate samples from tsv files downloaded from the official website Generate _filter.csv file
'''
df = pd.read_csv('/mnt/data0/users/lisg/Data/GWAS/ALS_original.tsv', sep='\t')
rs_count = df[df['SNPS'].str.startswith('rs')].shape[0]
non_rs_count = df[~df['SNPS'].str.startswith('rs')].shape[0] 
df = df[df['SNPS'].str.startswith('rs')]
df.insert(0, 'rsid', df['SNPS'])
x_rows = df[df['rsid'].str.contains('x')] 
df= df[~df['rsid'].str.contains('x')] 
rows_with_semicolon = df[df['rsid'].str.contains(';', na=False)] 
if not rows_with_semicolon.empty:
    print(rows_with_semicolon.index.tolist())
else:
df = df.drop(rows_with_semicolon.index) 
print(df.shape)
df.to_csv('/mnt/data0/users/lisg/Data/GWAS/ALS_filter.csv', index=False)

