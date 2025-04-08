import pandas as pd
'''
Since the GWAS file does not provide the corresponding variant base for rsid, we found the reference and variant base for rsid based on the dbsnp database,
The relevant scripts are saved in the dbsnp folder
Now read the found vcf file and filter file, add the corresponding bases to the CSV file, input. recode.vcf and _filter.csv, and output _ merge _ dbsnp. csv
The next step is to implement the filtering encoding area using R, and the result is _ merge _ bsnp _ ding _ filter. csv
Then use Generated_dataset.cy to generate the dataset and generate the H5 file using _marged_bsnp_comding_filter.csv
Then predict on each model that the next step is to calculate the g-value
'''
path = '/mnt/data0/users/lisg/Data/GWAS/Stroke'

csv_file = path + '_filter.csv'
vcf_file = path + '.recode.vcf'


df_csv = pd.read_csv(csv_file)

df_csv = df_csv.drop_duplicates(subset=['SNPS', 'CHR_POS', 'CHR_ID'], keep='first')
nan_rows = df_csv[df_csv['CHR_POS'].isna()]
nan_rows = df_csv[df_csv[['CHR_POS', 'CHR_ID']].isna().all(axis=1)]
df_csv = df_csv.dropna(subset=['CHR_POS', 'CHR_ID'], how='all')
'''===================================================================='''
vcf_data = [] 
vcf_columns = ['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO']  
with open(vcf_file, 'r') as vcf:
    for line in vcf:
        if line.startswith('#'):
            continue
        fields = line.strip().split('\t')
        if len(fields) >= 8: 
            vcf_data.append(fields[:8])
df_vcf = pd.DataFrame(vcf_data, columns=vcf_columns)

'''===================================================================='''

df_vcf = df_vcf[~df_vcf['CHROM'].str.startswith(('NW', 'NT'))]
filtered_df = df_vcf[~df_vcf['CHROM'].str.startswith(('NC', 'NW'))]
filtered_df['prefix'] = filtered_df['CHROM'].str[:2]
unique_prefixes = filtered_df['prefix'].unique()
df_vcf['CHROM'] = df_vcf['CHROM'].str.split('.').str[0]
df_vcf['CHROM'] = df_vcf['CHROM'].str[-2:]
df_vcf = df_vcf[~df_vcf['CHROM'].isin(['23', '24'])]
df_vcf = df_vcf.drop(columns=['QUAL', 'FILTER', 'INFO'])
df_vcf['CHROM'] = df_vcf['CHROM'].astype(str).str.lstrip('0')
print(df_vcf)
'''===================================================================='''
unique_chroms = df_vcf['CHROM'].unique()
'''===================================================================='''
duplicate_ids = df_vcf[df_vcf['ID'].duplicated(keep=False)] 
num_duplicates = len(duplicate_ids) 
'''===================================================================='''
merged_df = pd.merge(df_csv, df_vcf, left_on='rsid', right_on='ID', how='right')

columns_to_check = [col for col in merged_df.columns if col not in ['CHROM', 'POS', 'ID', 'REF', 'ALT']]

df_cleaned = merged_df.dropna(subset=columns_to_check, how='all')

print(merged_df)

merged_df['ALT_split'] = merged_df['ALT'].str.split(',')


merged_df = merged_df.explode('ALT_split')

merged_df['ALT'] = merged_df['ALT_split']
merged_df = merged_df.drop('ALT_split', axis=1)


print(merged_df)
merged_df.to_csv(path + '_merge_dbsnp.csv',index=False)