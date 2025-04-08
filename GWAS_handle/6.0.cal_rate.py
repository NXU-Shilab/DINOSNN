import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams['pdf.fonttype'] = 42 
plt.rcParams['ps.fonttype'] = 42
ROOT = [
    '/mnt/data0/users/lisg/Data/brain/acc',
    '/mnt/data0/users/lisg/Data/brain/cmn',
    '/mnt/data0/users/lisg/Data/brain/cbl',
    '/mnt/data0/users/lisg/Data/brain/sub',
    '/mnt/data0/users/lisg/Data/brain/ic',
    '/mnt/data0/users/lisg/Data/brain/pn',
    '/mnt/data0/users/lisg/Data/brain/pul',
]
gwas_data_path = '/mnt/data0/users/lisg/Data/GWAS'


columns = ['AD', 'ADHD', 'ALS', 'ASD', 'BD', 'MDD', 'MS', 'PD', 'SCZ', 'Stroke']


data = []


for index, root_path in enumerate(ROOT):
    data_dir = [
        [os.path.join(root_path, 'gwas/AD_final.csv')],
        [os.path.join(root_path, 'gwas/ADHD_final.csv')],
        [os.path.join(root_path, 'gwas/ALS_final.csv')],
        [os.path.join(root_path, 'gwas/ASD_final.csv')],
        [os.path.join(root_path, 'gwas/BD_final.csv')],
        [os.path.join(root_path, 'gwas/MDD_final.csv')],
        [os.path.join(root_path, 'gwas/MS_final.csv')],
        [os.path.join(root_path, 'gwas/PD_final.csv')],
        [os.path.join(root_path, 'gwas/SCZ_final.csv')],
        [os.path.join(root_path, 'gwas/Stroke_final.csv')],
    ]
    a = []
    for data_path in data_dir:
        file_path = data_path[0]  
        df = pd.read_csv(file_path)
        filtered_rows = (df['caqtl_probability'] > 0.5) & (df['G_value'] > 0.5)
        count_greater_than_05 = filtered_rows.sum()
        total_rows = len(df)
        proportion = count_greater_than_05 / total_rows
        a.append(proportion)
    data.append(a) 


row_labels = [os.path.basename(root) for root in ROOT]
df_result = pd.DataFrame(data, index=row_labels, columns=columns)


colors = ['#F9EFEF', '#EAEFF6', '#98CADD', '#61AACF']  
cmap_name = 'custom_cmap'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=256) 
plt.register_cmap(cmap=cm) 


plt.figure(figsize=(12, 8))  
sns.heatmap(df_result,
            annot=True,  
            fmt='.2f',  
            cmap=cmap_name, 
            cbar_kws={'label': 'Proportion'}, 
            linewidths=0.5)  


plt.xlabel('Diseases', fontsize=13)
plt.ylabel('Brain Regions', fontsize=13)


plt.tight_layout()
plt.savefig('/mnt/data0/users/lisg/Data/brain/plot_results/fig5a.pdf', format='pdf', bbox_inches='tight')
plt.show()
