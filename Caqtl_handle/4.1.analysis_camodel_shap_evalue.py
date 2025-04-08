import os
import pickle
import h5py
import joblib
import numpy as np
import pandas as pd
import shap
import scanpy as sc
import random
np.random.seed(10)
random.seed(10)
import matplotlib.pyplot as plt
from utils import add_suffix_to_duplicates, remove_suffix, analyze_cell
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
plt.rcParams['pdf.fonttype'] = 42  
plt.rcParams['ps.fonttype'] = 42
'''======================================================================='''



root = [
    '/mnt/data0/users/lisg/Data/brain/acc/',
    '/mnt/data0/users/lisg/Data/brain/cmn/',
    '/mnt/data0/users/lisg/Data/brain/cbl/',
    '/mnt/data0/users/lisg/Data/brain/sub/',
    '/mnt/data0/users/lisg/Data/brain/ic/',
    '/mnt/data0/users/lisg/Data/brain/pn/',
    '/mnt/data0/users/lisg/Data/brain/pul/',
]
G = []
Shap = []
result = []
for path in root:

    with h5py.File(path + 'caqtl/positive_predict.h5', 'r') as f:
        data = np.array(f['product_data'])
        print(data.shape)
    ad = sc.read_h5ad(path + 'cluster_ad.h5ad')
    cell_names = ad.obs['celltype']
    cell_names = cell_names.reset_index(drop=True)
    cell_names = cell_names.apply(remove_suffix)  
    def remove_itl_numbers(name):
        if name.startswith('ITL'):
            return 'ITL'
        return name
    cell_names = cell_names.apply(remove_itl_numbers)
    '''========================================================================'''

    model = pickle.load(open(path + 'caqtl/negative8/best_model.pkl', 'rb'))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(data)
    shap_values.feature_names = cell_names
    
    datag = np.load(os.path.join(path,'caqtl/g_value.npy'))
    significance = joblib.load(os.path.join(path,'gaussian_significance_data.pkl'))
    max_density = np.array([significance[i]['max_density'] for i in range(datag.shape[1])])
    min_density = np.array([significance[i]['min_density'] for i in range(datag.shape[1])])
    thresholds_per10 = np.array([significance[i]['percentile_10'] for i in range(datag.shape[1])])

    datag = (datag - min_density) / (max_density - min_density)
    thresholds_per10 = (thresholds_per10 - min_density) / (max_density - min_density)
    '''======================================================================'''
    
    empty_row = []
    empty_g_10= 0
    result_G_value = np.full(datag.shape, np.nan)
    result_shap_value = np.full(datag.shape, np.nan)
    for i in range(datag.shape[0]):

        row = datag[i]
        cols10 = np.where(row <= thresholds_per10)[0]
        if cols10.size == 0:
            empty_g_10 += 1
            empty_row.append(i)
        result_G_value[i, cols10] = row[cols10]
        
        row1 = shap_values.values[i]
        indices_ca = np.where(row1 > 0)[0]
        result_shap_value[i, indices_ca] = row1[indices_ca]
    print(f"empty_g_10: {empty_g_10}")
    empty_row = np.array(empty_row)

    G_df = pd.DataFrame(result_G_value, columns=cell_names)
   

    final_G = G_df.T.groupby(G_df.columns).mean().T
    final_G = pd.DataFrame(data=final_G.values,columns=final_G.columns.tolist())

    shap_df = pd.DataFrame(result_shap_value, columns=cell_names)
    
   
    final_shap = shap_df.T.groupby(shap_df.columns).mean().T
    final_shap = pd.DataFrame(data=final_shap.values,columns=final_shap.columns.tolist())
    G.append(final_G)
    Shap.append(final_shap)
    new_df = (final_G.notna() & final_shap.notna()).astype(int)

    result.append(new_df)


type_list = ['ITL', 'ET', 'TP', 'CT', 'L6B', 'NP', 'AMY', 'ERC', 'PIR', 'SUB', 'ITV1C', 'CHO', 'CBGRC']

resultxxx = []

for df in result:
  
    row_stats = {
        col: df[col].sum() if col in df.columns else 0
        for col in type_list
    }
    resultxxx.append(row_stats)


final_stats = pd.DataFrame(resultxxx)


result_df= final_stats.rename(index={0: 'ACC', 1: 'CMN', 2: 'CBL', 3: 'SUB', 4: 'IC', 5: 'PN', 6: 'PUL'})





sns.set_style("whitegrid")
colors = ["#C5DFF4", "#92A5D1",]


cmap = LinearSegmentedColormap.from_list("custom_gradient", colors)

plt.figure(figsize=(10, 8)) 
heatmap = sns.heatmap(
    result_df,            
    annot=True,    
    cmap=cmap,   
    fmt='.2f',     
    cbar=True       
)

plt.title("")
plt.tight_layout()
plt.savefig('/mnt/data0/users/lisg/Data/brain/plot_results/fig4d/fig4b.pdf', format='pdf', bbox_inches='tight')

plt.show()