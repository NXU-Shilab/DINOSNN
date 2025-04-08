import os
import pickle
import h5py
import joblib
import numpy as np
import pandas as pd
import shap
import scanpy as sc
import random

from matplotlib.colors import LinearSegmentedColormap

np.random.seed(10)
random.seed(10)
from utils import  remove_suffix
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42 
plt.rcParams['ps.fonttype'] = 42
def datajiazai(path):
    ad = sc.read_h5ad(path + 'cluster_ad.h5ad')
    cell_names = ad.obs['celltype']
    cell_names = cell_names.reset_index(drop=True)
    cell_names = cell_names.apply(remove_suffix)  
    return cell_names

def shapfenxi(data_path,model,cell_names,filtered_indices):
    with h5py.File(data_path, 'r') as f:
        data = np.array(f['product_data'])
    data = np.delete(data, filtered_indices, axis=0)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(data)
    shap_values.feature_names = cell_names
    result_shap_value = np.full(data.shape, np.nan)
    for i in range(data.shape[0]):
  
        row1 = shap_values.values[i]
        indices_ca = np.where(row1 > 0)[0]
        result_shap_value[i, indices_ca] = row1[indices_ca]

    shap_df = pd.DataFrame(result_shap_value, columns=cell_names) 
    final_shap = shap_df.T.groupby(shap_df.columns).mean().T
    final_shap = pd.DataFrame(data=final_shap.values, columns=final_shap.columns.tolist())
    final_shap = final_shap.fillna(0)
    return final_shap

def gjiazai(path2,significance):
    datag = np.load(path2)
    max_density = np.array([significance[i]['max_density'] for i in range(datag.shape[1])])
    min_density = np.array([significance[i]['min_density'] for i in range(datag.shape[1])])
    thresholds_per5 = np.array([significance[i]['percentile_5'] for i in range(datag.shape[1])])
    thresholds_per10 = np.array([significance[i]['percentile_10'] for i in range(datag.shape[1])])
    thresholds_per15 = np.array([significance[i]['percentile_15'] for i in range(datag.shape[1])])

    datag = (datag - min_density) / (max_density - min_density)
    thresholds_per5 = (thresholds_per5 - min_density) / (max_density - min_density)
    thresholds_per10 = (thresholds_per10 - min_density) / (max_density - min_density)
    thresholds_per15 = (thresholds_per15 - min_density) / (max_density - min_density)
    return datag,thresholds_per10,thresholds_per15,thresholds_per5

def process_single_sample(a, b):

    cell_types = a.columns.unique()


    sample_means = {}
    threshold_means = {}


    for ct in cell_types:

        cols = a.columns[a.columns == ct].tolist()

        sample_values = a[cols].values.flatten() 
        th_values = b[cols].values.flatten()

        mask = sample_values < th_values
        significant_values = sample_values[mask]
        significant_th = th_values[mask]

        if len(significant_values) > 0:
            sample_means[ct] = np.mean(significant_values)
            threshold_means[ct] = np.mean(significant_th)
        else:
            sample_means[ct] = np.mean(sample_values)
            threshold_means[ct] = np.mean(th_values)

    means_df = pd.DataFrame([sample_means], index=a.index)
    thresh_df = pd.DataFrame([threshold_means], index=a.index)

    return means_df, thresh_df

def process_cell_types(df_a, df_b):

    unique_cell_types = set(col.split('_')[0] if '_' in col else col for col in df_a.columns)

    result_mean = pd.DataFrame(index=df_a.index)
    result_threshold = pd.DataFrame(index=[0])

    for cell_type in unique_cell_types:

        cell_cols = [col for col in df_a.columns if col.startswith(cell_type)]

        cell_data = df_a[cell_cols]
        cell_thresholds = df_b[cell_cols].iloc[0] 

        significant_mask = cell_data < cell_thresholds
        significant_cols = significant_mask.any(axis=0)  

        if significant_cols.any():  

            sig_data = cell_data.loc[:, significant_cols]
            sig_thresholds = cell_thresholds[significant_cols]

            mean_values = sig_data.mean(axis=1)
            mean_threshold = sig_thresholds.mean()
        else:

            mean_values = cell_data.mean(axis=1)
            mean_threshold = cell_thresholds.mean()

        result_mean[cell_type] = mean_values
        result_threshold[cell_type] = mean_threshold

    return result_mean, result_threshold

def merge_neuron_types(df):

    itl_columns = [col for col in df.columns if col.startswith('ITL')]

    if itl_columns:

        df['ITL'] = df[itl_columns].mean(axis=1)

        df = df.drop(columns=itl_columns)

    itl_columns2 = [col for col in df.columns if col.startswith('LAMP5')]

    if itl_columns2:
        df['LAMP5'] = df[itl_columns2].mean(axis=1)
        df = df.drop(columns=itl_columns2)

    prerc_columns = [col for col in df.columns if col in ['PIR', 'TP', 'ERC']]

    if 'PRERC' in df.columns:
     
        prerc_columns.append('PRERC')
 
        df['PRERC'] = df[prerc_columns].mean(axis=1)

        df = df.drop(columns=[col for col in prerc_columns if col != 'PRERC'])
    elif prerc_columns:

        df['PRERC'] = df[prerc_columns].mean(axis=1)

        df = df.drop(columns=prerc_columns)
    return df


def old_analysis(datag,thresholds,cell_names):
    result_G_value = np.full(datag.shape, np.nan)
    for i in range(datag.shape[0]):

        row = datag[i]
        cols10 = np.where(row <= thresholds)[0]
        result_G_value[i, cols10] = row[cols10]

    G_df = pd.DataFrame(result_G_value, columns=cell_names) 

    final_G = G_df.T.groupby(G_df.columns).mean().T
    final_G = pd.DataFrame(data=final_G.values, columns=final_G.columns.tolist())
    return final_G


def process(root,a):
    for root in root:
        path = root[0]
        data_path = root[1]

        cell_names = datajiazai(path)
        model = pickle.load(open(path + 'caqtl/negative8/best_model.pkl', 'rb'))

        base_name = os.path.basename(data_path)
        prefix = base_name.split('_')[0]  

        df = pd.read_csv(os.path.join(path,'gwas',f'{prefix}_final.csv'))
        filtered_indices = df[(df['caqtl_probability'] > 0.5) & (df['G_value'] > 0.5)].index.to_numpy()


        final_shap = shapfenxi(data_path, model, cell_names,filtered_indices)
        final_shap = merge_neuron_types(final_shap)

        significance = joblib.load(os.path.join(path, 'gaussian_significance_data.pkl'))
        path2 = os.path.join(path, 'gwas', f'{prefix}_g_value.npy')
        datag, thresholds_per10, thresholds_per15, thresholds_per5 = gjiazai(path2, significance)
        datag = np.delete(datag, filtered_indices, axis=0)
        oldg = old_analysis(datag, thresholds_per10, cell_names)  
        oldg = merge_neuron_types(oldg)

        new_df = pd.DataFrame(0, index=final_shap.index, columns=final_shap.columns)
        print(new_df.shape)
 
        for i in range(final_shap.shape[0]): 
            for j in range(final_shap.shape[1]): 
                if final_shap.iloc[i, j] != 0 and pd.notna(oldg.iloc[i, j]):  
                    new_df.iloc[i, j] = 1  
        a.append(new_df)
    return a



print('计算AD------------------------------------------------')
ADroot = [
    ['/mnt/data0/users/lisg/Data/brain/acc/','/mnt/data0/users/lisg/Data/brain/acc/gwas/AD_predict.h5'],
    ['/mnt/data0/users/lisg/Data/brain/cmn/','/mnt/data0/users/lisg/Data/brain/cmn/gwas/AD_predict.h5'],
    ['/mnt/data0/users/lisg/Data/brain/cbl/','/mnt/data0/users/lisg/Data/brain/cbl/gwas/AD_predict.h5'],
    ['/mnt/data0/users/lisg/Data/brain/sub/','/mnt/data0/users/lisg/Data/brain/sub/gwas/AD_predict.h5'],
    ['/mnt/data0/users/lisg/Data/brain/ic/','/mnt/data0/users/lisg/Data/brain/ic/gwas/AD_predict.h5'],
    ['/mnt/data0/users/lisg/Data/brain/pn/','/mnt/data0/users/lisg/Data/brain/pn/gwas/AD_predict.h5'],
    ['/mnt/data0/users/lisg/Data/brain/pul/','/mnt/data0/users/lisg/Data/brain/pul/gwas/AD_predict.h5'],
]
a = []
process(ADroot,a)

all_columns = set()
for df in a:
    all_columns.update(df.columns)
all_columns = list(all_columns) 

heatmap_data = np.zeros((len(all_columns), len(a)))  
    for j, df in enumerate(a):
        if col in df.columns:
            heatmap_data[i, j] = (df[col] != 0).sum() 
        else:
            heatmap_data[i, j] = 0 

heatmap_df = pd.DataFrame(heatmap_data, index=all_columns, columns=[f'DF{i+1}' for i in range(len(a))])
heatmap_df.columns = ['ACC', 'CMN', 'CBL', 'SUB', 'IC', 'PN', 'PUL']
heatmap_df.to_csv('/mnt/data0/users/lisg/Data/GWAS/gwas_exp/AD_HEATMAP.csv')

print(heatmap_df.shape)

colors = ['#EAF5F8', '#87CBD8', '#0F98B0',]
cmap_name = 'custom_cmap'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=256)
plt.register_cmap(cmap=cm)

g = sns.clustermap(
    heatmap_df,
    annot=True,        
    fmt='.0f',          
    cmap=cmap_name,     
    figsize=(10, 12),    
    dendrogram_ratio=0.05, 
    cbar_pos=(0.95, 0.2, 0.03, 0.4), 
    yticklabels=True    
)


# g.ax_heatmap.set_xlabel('Brain Regions')
# g.ax_heatmap.set_ylabel('Cell Type')


plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=10, rotation=0)
plt.setp(g.ax_heatmap.get_xticklabels(), fontsize=10) 

plt.tight_layout()
plt.savefig('/mnt/data0/users/lisg/Data/brain/plot_results/fig5b.pdf', format='pdf', bbox_inches='tight')
plt.show()







# print('SCZ------------------------------------------------')
# SCZroot = [
#     ['/mnt/data0/users/lisg/Data/brain/acc/','/mnt/data0/users/lisg/Data/brain/acc/gwas/SCZ_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/cmn/','/mnt/data0/users/lisg/Data/brain/cmn/gwas/SCZ_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/cbl/','/mnt/data0/users/lisg/Data/brain/cbl/gwas/SCZ_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/sub/','/mnt/data0/users/lisg/Data/brain/sub/gwas/SCZ_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/ic/','/mnt/data0/users/lisg/Data/brain/ic/gwas/SCZ_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/pn/','/mnt/data0/users/lisg/Data/brain/pn/gwas/SCZ_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/pul/','/mnt/data0/users/lisg/Data/brain/pul/gwas/SCZ_predict.h5'],
# ]
# a= []
# process(SCZroot,a=a)




# print('Stroke------------------------------------------------')
# Strokeroot = [
#     ['/mnt/data0/users/lisg/Data/brain/acc/','/mnt/data0/users/lisg/Data/brain/acc/gwas/Stroke_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/cmn/','/mnt/data0/users/lisg/Data/brain/cmn/gwas/Stroke_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/cbl/','/mnt/data0/users/lisg/Data/brain/cbl/gwas/Stroke_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/sub/','/mnt/data0/users/lisg/Data/brain/sub/gwas/Stroke_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/ic/','/mnt/data0/users/lisg/Data/brain/ic/gwas/Stroke_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/pn/','/mnt/data0/users/lisg/Data/brain/pn/gwas/Stroke_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/pul/','/mnt/data0/users/lisg/Data/brain/pul/gwas/Stroke_predict.h5'],
# ]
# process(Strokeroot)



# print('ADHD------------------------------------------------')
# ADHDroot = [
#     ['/mnt/data0/users/lisg/Data/brain/acc/','/mnt/data0/users/lisg/Data/brain/acc/gwas/ADHD_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/cmn/','/mnt/data0/users/lisg/Data/brain/cmn/gwas/ADHD_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/cbl/','/mnt/data0/users/lisg/Data/brain/cbl/gwas/ADHD_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/sub/','/mnt/data0/users/lisg/Data/brain/sub/gwas/ADHD_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/ic/','/mnt/data0/users/lisg/Data/brain/ic/gwas/ADHD_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/pn/','/mnt/data0/users/lisg/Data/brain/pn/gwas/ADHD_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/pul/','/mnt/data0/users/lisg/Data/brain/pul/gwas/ADHD_predict.h5'],
# ]
# process(ADHDroot)
# print('ALS------------------------------------------------')
# ALSroot = [
#     ['/mnt/data0/users/lisg/Data/brain/acc/','/mnt/data0/users/lisg/Data/brain/acc/gwas/ALS_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/cmn/','/mnt/data0/users/lisg/Data/brain/cmn/gwas/ALS_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/cbl/','/mnt/data0/users/lisg/Data/brain/cbl/gwas/ALS_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/sub/','/mnt/data0/users/lisg/Data/brain/sub/gwas/ALS_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/ic/','/mnt/data0/users/lisg/Data/brain/ic/gwas/ALS_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/pn/','/mnt/data0/users/lisg/Data/brain/pn/gwas/ALS_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/pul/','/mnt/data0/users/lisg/Data/brain/pul/gwas/ALS_predict.h5'],
# ]
# process(ALSroot)
# print('ASD------------------------------------------------')
# ASDroot = [
#     ['/mnt/data0/users/lisg/Data/brain/acc/','/mnt/data0/users/lisg/Data/brain/acc/gwas/ASD_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/cmn/','/mnt/data0/users/lisg/Data/brain/cmn/gwas/ASD_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/cbl/','/mnt/data0/users/lisg/Data/brain/cbl/gwas/ASD_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/sub/','/mnt/data0/users/lisg/Data/brain/sub/gwas/ASD_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/ic/','/mnt/data0/users/lisg/Data/brain/ic/gwas/ASD_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/pn/','/mnt/data0/users/lisg/Data/brain/pn/gwas/ASD_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/pul/','/mnt/data0/users/lisg/Data/brain/pul/gwas/ASD_predict.h5'],
# ]
# process(ASDroot)
# print('BD------------------------------------------------')
# BDroot = [
#     ['/mnt/data0/users/lisg/Data/brain/acc/','/mnt/data0/users/lisg/Data/brain/acc/gwas/BD_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/cmn/','/mnt/data0/users/lisg/Data/brain/cmn/gwas/BD_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/cbl/','/mnt/data0/users/lisg/Data/brain/cbl/gwas/BD_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/sub/','/mnt/data0/users/lisg/Data/brain/sub/gwas/BD_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/ic/','/mnt/data0/users/lisg/Data/brain/ic/gwas/BD_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/pn/','/mnt/data0/users/lisg/Data/brain/pn/gwas/BD_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/pul/','/mnt/data0/users/lisg/Data/brain/pul/gwas/BD_predict.h5'],
# ]
# process(BDroot)
# print('MDD------------------------------------------------')
# MDDroot = [
#     ['/mnt/data0/users/lisg/Data/brain/acc/','/mnt/data0/users/lisg/Data/brain/acc/gwas/MDD_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/cmn/','/mnt/data0/users/lisg/Data/brain/cmn/gwas/MDD_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/cbl/','/mnt/data0/users/lisg/Data/brain/cbl/gwas/MDD_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/sub/','/mnt/data0/users/lisg/Data/brain/sub/gwas/MDD_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/ic/','/mnt/data0/users/lisg/Data/brain/ic/gwas/MDD_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/pn/','/mnt/data0/users/lisg/Data/brain/pn/gwas/MDD_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/pul/','/mnt/data0/users/lisg/Data/brain/pul/gwas/MDD_predict.h5'],
# ]
# process(MDDroot)
# print('MS------------------------------------------------')
# MSroot = [
#     ['/mnt/data0/users/lisg/Data/brain/acc/','/mnt/data0/users/lisg/Data/brain/acc/gwas/MS_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/cmn/','/mnt/data0/users/lisg/Data/brain/cmn/gwas/MS_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/cbl/','/mnt/data0/users/lisg/Data/brain/cbl/gwas/MS_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/sub/','/mnt/data0/users/lisg/Data/brain/sub/gwas/MS_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/ic/','/mnt/data0/users/lisg/Data/brain/ic/gwas/MS_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/pn/','/mnt/data0/users/lisg/Data/brain/pn/gwas/MS_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/pul/','/mnt/data0/users/lisg/Data/brain/pul/gwas/MS_predict.h5'],
# ]
# process(MSroot)
# print('PD------------------------------------------------')
# PDroot = [
#     ['/mnt/data0/users/lisg/Data/brain/acc/','/mnt/data0/users/lisg/Data/brain/acc/gwas/PD_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/cmn/','/mnt/data0/users/lisg/Data/brain/cmn/gwas/PD_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/cbl/','/mnt/data0/users/lisg/Data/brain/cbl/gwas/PD_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/sub/','/mnt/data0/users/lisg/Data/brain/sub/gwas/PD_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/ic/','/mnt/data0/users/lisg/Data/brain/ic/gwas/PD_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/pn/','/mnt/data0/users/lisg/Data/brain/pn/gwas/PD_predict.h5'],
#     ['/mnt/data0/users/lisg/Data/brain/pul/','/mnt/data0/users/lisg/Data/brain/pul/gwas/PD_predict.h5'],
# ]
# process(PDroot)




