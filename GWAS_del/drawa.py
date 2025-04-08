import os
import pickle
import re

import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import scanpy as sc
import shap
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
def remove_suffix(s):
    return re.sub(r'_\d+$', '', re.sub(r'_\d+$', '', s))
def datajiazai(path):
    ad = sc.read_h5ad(os.path.join(path,'cluster_ad.h5ad'))
    cell_names = ad.obs['celltype']
    cell_names = cell_names.reset_index(drop=True)
    cell_names = cell_names.apply(remove_suffix)  
    return cell_names

def shapfenxi(data_path,model,cell_names):
    with h5py.File(data_path, 'r') as f:
        data = np.array(f['product_data'])
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
    return final_shap,shap_values

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


def old_analysis(datag,thresholds,shap_values,cell_names):
    result_G_value = np.full(datag.shape, np.nan)
    result_shap_value = np.full(datag.shape, np.nan)
    for i in range(datag.shape[0]):

        row = datag[i]
        cols10 = np.where(row <= thresholds)[0]
        result_G_value[i, cols10] = row[cols10]

        row1 = shap_values.values[i]
        indices_ca = np.where(row1 > 0)[0]
        result_shap_value[i, indices_ca] = row1[indices_ca]


    G_df = pd.DataFrame(result_G_value, columns=cell_names)  
    final_G = G_df.T.groupby(G_df.columns).mean().T
    final_G = pd.DataFrame(data=final_G.values, columns=final_G.columns.tolist())
    final_G = merge_neuron_types(final_G)

    shap_df = pd.DataFrame(result_shap_value, columns=cell_names)
    final_shap = shap_df.T.groupby(shap_df.columns).mean().T
    final_shap = pd.DataFrame(data=final_shap.values,columns=final_shap.columns.tolist())
    final_shap = merge_neuron_types(final_shap)

    df1 = final_G
    df2 = final_shap

    for index, row in df1.iterrows():

        non_zero_columns = [
            col for col in df1.columns
            if pd.notna(row[col]) and row[col] != 0
        ]


        if non_zero_columns:
            print(non_zero_columns)


    for index, row1 in df1.iterrows():
        row2 = df2.iloc[index]


        common_columns = [
            col for col in df1.columns
            if pd.notna(row1[col]) and row1[col] != 0 and pd.notna(row2[col]) and row2[col] != 0
        ]


        if common_columns:
            print(common_columns)


def new_analysis(datag,thresholds,i,cell_names):
    datag = datag[i]
    datag_df = pd.DataFrame([datag], columns=cell_names)
    thresholds_df = pd.DataFrame([thresholds], columns=cell_names)

    means_datag_df, thresh_df = process_single_sample(datag_df, thresholds_df)
    print(means_datag_df)

    return means_datag_df,thresh_df
ROOT = [
    '/mnt/data0/users/lisg/Data/brain/acc',
    # '/mnt/data0/users/lisg/Data/brain/cmn',
    # '/mnt/data0/users/lisg/Data/brain/cbl',
    # '/mnt/data0/users/lisg/Data/brain/sub',
    # '/mnt/data0/users/lisg/Data/brain/ic',
    # '/mnt/data0/users/lisg/Data/brain/pn',
    # '/mnt/data0/users/lisg/Data/brain/pul',
]
gwas_data_path = '/mnt/data0/users/lisg/Data/GWAS'


columns = ['AD', 'ADHD', 'ALS', 'ASD', 'BD', 'MDD', 'MS', 'PD', 'SCZ', 'Stroke']

all_row_numbers = []

for index, root_path in enumerate(ROOT):
    data_dir = [
        [os.path.join(root_path, 'gwas/AD_final.csv'),os.path.join(root_path, 'gwas/AD_predict.h5')]]
    for data_path in data_dir:
        file_path = data_path[0]  
        df = pd.read_csv(file_path)
        filtered_indices = df[(df['caqtl_probability'] > 0.5) & (df['G_value'] > 0.5)].index.to_numpy()


        cell_names = datajiazai(root_path)
        model = pickle.load(open(os.path.join(root_path,'caqtl/negative8/best_model.pkl'), 'rb'))

        with h5py.File(data_path[1], 'r') as f:
            data = np.array(f['product_data'])
        data = np.delete(data, filtered_indices, axis=0)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer(data)
        shap_values.feature_names = cell_names
        shap.plots.beeswarm(shap_values)
        # result_shap_value = np.full(data.shape, np.nan)
        # for i in range(data.shape[0]):
        #     row1 = shap_values.values[i]
        #     indices_ca = np.where(row1 > 0)[0]
        #     result_shap_value[i, indices_ca] = row1[indices_ca]
        # shap_df = pd.DataFrame(result_shap_value, columns=cell_names)  
        # final_shap = shap_df.T.groupby(shap_df.columns).mean().T
        # final_shap = pd.DataFrame(data=final_shap.values, columns=final_shap.columns.tolist())
        # final_shap = final_shap.fillna(0)
        #
        # final_shap = merge_neuron_types(final_shap)



