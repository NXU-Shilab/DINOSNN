import os
import pickle
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, \
    roc_auc_score, average_precision_score
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


def remove_suffix(s):
    return re.sub(r'_\d+$', '', re.sub(r'_\d+$', '', s))
def add_suffix_to_duplicates(strings):
    count_dict = {}
    result = []
    for s in strings:
        if s in count_dict:
            count_dict[s] += 1
            result.append(f"{s}-{count_dict[s]}")
        else:
            count_dict[s] = 1
            result.append(s)
    return result

def analyze_cell(cell_name):
    counts = cell_name.value_counts()
    unique_count = cell_name.nunique()
    print("The number of each type of cell:",counts)
    print("How many types of cells are there in total:",unique_count)
    counts_df = pd.DataFrame(counts)
    counts_df.columns = ['count']
    # print(counts_df.index.tolist())
    # counts_df.to_csv('/mnt/data0/users/lisg/Data/counts2.csv')
    return counts_df

def load_bgmm(column_number,ecdfs_path):
    file_name = f'gmm_col_{column_number}.pkl'
    file_path = os.path.join(ecdfs_path, file_name)
    return joblib.load(file_path)
def normalize_array(arr, max_value, min_value):
    range_value = max_value - min_value
    normalized_arr = (arr - min_value) / range_value
    return normalized_arr
def process_row(array, strings):

    df = pd.DataFrame({
        'strings': strings,
        'values': array  
    })


    result = df.groupby('strings')['values'].mean()

    return (
        pd.Series(result.index), 
        result.values 
    )
def create_matrix(value, cell, all_strings):
  
    result = np.full((len(value), len(all_strings)), np.nan)

    
    string_to_idx = {s: i for i, s in enumerate(all_strings)}


    for row_idx in range(len(value)):
      
        current_strings = cell[row_idx]  
        current_values = value[row_idx]  


        assert len(current_strings) == len(current_values), f"Row {row_idx}: strings and values length mismatch"


        for s, v in zip(current_strings, current_values):
            col_idx = string_to_idx[s]
            result[row_idx, col_idx] = v

    return result

def calculate_metrics_with_best_threshold(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    youden_index = tpr - fpr
    best_threshold_index = np.argmax(youden_index)
    best_threshold = thresholds[best_threshold_index]
    y_pred = (y_scores >= best_threshold).astype(int)
    auc = roc_auc_score(y_true, y_scores)
    testaupr = average_precision_score(y_true, y_scores)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    mcc = matthews_corrcoef(y_true, y_pred)
    return round(auc, 3),round(accuracy, 3), round(recall, 3),round(specificity, 3),round(precision, 3),round(f1, 3),round(mcc, 3),round(testaupr, 3)

