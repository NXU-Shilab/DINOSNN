import os
import pickle
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, \
    roc_auc_score
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
def load_bgmm(column_number,ecdfs_path):
    file_name = f'gmm_col_{column_number}.pkl'
    file_path = os.path.join(ecdfs_path, file_name)
    return joblib.load(file_path)

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


def load_ecdf(column_number,ecdfs_path):
    file_name = f'ecdf_col_{column_number}.pkl'
    file_path = os.path.join(ecdfs_path, file_name)
    return joblib.load(file_path)

def calculate_metrics_with_best_threshold(y_true, y_scores):
    
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    youden_index = tpr - fpr
    
    best_threshold_index = np.argmax(youden_index)
    best_threshold = thresholds[best_threshold_index]

    y_pred = (y_scores >= best_threshold).astype(int)
  
    auc = roc_auc_score(y_true, y_scores)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
  
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)

    return round(auc, 3),round(accuracy, 3), round(recall, 3),round(specificity, 3),round(precision, 3),round(f1, 3),#round(mcc, 3)

'''--------------------------------------------------------------------------------------------------------'''
