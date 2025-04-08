import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score

def calculate_best_threshold(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    youden_index = tpr - fpr
    best_threshold_index = np.argmax(youden_index)
    best_threshold = thresholds[best_threshold_index]
    return best_threshold

ROOT = [
    '/mnt/data0/users/lisg/Data/brain/acc/eqtl_acc/',
    '/mnt/data0/users/lisg/Data/brain/cbl/eqtl_cbl/',
    '/mnt/data0/users/lisg/Data/brain/cmn/eqtl_cmn/',
    '/mnt/data0/users/lisg/Data/brain/sub/eqtl_sub/',
]
for root in ROOT:
    ML_AUC_store = pd.DataFrame(columns=['path', 'AUC', 'AUPR', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'MCC'])
    root_path = [os.path.join(root,'negative8'),os.path.join(root, 'negative2'),os.path.join(root, 'negative1'),]

    for path_model in root_path:
        path = [os.path.join(path_model, f'Fold_{i}') for i in range(1, 11)]
        for data_path in path:
            print(data_path)
            data = np.load(data_path + '/data.npz')
            data_test, label_test = data['data_test'], data['label_test']
            model_path = os.path.join(data_path, 'best_valid.pkl')
            with open(model_path, 'rb') as model_file:
                best_model = pickle.load(model_file)
            preds = best_model.predict(data_test)
            testauc = roc_auc_score(label_test, preds)
            testaupr = average_precision_score(label_test, preds)

            threshold  = 0.5
            y_pred = [1 if prob >= threshold else 0 for prob in preds]
            accuracy_value = accuracy_score(label_test, y_pred)# 计算 Accuracy
            precision_value = precision_score(label_test, y_pred)
            recall_value = recall_score(label_test, y_pred)
            f1_value = f1_score(label_test, y_pred)
            mcc_value = matthews_corrcoef(label_test, y_pred)
            ML_AUC_store.loc[len(ML_AUC_store)] = [data_path, testauc, testaupr, accuracy_value, precision_value, recall_value,
                                                       f1_value, mcc_value]


    print(ML_AUC_store)
    ML_AUC_store.to_csv(root + 'ML_result.csv',index=False)