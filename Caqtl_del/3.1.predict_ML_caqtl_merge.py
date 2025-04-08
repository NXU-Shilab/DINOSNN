import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score


ROOT = [
    '/mnt/data0/users/lisg/Data/brain/acc/caqtl/negative8/',
    '/mnt/data0/users/lisg/Data/brain/cbl/caqtl/negative8/',
    '/mnt/data0/users/lisg/Data/brain/cmn/caqtl/negative8/',
    '/mnt/data0/users/lisg/Data/brain/ic/caqtl/negative8/',
    '/mnt/data0/users/lisg/Data/brain/pn/caqtl/negative8/',
    '/mnt/data0/users/lisg/Data/brain/pul/caqtl/negative8/',
    '/mnt/data0/users/lisg/Data/brain/sub/caqtl/negative8/',
]
ML_AUC_store = pd.DataFrame(columns=['tissue', 'AUC', 'AUPR', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'MCC'])
for root in ROOT:


    data = np.load(os.path.join(root,'data.npz'))
    data_test, label_test = data['data_test'], data['label_test']
    model_path = os.path.join(root, 'best_model.pkl')
    with open(model_path, 'rb') as model_file:
        best_model = pickle.load(model_file)
    preds = best_model.predict(data_test)
    testauc = roc_auc_score(label_test, preds)
    testaupr = average_precision_score(label_test, preds)

    threshold = 0.5
    y_pred = [1 if prob >= threshold else 0 for prob in preds]
    accuracy_value = accuracy_score(label_test, y_pred) 
    precision_value = precision_score(label_test, y_pred)
    recall_value = recall_score(label_test, y_pred)
    f1_value = f1_score(label_test, y_pred)
    mcc_value = matthews_corrcoef(label_test, y_pred)

    ML_AUC_store.loc[len(ML_AUC_store)] = [root.split(os.sep)[-4], testauc, testaupr, accuracy_value, precision_value,
                                           recall_value,
                                           f1_value, mcc_value]

print(ML_AUC_store)
ML_AUC_store.to_csv('/mnt/data0/users/lisg/Data/brain/caqtl_ml_result.csv', index=False)
