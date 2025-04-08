import gc
import os
import pickle
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import roc_auc_score
from scipy.stats import hmean
from tqdm import tqdm
from utils import load_bgmm,remove_suffix, analyze_cell
import xgboost as xgb
import joblib

caqtlpath = '/mnt/data0/users/lisg/Data/brain/sub/'
bgmm_path = caqtlpath + 'gaussian'
with h5py.File(caqtlpath + 'caqtl/positive_predict.h5', 'r') as f:
    positive = f['product_data']
    positive = np.array(positive)
with h5py.File(os.path.join(caqtlpath,'caqtl/negative0.008_predict.h5'), 'r') as f:
    negative8 = f['product_data']
    negative8 = np.array(negative8)

all_data = np.vstack((positive, negative8)) 
ad =sc.read_h5ad(caqtlpath + 'cluster_ad.h5ad')
cell_names = ad.obs['celltype']
cell_names = cell_names.reset_index(drop=True)
cell_names = cell_names.apply(remove_suffix) 
original_count = analyze_cell(cell_names)

data = np.load(caqtlpath + 'caqtl/negative8/data.npz')
data_test, label_test, = data['data_test'], data['label_test']
data_train, label_train, = data['data_train'], data['label_train']
test_indices = data['test_indices_in_original']
data_train_g,data_test_g = [],[]
for i in tqdm(range(data_train.shape[1])):
    gmm = load_bgmm(i, bgmm_path)
    col_data = data_train[:, i].reshape(-1, 1)
    col_data2 = data_test[:, i].reshape(-1, 1)
    density = np.exp(gmm.score_samples(col_data))
    density2 = np.exp(gmm.score_samples(col_data2))
    data_train_g.append(density)
    data_test_g.append(density2)
data_train_1 = np.array(data_train_g).transpose()
data_test_1 = np.array(data_test_g).transpose()


cell_type_indices = {}
for cell_type in sorted(set(cell_names)):
    cell_type_indices[cell_type] = [i for i, name in enumerate(cell_names) if name == cell_type]
data_train_2 = pd.DataFrame(data_train_1, columns=cell_names)
data_train_2_1 = {}
for cell_type, indices in cell_type_indices.items():
    data_train_2_1[cell_type] = hmean(data_train_2.iloc[:, indices].values, axis=1)
X_train = pd.DataFrame(data_train_2_1).values
data_test_2 = pd.DataFrame(data_test_1, columns=cell_names)
data_test_2_1 = {}
for cell_type, indices in cell_type_indices.items():
    data_test_2_1[cell_type] = hmean(data_test_2.iloc[:, indices].values, axis=1)
X_test = pd.DataFrame(data_test_2_1).values

xgb_clf = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.01,
    max_depth=5,
    random_state=42,
    eval_metric='auc',  
    early_stopping_rounds=10
)

eval_set = [(X_train, label_train), (X_test, label_test)]
xgb_clf.fit(
    X_train,
    label_train,
    eval_set=eval_set,
    verbose=True,
)
y_pred_proba = xgb_clf.predict_proba(X_test)[:, 1]



model_save_path = os.path.join(caqtlpath,'xgb_model.json')
xgb_clf.save_model(model_save_path)

# loaded_model = xgb.XGBClassifier()
# loaded_model.load_model(model_save_path)
# loaded_pred = loaded_model.predict_proba(X_test)[:, 1]  
# print(roc_auc_score(label_test, loaded_pred))
