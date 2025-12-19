import gc
import os
import pickle
import shutil

import configargparse
import h5py
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import  train_test_split
import numpy as np
import optuna
import optuna.integration.lightgbm as lgb
from lightgbm import early_stopping

def make_parser():
    parser = configargparse.ArgParser(description="")
    parser.add_argument('--caqtl', type=str,default='../PartII_data/CaQTL_data')
    parser.add_argument('--seed', type=int, default=10)
    return parser

def main():
    parser = make_parser()
    args = parser.parse_args()
    np.random.seed(args.seed)

    positie_path = os.path.join(args.caqtl,"positive_predict.h5")
    negative_path = os.path.join(args.caqtl,"negative_predict.h5")
    with h5py.File(positie_path, 'r') as f:
        positive = f['product_data']
        positive = np.array(positive)
    with h5py.File(negative_path, 'r') as f:
        negative = f['product_data']
        negative = np.array(negative)
    out_dir= os.path.join(args.caqtl, "ML")
    os.makedirs(out_dir, exist_ok=True)
    train_data, test_data, train_label, test_label, train_indices, test_indices=split_data(pos_data=positive, nega_data=negative, out_dir=out_dir, seed=args.seed)
    indices = np.random.permutation(train_data.shape[0])
    data_train = train_data[indices]
    label_train = train_label[indices]
    train_data = lgb.Dataset(data_train, label=label_train)
    valid_data = lgb.Dataset(test_data, test_label, reference=train_data)
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'num_threads': 20,
        'device': 'gpu',
        'verbosity': -1,
        'metrics': ['binary_logloss', 'auc'],
        'feature_pre_filter': False,
        'is_unbalance': True,
    }
    out_path = os.path.join(out_dir, 'save_boosters')
    os.makedirs(out_path, exist_ok=True)
    tuner = lgb.LightGBMTuner(
        params=params, train_set=train_data, valid_sets=[valid_data], valid_names=['yanzheng'],
        callbacks=[early_stopping(30)],
        model_dir=out_path
    )
    tuner.run()

    model_dir = out_path
    results = []
    for model_file in os.listdir(model_dir):
        if model_file.endswith('.pkl'):
            model_path = os.path.join(model_dir, model_file)
            model_name = os.path.basename(model_path)
            with open(model_path, 'rb') as model_file:
                best_model = pickle.load(model_file)
            testpreds = best_model.predict(test_data)
            testauc = roc_auc_score(test_label, testpreds)
            results.append((model_name, testauc))
            del best_model
            gc.collect()
    df = pd.DataFrame(results, columns=['Model File', 'testAUC'])
    best_model_row = df.loc[df['testAUC'].idxmax()]
    best_model_file = best_model_row['Model File']
    best_model_path = os.path.join(model_dir, best_model_file)
    best_valid_path = os.path.join(out_dir, 'best_model.pkl')
    shutil.copy2(best_model_path, best_valid_path)
    shutil.rmtree(out_path)




def split_data(pos_data,nega_data,out_dir,seed):
    if not os.path.exists(out_dir): os.mkdir(out_dir)
    pos_label =  np.ones((pos_data.shape[0],1))
    nega_label = np.zeros((nega_data.shape[0],1))
    pos_data = np.hstack((pos_data, pos_label))
    nega_data = np.hstack((nega_data, nega_label))
    all_data = np.vstack((pos_data, nega_data))
    Data = all_data[:, :all_data.shape[1] - 1]
    label = all_data[:, all_data.shape[1] - 1]
    original_indices = np.arange(len(all_data))
    train_data, test_data, train_label, test_label, train_indices, test_indices = train_test_split(
        Data, label, original_indices, test_size=0.2, stratify=label, random_state=seed)
    np.savez(os.path.join(out_dir, "data.npz"),
             data_train=train_data, data_test=test_data,
             label_train=train_label, label_test=test_label,
             test_indices_in_original=test_indices)

    return train_data, test_data, train_label, test_label, train_indices, test_indices








