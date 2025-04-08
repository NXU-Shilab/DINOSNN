import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import gc
import shutil

ROOT = [
    '/mnt/data0/users/lisg/Data/brain/acc/caqtl/negative8/',
    '/mnt/data0/users/lisg/Data/brain/cbl/caqtl/negative8/',
    '/mnt/data0/users/lisg/Data/brain/cmn/caqtl/negative8/',
    '/mnt/data0/users/lisg/Data/brain/ic/caqtl/negative8/',
    '/mnt/data0/users/lisg/Data/brain/pn/caqtl/negative8/',
    '/mnt/data0/users/lisg/Data/brain/pul/caqtl/negative8/',
    '/mnt/data0/users/lisg/Data/brain/sub/caqtl/negative8/',
]

for root_path in ROOT:
    model_dir = os.path.join(root_path, 'save_boosters')
    data = np.load(root_path + '/data.npz')
    data_test, label_test, = data['data_test'], data['label_test']
    results = []
    for model_file in os.listdir(model_dir):
        if model_file.endswith('.pkl'):
            model_path = os.path.join(model_dir, model_file)
            model_name = os.path.basename(model_path)

            with open(model_path, 'rb') as model_file:
                best_model = pickle.load(model_file)

            testpreds = best_model.predict(data_test)
            testauc = roc_auc_score(label_test, testpreds)

            results.append((model_name, testauc))

            del best_model
            gc.collect()

    df = pd.DataFrame(results, columns=['Model File','testAUC'])
    df.to_csv(root_path + 'ML_AUC.csv', index=False)


    best_model_row = df.loc[df['testAUC'].idxmax()]
    best_model_file = best_model_row['Model File']
    best_model_path = os.path.join(model_dir, best_model_file)

   
    best_valid_path = os.path.join(root_path, 'best_model.pkl')
    shutil.copy2(best_model_path, best_valid_path)