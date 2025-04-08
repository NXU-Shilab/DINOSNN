import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import gc
import shutil

ROOT = [
    '/mnt/data0/users/lisg/Data/brain2/acc/eqtl_acc/negative8/',
    '/mnt/data0/users/lisg/Data/brain2/acc/eqtl_acc/negative2/',
    '/mnt/data0/users/lisg/Data/brain2/acc/eqtl_acc/negative1/',
    '/mnt/data0/users/lisg/Data/brain2/cbl/eqtl_cbl/negative8/',
    '/mnt/data0/users/lisg/Data/brain2/cbl/eqtl_cbl/negative2/',
    '/mnt/data0/users/lisg/Data/brain2/cbl/eqtl_cbl/negative1/',
    '/mnt/data0/users/lisg/Data/brain2/cmn/eqtl_cmn/negative8/',
    '/mnt/data0/users/lisg/Data/brain2/cmn/eqtl_cmn/negative2/',
    '/mnt/data0/users/lisg/Data/brain2/cmn/eqtl_cmn/negative1/',
    '/mnt/data0/users/lisg/Data/brain2/sub/eqtl_sub/negative8/',
    '/mnt/data0/users/lisg/Data/brain2/sub/eqtl_sub/negative2/',
    '/mnt/data0/users/lisg/Data/brain2/sub/eqtl_sub/negative1/',
]

for root_path in ROOT:
    print(root_path)
    path = [os.path.join(root_path, f'Fold_{i}') for i in range(1, 11)]

    for data_path in path:
        model_dir = os.path.join(data_path, 'save_boosters')
        data = np.load(data_path + '/data.npz')
        valid_X, valid_y = data['data_valid'], data['label_valid']
        data_test, label_test = data['data_test'], data['label_test']

        results = []
        for model_file in os.listdir(model_dir):
            if model_file.endswith('.pkl'):
                model_path = os.path.join(model_dir, model_file)
                model_name = os.path.basename(model_path)

                with open(model_path, 'rb') as model_file:
                    best_model = pickle.load(model_file)

                validpreds = best_model.predict(valid_X)
                validauc = roc_auc_score(valid_y, validpreds)

                testpreds = best_model.predict(data_test)
                testauc = roc_auc_score(label_test, testpreds)

                results.append((model_name, validauc, testauc))

                del best_model
                gc.collect()

        df = pd.DataFrame(results, columns=['Model File', 'validAUC', 'testAUC'])
        df.to_csv(data_path + '_ML_AUC.csv', index=False)


        best_model_row = df.loc[df['validAUC'].idxmax()]
        best_model_file = best_model_row['Model File']
        best_model_path = os.path.join(model_dir, best_model_file)


        best_valid_path = os.path.join(data_path, 'best_valid.pkl')
        shutil.copy2(best_model_path, best_valid_path)

        print(f"Best model for {data_path}: {best_model_file}")
        print(f"Validation AUC: {best_model_row['validAUC']}")
        print(f"Test AUC: {best_model_row['testAUC']}")
        print(f"Copied to: {best_valid_path}")
        print("--------------------")