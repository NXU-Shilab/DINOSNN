import numpy as np
import optuna
import os
import optuna.integration.lightgbm as lgb
from lightgbm import early_stopping
seed = 10
np.random.seed(seed)


directory_path = [
    '/mnt/data0/users/lisg/Data/brain2/acc/eqtl_acc/negative1',
    '/mnt/data0/users/lisg/Data/brain2/acc/eqtl_acc/negative2',
    '/mnt/data0/users/lisg/Data/brain2/acc/eqtl_acc/negative8'
]


for directory in directory_path:
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            if file_name == 'data.npz':

                file_path = os.path.join(root, file_name)

                data = np.load(file_path)

                print(f"File path: {file_path}")
                print(root)
                out_path = os.path.join(root, 'save_boosters')
                print(out_path)
                '''--------------------------------------------------------------'''
                data_train, label_train,= data['data_train'],data['label_train']
                data_valid, label_valid,= data['data_valid'],data['label_valid']
                indices = np.random.permutation(label_train.shape[0])
                data_train = data_train[indices]
                label_train = label_train[indices]
                train_data = lgb.Dataset(data_train, label=label_train)
                valid_data = lgb.Dataset(data_valid ,label_valid, reference=train_data)
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
                tuner = lgb.LightGBMTuner(
                    params = params,train_set = train_data,valid_sets =[valid_data],valid_names = ['yanzheng'],
                    callbacks=[early_stopping(30)],
                    model_dir= out_path
                )
                tuner.run()


                print('--------------------------------')