import numpy as np
import os
import h5py
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
import joblib

root = [
    '/mnt/data0/users/lisg/Data/brain/cbl/',
    '/mnt/data0/users/lisg/Data/brain/cmn/',
    '/mnt/data0/users/lisg/Data/brain/ic/',
    '/mnt/data0/users/lisg/Data/brain/pn/',
    '/mnt/data0/users/lisg/Data/brain/pul/',
    '/mnt/data0/users/lisg/Data/brain/sub/',
    '/mnt/data0/users/lisg/Data/brain/acc/',
]
for file_path in root:
    input_file = os.path.join(file_path,'random_1kgp_predict.h5')
    output_dir = os.path.join(file_path, 'gaussian')
    os.makedirs(output_dir, exist_ok=True)


    significance_data = {}

    with h5py.File(input_file, 'r') as input_data:
        product_data = input_data["product_data"]

        total_columns = product_data.shape[1]

        for col in tqdm(range(total_columns), desc="Processing columns"):
            column_data = product_data[:, col]


            nan_rows = np.isnan(column_data)
            if nan_rows.any():
                nan_indices = np.where(nan_rows)[0]
                print(f"Column {col} contains NaN values at rows: {nan_indices}")
                mean_value = np.nanmean(column_data)
                column_data[nan_rows] = mean_value
            inf_rows = np.isinf(column_data)
            if inf_rows.any():
                inf_indices = np.where(inf_rows)[0]
                print(f"Column {col} contains inf values at rows: {inf_indices}")
                column_data[inf_rows] = np.nan
            max_value = np.nanmax(column_data) 
            column_data[inf_rows] = max_value


            gmm = GaussianMixture(n_components=7, random_state=10)
            gmm.fit(column_data.reshape(-1, 1))


            output_file_path = os.path.join(output_dir, f'gmm_col_{col}.pkl')
            joblib.dump(gmm, output_file_path)

            densities = np.exp(gmm.score_samples(column_data.reshape(-1, 1)))


            percentile_5 = np.percentile(densities, 5)
            percentile_10 = np.percentile(densities, 10)
            percentile_15 = np.percentile(densities, 15)
            percentile_20 = np.percentile(densities, 20)


            mean_value = np.mean(column_data)
            std_value = np.std(column_data)

            # 计算概率密度的最大值和最小值
            max_density = np.max(densities)
            min_density = np.min(densities)


            significance_data[col] = {
                "percentile_5": percentile_5,
                "percentile_10": percentile_10,
                "percentile_15": percentile_15,
                "percentile_20": percentile_20,
                "mean": mean_value,
                "std": std_value,
                "max_density": max_density,
                "min_density": min_density
            }

    significance_data_file = os.path.join(file_path, "gaussian_significance_data.pkl")
    joblib.dump(significance_data, significance_data_file)
    print("Significance data saved to:", significance_data_file)