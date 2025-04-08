import configargparse
import pandas as pd
import sys
import os
import time
import torch
import h5py
from torch import nn
import numpy as np
from tqdm import tqdm
class Dataset(torch.utils.data.Dataset):
    def __init__(self, ref, vary):
        self.ref = ref
        self.vary = vary
    def __len__(self):
        return self.ref.shape[0]
    def __getitem__(self, index):
        ref_data = self.ref[index]
        vary_data = self.vary[index]
        return ref_data,vary_data
'''
Used for predicting positive and negative data of eqtl on corresponding models
Enter eqtl folder:/ mnt/data0/users/lisg/Data/eqtl/eqtl_acc/， Contains 4 files: positive.h5 negative0.2.h5 negative1.h5 negative0.008.h5
Enter the best model folder: '/ mnt/data0/users/lisg/Data/brain/acc'
'''

def make_parser():
    parser = configargparse.ArgParser(description="Preprocessing eqtl data")
    parser.add_argument('--bestmodel', type=str,
                        help='/mnt/data0/users/lisg/Data/brain/acc')
    parser.add_argument('--gpu_id', type=int, nargs='+', help='gpu id')
    parser.add_argument('--eqtl', type=str, default=None,
                        help='')
    return parser

def main():
    parser = make_parser()
    args = parser.parse_args()
    bestmodel = ensure_trailing_slash(args.bestmodel)
    bestmodel_path = bestmodel + 'train_output/'
    device_id = args.gpu_id
    use_cuda = torch.cuda.is_available()  
    device = torch.device("cuda:%s" % device_id[0] if use_cuda else "cpu")

    checkpoint = torch.load('%sbest_val_auc_model.pth' % bestmodel_path, map_location=device)
    test_data = h5py.File('%stest_data.h5' % bestmodel, 'r')
    test_Y = test_data["test_Y"]
    num_cell = len(test_Y[0])
    test_data.close()

    if args.eqtl is not None:

        eqtl = ensure_trailing_slash(args.eqtl)
        out = eqtl.split('/')[-2]
        output_path = bestmodel + out + '/'
        if not os.path.exists(output_path): os.mkdir(output_path)

        from Model_smt import model
        model = model(num_cell=num_cell)
        model = nn.DataParallel(model, device_ids=device_id)
        model.to(device)
        model.load_state_dict(checkpoint['best_model_state'])  

        positive = eqtl + 'positive.h5'
        negative_8 = eqtl + 'negative0.008.h5'
        negative_2 = eqtl + 'negative0.2.h5'
        negative_1 = eqtl + 'negative1.h5'

        # positive
        with h5py.File(positive, 'r') as f:
            ref_dataset = f['ref']
            vary_dataset = f["vary"]
            DataLoader = torch.utils.data.DataLoader(Dataset(ref_dataset, vary_dataset), batch_size=1, shuffle=False)
            diff_data, logit_data, product_data = test_func(model=model, DataLoader=DataLoader, device=device)
            with h5py.File(output_path + 'positive_predict.h5', 'w') as hf:
                hf.create_dataset("diff_data", data=diff_data)
                hf.create_dataset("logit_data", data=logit_data)
                hf.create_dataset("product_data", data=product_data)

        # negative_0.008
        with h5py.File(negative_8, 'r') as f:
            ref_dataset = f['ref']
            vary_dataset = f["vary"]
            DataLoader = torch.utils.data.DataLoader(Dataset(ref_dataset, vary_dataset), batch_size=1, shuffle=False)
            diff_data, logit_data, product_data = test_func(model=model, DataLoader=DataLoader, device=device)
            with h5py.File(output_path + 'negative0.008_predict.h5', 'w') as hf:
                hf.create_dataset("diff_data", data=diff_data)
                hf.create_dataset("logit_data", data=logit_data)
                hf.create_dataset("product_data", data=product_data)

        # negative_0.2
        with h5py.File(negative_2, 'r') as f:
            ref_dataset = f['ref']
            vary_dataset = f["vary"]
            DataLoader = torch.utils.data.DataLoader(Dataset(ref_dataset, vary_dataset), batch_size=1, shuffle=False)
            diff_data, logit_data, product_data = test_func(model=model, DataLoader=DataLoader, device=device)
            with h5py.File(output_path + 'negative0.2_predict.h5', 'w') as hf:
                hf.create_dataset("diff_data", data=diff_data)
                hf.create_dataset("logit_data", data=logit_data)
                hf.create_dataset("product_data", data=product_data)

        # negative_1
        with h5py.File(negative_1, 'r') as f:
            ref_dataset = f['ref']
            vary_dataset = f["vary"]
            DataLoader = torch.utils.data.DataLoader(Dataset(ref_dataset, vary_dataset), batch_size=1, shuffle=False)
            diff_data, logit_data, product_data = test_func(model=model, DataLoader=DataLoader, device=device)
            with h5py.File(output_path + 'negative1_predict.h5', 'w') as hf:
                hf.create_dataset("diff_data", data=diff_data)
                hf.create_dataset("logit_data", data=logit_data)
                hf.create_dataset("product_data", data=product_data)



def test_func(model, DataLoader, device):
    model.eval()
    diff_data, logit_data, product_data = [], [], []
    sig = nn.Sigmoid()
    with torch.no_grad():
        val_bar = tqdm(DataLoader, file=sys.stdout)
        for id, (ref, vary) in enumerate(val_bar):
            ref = ref.to(device, non_blocking=True)
            vary = vary.to(device, non_blocking=True)
            ref_predict = sig(model(ref))
            vary_predict = sig(model(vary))
   
            diff_abs = ref_predict - vary_predict
    
            logit_diff = torch.log(ref_predict / (1 - ref_predict + 1e-12)) - torch.log(vary_predict / (1 - vary_predict + 1e-12))


            product_abs = torch.abs(diff_abs * logit_diff)

            diff_data.append(diff_abs.cpu())
            logit_data.append(logit_diff.cpu())
            product_data.append(product_abs.cpu())

        diff_data = torch.cat(diff_data, 0).numpy()
        logit_data = torch.cat(logit_data, 0).numpy()
        product_data = torch.cat(product_data, 0).numpy()

    return diff_data, logit_data ,product_data

def ensure_trailing_slash(path):
    if not path.endswith('/'):
        path += '/'
    return path



if __name__ == "__main__":
    main()

