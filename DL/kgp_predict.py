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
You must input the root folder where the best model is located
If you want to predict 1 million random SNPs in a thousand human genomes, then the - kgp parameter input is: "/mnt/data0/users/lisg/Data/1kgp/random_million_1kgp.h5"
The output will be saved to the best model folder in the input random_1kgp_redist.h5
'''
def make_parser():
    parser = configargparse.ArgParser(description="Preprocessing eqtl data")
    parser.add_argument('--bestmodel', type=str,
                        help='/mnt/data0/users/lisg/Data/brain2/acc')
    parser.add_argument('--gpu_id', type=int, nargs='+', help='gpu id')
    parser.add_argument('--kgp', type=str, default=None,
                        help='Pass in the absolute path of the file, e.g.:')
    parser.add_argument('--eqtl', type=str, default=None,
                        help='')
    parser.add_argument('--caqtl', type=str, default=None,
                        help='')
    parser.add_argument('--gwas', type=str, default=None,
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

    checkpoint = torch.load('%sbest_val_auc_model.pth' % bestmodel_path)
    test_data = h5py.File('%stest_data.h5' % bestmodel, 'r')
    test_Y = test_data["test_Y"]
    num_cell = len(test_Y[0])
    test_data.close()


    if args.kgp is not None:
        print('predict 1kgp data')
        data = h5py.File(args.kgp, 'r')
        ref_dataset = data["ref"]
        vary_dataset = data["vary"]
        from Model_smt import model
        DataLoader = torch.utils.data.DataLoader(Dataset(ref_dataset, vary_dataset), batch_size=1, shuffle=False)
        model = model(num_cell=num_cell)
        model = nn.DataParallel(model, device_ids=device_id)
        model.to(device)
        model.load_state_dict(checkpoint['best_model_state'])  

        diff_data, logit_data,product_data = test_func(model=model, DataLoader=DataLoader, device=device)
        with h5py.File('%srandom_1kgp_predict.h5' % bestmodel, 'w') as hf:
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

