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

'''

Used for predicting positive and negative data of CAQTL on corresponding models
Enter the caQTL folder:/ mnt/data0/users/lisg/Data/caqtl， Contains 2 files: positive.h5 negative0.008.h5
Enter the upper level common path of the 7 best model folders:/ mnt/data0/users/lisg/Data/brain'
Input GPU: Example: 0 1 2 3
'''

def make_parser():
    parser = configargparse.ArgParser(description="Preprocessing eqtl data")
    parser.add_argument('--model', type=str, default='/mnt/data0/users/lisg/Data/brain2')
    parser.add_argument('--gpu', type=int, nargs='+', help='gpu id')
    parser.add_argument('--gwas', type=str, default='/mnt/data0/users/lisg/Data/GWAS',)
    return parser
def main():
    parser = make_parser()
    args = parser.parse_args()
    bestmodel = ensure_trailing_slash(args.model)
    device_id = args.gpu
    model_path = [
        os.path.join(bestmodel,'acc'),os.path.join(bestmodel,'cbl'),
        os.path.join(bestmodel,'cmn'),os.path.join(bestmodel,'ic'),
        os.path.join(bestmodel,'pn'),os.path.join(bestmodel,'pul'),
        os.path.join(bestmodel,'sub')]

    use_cuda = torch.cuda.is_available()  
    device = torch.device("cuda:%s" % device_id[0] if use_cuda else "cpu")

    for Path in model_path:
        bestmodel_path = os.path.join(Path,'train_output')

        checkpoint = torch.load(os.path.join(bestmodel_path,'best_val_auc_model.pth'), map_location=device)
        test_data = h5py.File(os.path.join(Path,'test_data.h5'), 'r')
        test_Y = test_data["test_Y"]
        num_cell = len(test_Y[0])
        test_data.close()

        from Model_smt import model
        model = model(num_cell=num_cell)
        model = nn.DataParallel(model, device_ids=device_id)
        model.to(device)
        model.load_state_dict(checkpoint['best_model_state'])  

        gwas = ensure_trailing_slash(args.gwas)
        gwas_data_path = [
            os.path.join(gwas, 'AD/AD.h5'), os.path.join(gwas, 'ADHD/ADHD.h5'),
            os.path.join(gwas, 'ALS/ALS.h5'), os.path.join(gwas, 'ASD/ASD.h5'),
            os.path.join(gwas, 'BD/BD.h5'), os.path.join(gwas, 'MDD/MDD.h5'),
            os.path.join(gwas, 'MS/MS.h5'), os.path.join(gwas, 'PD/PD.h5'),
            os.path.join(gwas, 'SCZ/SCZ.h5'),os.path.join(gwas, 'Stroke/Stroke.h5'),
        ]
        for gwas_data in gwas_data_path:
 
            output_path = os.path.join(Path, 'gwas')
            file_name = os.path.splitext(os.path.basename(gwas_data))[0]
 
            with h5py.File(gwas_data, 'r') as f:
                ref_dataset = f['ref']
                vary_dataset = f["vary"]
                DataLoader = torch.utils.data.DataLoader(Dataset(ref_dataset, vary_dataset), batch_size=1,
                                                         shuffle=False)
                diff_data, logit_data, product_data = test_func(model=model, DataLoader=DataLoader, device=device)
                with h5py.File(output_path + '/' + file_name + '_predict.h5' , 'w') as hf:
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

if __name__ == "__main__":
    main()

