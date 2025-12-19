import configargparse
import sys
import os
import torch
import h5py
from torch import nn
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

def make_parser():
    parser = configargparse.ArgParser("")
    parser.add_argument('--bestmodel', type=str,default='../processed_data')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0])
    parser.add_argument('--kgp', type=str, default=None)
    parser.add_argument('--caqtl', type=str, default=None)
    return parser

def main():
    parser = make_parser()
    args = parser.parse_args()
    bestmodel_path = os.path.join(args.bestmodel, 'train_output')
    device_id = args.gpu
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:%s" % device_id[0] if use_cuda else "cpu")
    checkpoint = torch.load(os.path.join(bestmodel_path, 'best_val_auc_model.pth'))
    test_data = h5py.File(os.path.join(args.bestmodel, 'test_data.h5'), 'r')
    test_Y = test_data["test_Y"]
    num_cell = len(test_Y[0])
    test_data.close()


    if args.kgp is not None:
        data = h5py.File(args.kgp, 'r')
        ref_dataset = data["ref"]
        vary_dataset = data["vary"]
        from Model import model
        DataLoader = torch.utils.data.DataLoader(Dataset(ref_dataset, vary_dataset), batch_size=1, shuffle=False)
        model = model(num_cell=num_cell)
        model = nn.DataParallel(model, device_ids=device_id)
        model.to(device)
        model.load_state_dict(checkpoint['best_model_state'])  # 加载模型参数到模型

        product_data = test_func(model=model, DataLoader=DataLoader, device=device)
        with h5py.File(os.path.join(os.path.dirname(args.kgp), 'random_1kgp_predict.h5'), 'w') as hf:
            hf.create_dataset("product_data", data=product_data)

    if args.caqtl is not None:
        from Model import model
        model = model(num_cell=num_cell)
        model = nn.DataParallel(model, device_ids=device_id)
        model.to(device)
        model.load_state_dict(checkpoint['best_model_state'])  # 加载模型参数到模型

        positive = os.path.join(args.caqtl, 'positive.h5')
        negative = os.path.join(args.caqtl, 'negative0.008.h5')

        with h5py.File(positive, 'r') as f:
            ref_dataset = f['ref']
            vary_dataset = f["vary"]
            DataLoader = torch.utils.data.DataLoader(Dataset(ref_dataset, vary_dataset), batch_size=1, shuffle=False)
            product_data = test_func(model=model, DataLoader=DataLoader, device=device)
            with h5py.File(os.path.join(args.caqtl,'positive_predict.h5'), 'w') as hf:
                hf.create_dataset("product_data", data=product_data)
        with h5py.File(negative, 'r') as f:
            ref_dataset = f['ref']
            vary_dataset = f["vary"]
            DataLoader = torch.utils.data.DataLoader(Dataset(ref_dataset, vary_dataset), batch_size=1, shuffle=False)
            product_data = test_func(model=model, DataLoader=DataLoader, device=device)
            with h5py.File(os.path.join(args.caqtl,'negative_predict.h5'), 'w') as hf:
                hf.create_dataset("product_data", data=product_data)

def test_func(model, DataLoader, device):
    model.eval()
    product_data = []
    sig = nn.Sigmoid()
    with torch.no_grad():
        val_bar = tqdm(DataLoader, file=sys.stdout)
        for id, (ref, vary) in enumerate(val_bar):
            ref = ref.to(device, non_blocking=True)
            vary = vary.to(device, non_blocking=True)
            ref_predict = sig(model(ref))
            vary_predict = sig(model(vary))
            product_abs = torch.abs((ref_predict - vary_predict) * (torch.log(ref_predict / (1 - ref_predict + 1e-12)) - torch.log(vary_predict / (1 - vary_predict + 1e-12))))
            product_data.append(product_abs.cpu())
        product_data = torch.cat(product_data, 0).numpy()
    return product_data



if __name__ == "__main__":
    main()

