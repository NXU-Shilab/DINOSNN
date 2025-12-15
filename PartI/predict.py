import os.path
import sys

import configargparse
import torch
import h5py
from sklearn.metrics import roc_auc_score
from torch import nn
import numpy as np
from tqdm import tqdm

from utils import Dataset
def make_parser():
    parser = configargparse.ArgParser(description="")
    parser.add_argument('--data', type=str, default='../processed_data')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0])
    return parser

def main():
    parser = make_parser()
    args = parser.parse_args()
    output_path = os.path.join(args.data, 'train_output')
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:%s" % args.gpu[0] if use_cuda else "cpu")
    test_data = h5py.File(os.path.join(args.data, 'test_data.h5'), 'r')
    test_Y = test_data["test_Y"]
    test_ph = test_data["test_ph_X"]
    num_cell = len(test_Y[0])
    test_set = Dataset(test_ph, test_Y)
    test_DataLoader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
    checkpoint = torch.load('%s/best_val_auc_model.pth' % output_path, map_location=device)
    print('epochs:', checkpoint['epoch'])
    print('val_auc:', checkpoint['val_auc'])
    from Model import model
    model = model(num_cell=num_cell)
    model = nn.DataParallel(model, device_ids=args.gpu)
    model.to(device)
    model.load_state_dict(checkpoint['best_model_state'])  # 加载模型参数到模型
    model.eval()
    true, pred = [], []
    cell_auc, peak_auc, cell_aupr, peak_aupr = [], [], [], []
    with torch.no_grad():
        val_bar = tqdm(test_DataLoader, file=sys.stdout)
        for val_id, (x, y) in enumerate(val_bar):
            sig = nn.Sigmoid()
            x = x.to(device, non_blocking=True, dtype=torch.float32)
            y = y.to(device, non_blocking=True, dtype=torch.float32)
            output = model(x)
            new_out = sig(output)
            true.append(y.detach().cpu())
            pred.append(new_out.detach().cpu())
        true = torch.cat(true, 0).numpy()
        pred = torch.cat(pred, 0).numpy()
        for i in range(true.shape[0]):
            peak_auc.append(roc_auc_score(y_true=true[i, :], y_score=pred[i, :]))
        for i in range(true.shape[1]):
            cell_auc.append(roc_auc_score(y_true=true[:, i], y_score=pred[:, i]))
        print('auROC per peak:', sum(peak_auc) / len(peak_auc))
        print('auROC per cell:', sum(cell_auc) / len(cell_auc))

    np.save(os.path.join(output_path, 'true.npy'), true)


if __name__ == "__main__":
    main()

