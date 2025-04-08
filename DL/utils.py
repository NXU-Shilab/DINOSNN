import os
import re
import math
import random
from typing import Optional
from copy import deepcopy
import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score, accuracy_score, f1_score
from torch import nn, Tensor, optim
import torch.nn.functional as F
import scanpy as sc
class StochasticReverseComplement(nn.Module):
    def __init__(self):
        super(StochasticReverseComplement, self).__init__()
    def forward(self, seq_1hot, training=None):
        if training:
           rc_seq_1hot = seq_1hot.index_select(1, torch.tensor([3, 2, 1, 0]))  
           rc_seq_1hot = rc_seq_1hot.flip([2]) 
           reverse_bool = torch.rand(1) > 0.5 
           src_seq_1hot = rc_seq_1hot if reverse_bool else seq_1hot 
           return src_seq_1hot, reverse_bool
        else:
           return seq_1hot, torch.tensor(False)
class StochasticShift(nn.Module):
    def __init__(self, shift_max=3, pad="uniform"):
        super(StochasticShift, self).__init__()
        self.shift_max = shift_max
        self.pad = pad
        self.augment_shifts = torch.arange(-self.shift_max, self.shift_max + 1)

    def forward(self, seq_1hot):
        if self.training:
            shift_i = torch.randint(low=0, high=len(self.augment_shifts), size=())
            shift = self.augment_shifts[shift_i]
            if shift != 0:
                sseq_1hot = shift_sequence(seq_1hot, shift)  # You need to implement this function
            else:
                sseq_1hot = seq_1hot
            return sseq_1hot
        else:
            return seq_1hot
def shift_sequence(seq_1hot, shift):
    seq =seq_1hot
    if len(seq.shape) != 3:
        raise ValueError("input sequence should be rank 3")
    input_shape = seq.shape
    # Create padding
    pad = 0.25 * torch.ones_like(seq[:, :, :abs(shift)])
    if shift > 0:
        # Shift to the right
        sliced_seq = seq[:, :,:-shift,]
        sseq = torch.cat([pad, sliced_seq], dim=2)
    else:
        # Shift to the left
        sliced_seq = seq[:,:, -shift:,]
        sseq = torch.cat([sliced_seq, pad], dim=2)

    sseq = sseq.view(input_shape)
    return sseq
class SwitchReverse(nn.Module):
    """Reverse predictions if the inputs were reverse complemented."""
    def __init__(self):
        super(SwitchReverse, self).__init__()

    def forward(self, x_reverse,):
        x = x_reverse[0]
        reverse = x_reverse[1].to(x.device)
        return torch.where(reverse, torch.flip(x, dims=[1]), x)
class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, dataset, labels,transform=None):
        'Initialization'
        self.labels = labels
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        'Denotes the total number of samples'
        return self.dataset.shape[0]

    def __getitem__(self, index):
        X = self.dataset[index]
        y = self.labels[index]
        if self.transform is not None:
            X = self.transform(X)
        return X, y

def load_data_cell(data_path):

    train_data = h5py.File('%s/train_data.h5' % data_path, 'r')
    train_X = train_data["train_X"]
    train_Y = train_data["train_Y"]
    num_cell = train_Y.shape[1]
    training_set = Dataset(train_X, train_Y)

    return training_set,num_cell

def load_data_phastcons(data_path):

    train_data = h5py.File('%strain_data.h5' % data_path, 'r')
    train_ph_X = train_data["train_ph_X"]
    train_Y = train_data["train_Y"]
    val_data = h5py.File('%sval_data.h5' % data_path, 'r')
    val_ph_X = val_data['val_ph_X']
    val_Y = val_data["val_Y"]
    num_cell = train_Y.shape[1]
    training_set = Dataset(train_ph_X, train_Y)
    validation_set = Dataset(val_ph_X, val_Y)

    return training_set, validation_set,num_cell

def fix_random(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  

class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()

        # import ipdb; ipdb.set_trace()

        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        self.patience = patience  
        self.delta = delta  
        self.verbose = verbose  
        self.counter = 0  
        self.best_score = None  
        self.early_stop = False 

    def __call__(self, metrics):
        if self.best_score is None:  
            self.best_score = metrics

        elif metrics > self.best_score + self.delta: 
            
            self.best_score = metrics
            self.counter = 0
        else:                          
            self.counter += 1
            if self.verbose:
                print(f'now_auc:{metrics},best_auc:{self.best_score},EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

class EarlyStopping_loss:
    def __init__(self, patience=5, delta=0, verbose=False):
        self.patience = patience  
        self.delta = delta 
        self.verbose = verbose  
        self.counter = 0  
        self.best_score = None  
        self.early_stop = False 

    def __call__(self, metrics):
        if self.best_score is None:   
            self.best_score = metrics

        elif metrics < self.best_score: 
            
            self.best_score = metrics
            self.counter = 0
        else:                          
            self.counter += 1
            if self.verbose:
                print(f'now_auc:{metrics},best_auc:{self.best_score},EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

def smooth_labels(labels, smoothing=0.1):
    with torch.no_grad(): 
        labels = labels * (1 - smoothing) + 0.5 * smoothing
    return labels



class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor

class AsymmetricLoss(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()

def dna_1hot_2vec(seq, seq_len=None):
    """dna_1hot
    Args:
      seq:       nucleotide sequence.
      seq_len:   length to extend/trim sequences to.
      n_uniform: represent N's as 0.25, forcing float16,
                 rather than sampling.
    Returns:
      seq_code: length by nucleotides array representation.
    """
    if seq_len is None:
        seq_len = len(seq)
        seq_start = 0
    else:
        if seq_len <= len(seq):
            # trim the sequence
            seq_trim = (len(seq) - seq_len) // 2
            seq = seq[seq_trim: seq_trim + seq_len]
            seq_start = 0
        else:
            seq_start = (seq_len - len(seq)) // 2
    seq = seq.upper()

    # map nt's to a matrix len(seq)x4 of 0's and 1's.
    seq_code = np.zeros((seq_len,), dtype="int8")

    for i in range(seq_len):
        if i >= seq_start and i - seq_start < len(seq):
            nt = seq[i - seq_start]
            if nt == "A":
                seq_code[i] = 0
            elif nt == "C":
                seq_code[i] = 1
            elif nt == "G":
                seq_code[i] = 2
            elif nt == "T":
                seq_code[i] = 3
            else:
                seq_code[i] = random.randint(0, 3)
    return seq_code

def trim_array(a, b):
    '''
    :param a:
    :param b:
    :return:
    '''
    if b == 0:
        return a
    if b % 2 == 0:
        half_b = b // 2
        return a[half_b:-half_b]
    else:
        half_b = b // 2
        return a[half_b:-half_b-1]


def calculate_metrics_with_best_threshold(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    youden_index = tpr - fpr
    best_threshold_index = np.argmax(youden_index)
    best_threshold = thresholds[best_threshold_index]
    y_pred = (y_scores >= best_threshold).astype(int)
    auc = roc_auc_score(y_true, y_scores)
    aupr = average_precision_score(y_true, y_scores)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred,average='weighted')
    # tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # specificity = tn / (tn + fp)
    # print('AUC:', round(auc, 3), 'ACC:', round(accuracy, 3), )
    # print('f1_score:', round(f1, 3), )
    return auc,aupr,accuracy,f1,

def cal_performance(D_true,D_pred,s_true,s_pred):

    D_peak_auc, D_peak_aupr, D_peak_acc, D_peak_f1 = [],[],[],[]
    s_peak_auc, s_peak_aupr, s_peak_acc, s_peak_f1 = [],[],[],[]
    for i in range(D_true.shape[0]):
        results = calculate_metrics_with_best_threshold(y_true=D_true[i, :], y_scores=D_pred[i, :])
        D_peak_auc.append(results[0])
        D_peak_aupr.append(results[1])
        D_peak_acc.append(results[2])
        D_peak_f1.append(results[3])
        results1= calculate_metrics_with_best_threshold(y_true=s_true[i, :], y_scores=s_pred[i, :])
        s_peak_auc.append(results1[0])
        s_peak_aupr.append(results1[1])
        s_peak_acc.append(results1[2])
        s_peak_f1.append(results1[3])
    peak_data = {
        'D_peak_auc': D_peak_auc,'D_peak_aupr': D_peak_aupr,'D_peak_acc': D_peak_acc,'D_peak_f1': D_peak_f1,
        's_peak_auc': s_peak_auc,'s_peak_aupr': s_peak_aupr,'s_peak_acc': s_peak_acc,'s_peak_f1': s_peak_f1}
    peak_df = pd.DataFrame(peak_data)


    D_cell_auc, D_cell_aupr, D_cell_acc, D_cell_f1 = [],[],[],[]
    s_cell_auc, s_cell_aupr, s_cell_acc, s_cell_f1 = [],[],[],[]
    for i in range(D_true.shape[1]):
        results= calculate_metrics_with_best_threshold(y_true = D_true[:, i], y_scores = D_pred[:, i])
        D_cell_auc.append(results[0])
        D_cell_aupr.append(results[1])
        D_cell_acc.append(results[2])
        D_cell_f1.append(results[3])
        results1 = calculate_metrics_with_best_threshold(y_true=s_true[:, i], y_scores=s_pred[:, i])
        s_cell_auc.append(results1[0])
        s_cell_aupr.append(results1[1])
        s_cell_acc.append(results1[2])
        s_cell_f1.append(results1[3])
    cell_data = {
        'D_cell_auc': D_cell_auc,'D_cell_aupr': D_cell_aupr,'D_cell_acc': D_cell_acc,'D_cell_f1': D_cell_f1,
        's_cell_auc': s_cell_auc,'s_cell_aupr': s_cell_aupr,'s_cell_acc': s_cell_acc,'s_cell_f1': s_cell_f1}
    cell_df = pd.DataFrame(cell_data)

    return peak_df,cell_df

def remove_suffix(s):
    return re.sub(r'_\d+$', '', re.sub(r'_\d+$', '', s))
def add_suffix_to_duplicates(strings):
    count_dict = {}
    result = []
    for s in strings:
        if s in count_dict:
            count_dict[s] += 1
            result.append(f"{s}-{count_dict[s]}")
        else:
            count_dict[s] = 1
            result.append(s)
    return result

def analyze_cell(cell_name):
    counts = cell_name.value_counts()
    unique_count = cell_name.nunique()
    print("The number of each type of cell:",counts)
    print("How many types of cells are there in total:",unique_count)
    counts_df = pd.DataFrame(counts)
    counts_df.columns = ['count']
    # print(counts_df.index.tolist())
    # counts_df.to_csv('/mnt/data0/users/lisg/Data/counts2.csv')
    return counts_df