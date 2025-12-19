import os.path
import pysam
import random
from copy import deepcopy
import sys
import os
import time
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, accuracy_score, f1_score
import torch
import h5py
from torch import nn
import numpy as np
from tqdm import tqdm

def make_h5_sparse(atac_ad, h5_name, fasta_file, seq_len, batch_size=1000):
    t0 = time.time()
    m = atac_ad.X
    m = m.tocoo().transpose().tocsr()
    n_peaks = atac_ad.shape[1]
    bed_df = atac_ad.var.loc[:, ['chr', 'start', 'end']]
    bed_df.index = np.arange(bed_df.shape[0])
    n_batch = int(np.floor(n_peaks / batch_size))
    batches = np.array_split(np.arange(n_peaks), n_batch)
    f = h5py.File(h5_name, "w")
    ds_X = f.create_dataset(name="all_seqs",shape=(n_peaks, seq_len),dtype="int8",)
    for i in range(len(batches)):
        idx = batches[i]
        seqs_dna, _ = make_bed_seqs_from_df(
            bed_df.iloc[idx, :],
            fasta_file=fasta_file,
            seq_len=seq_len,
        )
        dna_array_dense = [dna_1hot_2vec(x) for x in seqs_dna]
        dna_array_dense = np.array(dna_array_dense)
        ds_X[idx] = dna_array_dense
        t1 = time.time()
        total = t1 - t0
        print('process %d peaks takes %.1f s' % (i * batch_size, total))
    f.close()

def split_train_test_val(ids, seed, train_ratio):
    np.random.seed(seed)
    test_val_ids = np.random.choice(ids,int(len(ids) * (1 - train_ratio)),replace=False,)
    train_ids = np.setdiff1d(ids, test_val_ids)
    val_ids = np.random.choice(test_val_ids,int(len(test_val_ids) / 2),replace=False,)
    test_ids = np.setdiff1d(test_val_ids, val_ids)
    return train_ids, test_ids, val_ids

def get_conservation_scores(chr, start, end, consbw):
    try:
        if start >= end:
            raise ValueError(f"Invalid interval: start ({start}) must be less than end ({end})")
        values = consbw.values(chr, start, end)
        if values is None:
            print(f"No data available for the specified region: {chr}:{start}-{end}")
            return np.full(end - start, np.nan)
        return values
    except ValueError as ve:
        print(f"ValueError: {ve}")
        return np.full(end - start, np.nan)
    except RuntimeError as re:
        print(f"RuntimeError: {re}")
        return np.full(end - start, np.nan)
    except Exception as e:
        print(f"Unexpected error: {e}")
        return np.full(end - start, np.nan)

def conservation_conv(values, a, b):
    values = np.array(values)
    if np.all(np.isnan(values)):
        values = np.nan_to_num(values)
        a += 1
    else:
        nan_indices = np.where(np.isnan(values))[0]
        average_score = np.nanmean(values)
        values[nan_indices] = average_score
        b += 1
    return np.exp(values), a, b

def make_bed_seqs_from_df(input_bed, fasta_file, seq_len, stranded=False):
    fasta_open = pysam.Fastafile(fasta_file)
    seqs_dna = []
    seqs_coords = []

    for i in range(input_bed.shape[0]):
        chrm = input_bed.iloc[i, 0]
        start = int(input_bed.iloc[i, 1])
        end = int(input_bed.iloc[i, 2])
        strand = "+"
        mid = (start + end) // 2
        seq_start = mid - seq_len // 2
        seq_end = seq_start + seq_len
        if stranded:
            seqs_coords.append((chrm, seq_start, seq_end, strand))
        else:
            seqs_coords.append((chrm, seq_start, seq_end))
        seq_dna = ""

        if seq_start < 0:
            print(
                "Adding %d Ns to %s:%d-%s" % (-seq_start, chrm, start, end),
                file=sys.stderr,
            )
            seq_dna = "N" * (-seq_start)
            seq_start = 0
        seq_dna += fasta_open.fetch(chrm, seq_start, seq_end).upper()

        if len(seq_dna) < seq_len:
            print(
                "Adding %d Ns to %s:%d-%s" % (seq_len - len(seq_dna), chrm, start, end),
                file=sys.stderr,
            )
            seq_dna += "N" * (seq_len - len(seq_dna))
        seqs_dna.append(seq_dna)
    fasta_open.close()
    return seqs_dna, seqs_coords
def dna_1hot_2vec(seq, seq_len=None):
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


def fix_random(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def load_data(data_path):
    train_data = h5py.File(os.path.join(data_path, 'train_data.h5'), 'r')
    train_ph_X = train_data["train_ph_X"]
    train_Y = train_data["train_Y"]
    val_data = h5py.File(os.path.join(data_path, 'val_data.h5'), 'r')
    val_ph_X = val_data['val_ph_X']
    val_Y = val_data["val_Y"]
    num_cell = train_Y.shape[1]
    training_set = Dataset(train_ph_X, train_Y)
    validation_set = Dataset(val_ph_X, val_Y)
    return training_set, validation_set,num_cell

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset, labels,transform=None):
        self.labels = labels
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, index):
        X = self.dataset[index]
        y = self.labels[index]
        if self.transform is not None:
            X = self.transform(X)
        return X, y


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
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
        sliced_seq = seq[:, :,:-shift,]
        sseq = torch.cat([pad, sliced_seq], dim=2)
    else:
        sliced_seq = seq[:,:, -shift:,]
        sseq = torch.cat([sliced_seq, pad], dim=2)

    sseq = sseq.view(input_shape)
    return sseq
class SwitchReverse(nn.Module):
    def __init__(self):
        super(SwitchReverse, self).__init__()

    def forward(self, x_reverse,):
        x = x_reverse[0]
        reverse = x_reverse[1].to(x.device)
        return torch.where(reverse, torch.flip(x, dims=[1]), x)

