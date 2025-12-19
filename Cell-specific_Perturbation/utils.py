import numpy as np
import random
import os
import h5py
import pandas as pd
import re
from torch import nn
import torch
from tqdm import tqdm
from scipy import sparse
import pysam
def check_array_length(array, expected_length):
    if isinstance(array, np.ndarray):
        if np.isnan(array).any():
            print("The array contains NaN values.")
        if array.shape[0] != expected_length:
            print(f"NumPy matrix has length {array.shape[0]}, which is not {expected_length}")
    elif isinstance(array, list):
        if any(np.isnan(x) for x in array):
            print("The array contains NaN values.")
        if len(array) != expected_length:
            print(f"Array has length {len(array)}, which is not {expected_length}")
def get_conservation_scores(chr, start, end,consbw):
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

def conservation_conv(values):
    values = np.array(values)
    if np.all(np.isnan(values)):
        values = np.nan_to_num(values)
    else:
        nan_indices = np.where(np.isnan(values))[0]
        average_score = np.nanmean(values)
        values[nan_indices] = average_score
    return np.exp(values)

def trim_array(a, b):
    if b == 0:
        return a
    if b % 2 == 0:
        half_b = b // 2
        return a[half_b:-half_b]
    else:
        half_b = b // 2
        return a[half_b:-half_b-1]

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

def extract_af(info):
    match = re.search(r'AF=([^;]+)', info)
    if match:
        return match.group(1)
    return None

def make_dataset(metrics,fasta_file,consbw,seqlen,output_name=None,output_path=None):
    fasta_open = pysam.Fastafile(fasta_file)
    with h5py.File(os.path.join(output_path, output_name + '.h5'), "w") as f:
        ref_dset = f.create_dataset("ref", shape=(metrics.shape[0], 4, seqlen), dtype=np.float32)
        vary_dset = f.create_dataset("vary", shape=(metrics.shape[0], 4, 2688), dtype=np.float32)

        for i in tqdm(range(metrics.shape[0])):
            chrm = metrics['chr'][i]
            pos = int(metrics['pos'][i]) - 1
            if len(metrics['ref'][i]) == 1:
                if len(metrics['alt'][i]) == 1:
                    ref_l = fasta_open.fetch(chrm, int(pos - 1344), pos).upper()
                    ref_r = fasta_open.fetch(chrm, pos + 1, pos + 1 + 1343).upper()
                    refseq_dna = ref_l + metrics['ref'][i] + ref_r
                    varseq_dna = ref_l + metrics['alt'][i] + ref_r
                    values_l = get_conservation_scores(chrm, int(pos + 1 - 1344), int(pos + 1), consbw=consbw)
                    value = get_conservation_scores(chrm, int(pos + 1), int(pos + 2), consbw=consbw)
                    values_r = get_conservation_scores(chrm, int(pos + 2), int(pos + 2 + 1343), consbw=consbw)
                    values_l = conservation_conv(values_l)
                    value = conservation_conv(value)
                    values_r = conservation_conv(values_r)
                    cons_values = np.concatenate((values_l, value, values_r))
                else:
                    ref_l = fasta_open.fetch(chrm, int(pos - 1344), pos).upper()
                    ref_r = fasta_open.fetch(chrm, pos + 1, pos + 1 + 1343).upper()
                    refseq_dna = ref_l + metrics['ref'][i] + ref_r
                    vary_length = len(metrics['alt'][i])
                    varseq_dna = ref_l + metrics['alt'][i] + ref_r
                    varseq_dna = trim_array(varseq_dna, vary_length - 1)
                    values_l = get_conservation_scores(chrm, int(pos + 1 - 1344), int(pos + 1),consbw=consbw)
                    value = get_conservation_scores(chrm, int(pos + 1), int(pos + 2),consbw=consbw)
                    value = [value[0]] * vary_length
                    values_r = get_conservation_scores(chrm, int(pos + 2), int(pos + 2 + 1343),consbw=consbw)
                    values_l = conservation_conv(values_l)
                    value = conservation_conv(value)
                    values_r = conservation_conv(values_r)
                    cons_values = np.concatenate((values_l, value, values_r))   # 保守性评分
                    cons_values = trim_array(cons_values, vary_length - 1)
            else:
                ref_length = len(metrics['ref'][i])
                refstart = int(pos - 1344)
                refend = int(pos + ref_length + 1343)
                ref_l = fasta_open.fetch(chrm, refstart, pos).upper()
                ref_r = fasta_open.fetch(chrm, pos + ref_length, refend).upper()
                refseq_dna = ref_l + metrics['ref'][i] + ref_r
                refseq_dna = trim_array(refseq_dna, ref_length - 1)
                varseq_dna = ref_l + metrics['alt'][i] + ref_r
                vary_length = len(metrics['alt'][i])
                varseq_dna = trim_array(varseq_dna, vary_length - 1)
                values_l = get_conservation_scores(chrm, int(refstart + 1), int(pos + 1),consbw=consbw)
                value = get_conservation_scores(chrm, int(pos + 1), int(pos + 2),consbw=consbw)
                value = [value[0]] * ref_length
                values_r = get_conservation_scores(chrm, int(pos + 1 + ref_length), int(refend + 1),consbw=consbw)
                values_l = conservation_conv(values_l)
                value = conservation_conv(value)
                values_r = conservation_conv(values_r)
                cons_values = np.concatenate((values_l, value, values_r))
                cons_values = trim_array(cons_values, ref_length - 1)
            check_array_length(refseq_dna, seqlen)
            check_array_length(varseq_dna, seqlen)
            check_array_length(cons_values, seqlen)
            ref_dna_dense = dna_1hot_2vec(refseq_dna)
            vary_dna_dense = dna_1hot_2vec(varseq_dna)
            ref = sparse.coo_matrix((np.ones(seqlen), (np.arange(seqlen), ref_dna_dense)), shape=(seqlen, 4),dtype='int8')
            vary = sparse.coo_matrix((np.ones(seqlen), (np.arange(seqlen), vary_dna_dense)), shape=(seqlen, 4),dtype='int8')
            ref_dset[i] = (ref.toarray().transpose() * cons_values).astype(np.float32)
            vary_dset[i] = (vary.toarray().transpose() * cons_values).astype(np.float32)



def extract_values(row):
    parts = row['rsid_all'].split('_')
    rsid = parts[1]
    ref = parts[2]
    alt = parts[3]
    return pd.Series([rsid, ref, alt])


def extract_values2(row):
    parts = row['pos_all'].split(':')
    chr_value = parts[0]
    start_pos = int(parts[1].split('-')[0])
    new_pos = start_pos + 135
    return pd.Series([chr_value, new_pos])


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