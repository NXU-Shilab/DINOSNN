import sys
import time
import random
import h5py
import numpy as np
import pysam

def make_bed_seqs_from_df(input_bed, fasta_file, seq_len, stranded=False):
    fasta_open = pysam.Fastafile(fasta_file)

    seqs_dna = []
    seqs_coords = []

    for i in range(input_bed.shape[0]):
        chrm = input_bed.iloc[i, 0] 
        start = int(input_bed.iloc[i, 1]) 
        end = int(input_bed.iloc[i, 2])
        strand = "+"

        # determine sequence limits
        mid = (start + end) // 2
        seq_start = mid - seq_len // 2
        seq_end = seq_start + seq_len 

        # save
        if stranded:
            seqs_coords.append((chrm, seq_start, seq_end, strand))
        else:
            seqs_coords.append((chrm, seq_start, seq_end))
        # initialize sequence
        seq_dna = ""

        if seq_start < 0:

            print(
                "Adding %d Ns to %s:%d-%s" % (-seq_start, chrm, start, end),
                file=sys.stderr,
            )
            seq_dna = "N" * (-seq_start)
            seq_start = 0

        # get dna
        seq_dna += fasta_open.fetch(chrm, seq_start, seq_end).upper()

        if len(seq_dna) < seq_len:
           

            print(
                "Adding %d Ns to %s:%d-%s" % (seq_len - len(seq_dna), chrm, start, end),
                file=sys.stderr,
            )
            seq_dna += "N" * (seq_len - len(seq_dna))
        # append
        seqs_dna.append(seq_dna)
    fasta_open.close()
    return seqs_dna, seqs_coords
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
def make_h5_sparse(atac_ad, h5_name, fasta_file, seq_len, batch_size=1000):
    t0 = time.time()
    m = atac_ad.X 
    m = m.tocoo().transpose().tocsr()
    n_peaks = atac_ad.shape[1] 
   
    bed_df = atac_ad.var.loc[:, ['chr', 'start', 'end']]
    
    bed_df.index = np.arange(bed_df.shape[0])
    
    n_batch = int(np.floor(n_peaks / batch_size))

    batches = np.array_split(np.arange(n_peaks), n_batch)
    

    ### create h5 file
    # X is a matrix of n_peaks * 1344
    f = h5py.File(h5_name, "w")
  
    ds_X = f.create_dataset("X",(n_peaks, seq_len),dtype="int8",)
  
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

