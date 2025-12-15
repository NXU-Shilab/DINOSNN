# **DINOSNN**

Welcome to the `DINOSNN` framework repository! `DINOSNN` is a computational framework based on chromatin accessibility perturbation modeling to decode cell type-specific regulatory effects of noncoding variants using single-cell ATAC-seq data (scATAC-seq). DINOSNN consists of two components: the first employs a deep neural network to model scATAC-seq profiles and predict single-cell chromatin accessibility. The second component identifies functional noncoding single-nucleotide polymorphisms (SNPs) by quantifying chromatin accessibility differences between reference and variant sequences, enabling precise mapping of SNP-induced cell type- and region-specific regulatory effects in the brain.

<img src alt="DINOSNN Framework" width="350">

# **Requirements**

You may create an anaconda environment for `DINOSNN` with the following commands:

```bash
git clone https://github.com/NXU-Shilab/DINOSNN.git
cd DINOSNN
conda env create -f environment.yml
conda activate DINOSNN
```

Note: `DINOSNN` is designed to run on a GPU-enabled environment. Our experiments were conducted on a Linux server with an Intel(R) Xeon(R) Silver 4316 CPU, 512 GB system memory, and an NVIDIA GeForce RTX 3090 GPU (24 GB VRAM). We recommend running DINOSNN on a system with at least 128 GB RAM and an NVIDIA GPU with ≥ 24 GB VRAM. We also provide a full snapshot of the conda environment used to run DINOSNN. Because this snapshot includes all exact package versions and dependencies, creating the conda environment from it may take some time.
