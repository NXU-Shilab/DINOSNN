# **DINOSNN(The code and tutorial are being uploaded, please wait.**)

Welcome to the `DINOSNN` framework repository! `DINOSNN` is a computational framework based on chromatin accessibility perturbation modeling to decode cell type-specific regulatory effects of noncoding variants using single-cell ATAC-seq data (scATAC-seq). DINOSNN consists of two components: the first employs a deep neural network to model scATAC-seq profiles and predict single-cell chromatin accessibility. The second component identifies functional noncoding single-nucleotide polymorphisms (SNPs) by quantifying chromatin accessibility differences between reference and variant sequences, enabling precise mapping of SNP-induced cell type- and region-specific regulatory effects in the brain.

<img src alt="DINOSNN Framework" width="350">

## **Requirements**

You may create an anaconda environment for `DINOSNN` with the following commands:

```bash
git clone https://github.com/NXU-Shilab/DINOSNN.git
cd DINOSNN
conda env create -f environment.yml
conda activate DINOSNN
```

Note: `DINOSNN` is designed to run on a GPU-enabled environment. Our experiments were conducted on a Linux server with an Intel(R) Xeon(R) Silver 4316 CPU, 512 GB system memory, and an NVIDIA GeForce RTX 3090 GPU (24 GB VRAM). We recommend running DINOSNN on a system with at least `128GB RAM` and an `NVIDIA GPU with >= 24 GB VRAM`. We also provide a full snapshot of the conda environment used to run DINOSNN. Because this snapshot includes all exact package versions and dependencies, creating the conda environment from it may take some time.

## **Part I. Single-cell Accessibility Modeling**

### **Datasets**

\*\*1.Using the datasets from our paper. \*\*We provide preprocessed `h5ad` files used in our experiments:\
<https://drive.google.com/drive/folders/1eW0fZL_XXeg98cLkMSD4Pkw-zCGjpGVO?usp=drive_link>

\*\*2.Using your own scATAC dataset. \*\*You can also use your own scATAC-seq dataset, as long as it is stored in `h5ad` format and meets the following requirements:

`adata.var` must contain at least three columns:

*   `'chr'`: chromosome of each peak
*   `'start'`: genomic start position of each peak
*   `'end'`: genomic end position of each peak

`adata.obs` must contain a `'celltype'` column indicating the cell type of each cell.

### **Data Preparation**

Data preparation requires an AnnData (h5ad) file, the human reference genome (hg38), and human PhastCons conservation score files (phastCons100way). The reference genome can be downloaded from: <https://hgdownload.soe.ucsc.edu/downloads.html>  The conservation score files can be downloaded from: <http://hgdownload.cse.ucsc.edu/goldenpath/hg38/phastCons100way/>  Run the following file after completion:

```python
cd PartI
python preprocess_data.py --ad <h5ad_file> --output <output_path> --fa <hg38_file> --pha <PhastCons_file>
```

Arguments:

*   `--ad`: scATAC dataset.
*   `--output`: (Not necessary) folder for saving training and testing data. By default, a 'processed\_data' folder is created under the DINOSNN folder for storage.
*   `--fa`: human reference genome (hg38).
*   `--pha`: human PhastCons conservation score files.

### **Train**

To perform train, simply run the following command:

```python
python train.py --data <data_folder> --gpu <GPU_id>
```

Note: Using the datasets from the paper, each dataset converges in roughly 12 hours of training.

Arguments:

*   `--data`: (Not necessary) data folder. By default, the folder generated in the **Data Preparation** step is used.
*   `--gpu`: (Not necessary) GPU id. By default, training runs on GPU 0. You can also enable multi-GPU training by specifying multiple device IDs, e.g. `--gpu 0,1,2,3`.

### **Predict**

Using the best model in `processed_data/train_output/` to evaluate AUROC per peak and per cell on the test set:

```python
python predict.py --data <data_folder> --gpu <GPU_id>
```

The prediction step generates a `processed_data/train_output/pred.npy` file, which stores the prediction results for each sample in the test set.

## **Part II. Cell-specific Perturbation**
