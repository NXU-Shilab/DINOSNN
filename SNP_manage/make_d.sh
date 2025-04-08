#!/bin/bash

python handle_eqtl.py --eqtl "/mnt/data0/users/lisg/Data/eqtl/eqtl_acc/coding_filter.csv"


python Generate_dataset.py --eqtl "/mnt/data0/users/lisg/Data/eqtl/eqtl_acc/"

python /mnt/data0/users/lisg/Project_one/Brain/DL_train/eqtl_predict.py --bestmodel '/mnt/data0/users/lisg/Data/brain/acc' --eqtl '/mnt/data0/users/lisg/Data/eqtl/eqtl_acc/' --gpu_id 6 7