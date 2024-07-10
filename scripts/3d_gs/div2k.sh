#!/bin/bash

data_path=$1

if [ -z "$data_path" ]; then
    echo "Error: No data_path provided."
    echo "Usage: $0 <data_path>"
    exit 1
fi

for num_points in 5000 10000 15000 20000 25000 30000 50000 60000 70000
do
CUDA_VISIBLE_DEVICES=0 python train.py -d $data_path \
--data_name DIV2K_valid_LRX2 --model_name 3DGS --num_points $num_points --iterations 50000 --sh_degree 3
done

