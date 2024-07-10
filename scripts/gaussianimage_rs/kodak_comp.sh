#!/bin/bash

data_path=$1

if [ -z "$data_path" ]; then
    echo "Error: No data_path provided."
    echo "Usage: $0 <data_path>"
    exit 1
fi

for num_points in 800 1000 3000 5000 7000 9000
do
CUDA_VISIBLE_DEVICES=0 python train_quantize.py -d $data_path \
--data_name kodak --model_name GaussianImage_RS --num_points $num_points --iterations 50000 \
--model_path ./checkpoints/kodak/GaussianImage_RS_50000_$num_points
done

for num_points in 800 1000 3000 5000 7000 9000
do
CUDA_VISIBLE_DEVICES=0 python test_quantize.py -d $data_path \
--data_name kodak --model_name GaussianImage_RS --num_points $num_points --iterations 50000 \
--model_path ./checkpoints_quant/kodak/GaussianImage_50000_RS_$num_points
done
