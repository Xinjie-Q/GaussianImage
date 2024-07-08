

for num_points in 2000 4000 6000 8000 10000 12000 14000
do
CUDA_VISIBLE_DEVICES=6 python train_quantize.py -d /home/xzhangga/dataset/DIV2K_valid_LR_bicubic/X2 \
--data_name DIV2K_valid_LRX2 --model_name GaussianImage_RS --num_points $num_points --iterations 50000 \
--model_path ./checkpoints/DIV2K_valid_LRX2/GaussianImage_RS_50000_$num_points
done

for num_points in 2000 4000 6000 8000 10000 12000 14000
do
CUDA_VISIBLE_DEVICES=6 python test_quantize.py -d /home/xzhangga/dataset/DIV2K_valid_LR_bicubic/X2 \
--data_name DIV2K_valid_LRX2 --model_name GaussianImage_RS --num_points $num_points --iterations 50000 \
--model_path ./checkpoints_quant/DIV2K_valid_LRX2/GaussianImage_RS_50000_$num_points
done