

for num_points in 800 1000 3000 5000 7000 9000
do
CUDA_VISIBLE_DEVICES=6 python train_quantize.py -d /home/xzhangga/dataset/kodak \
--data_name kodak --model_name GaussianImage_Cholesky --num_points $num_points --iterations 50000 \
--model_path ./checkpoints/kodak/GaussianImage_Cholesky_50000_$num_points
done

for num_points in 800 1000 3000 5000 7000 9000
do
CUDA_VISIBLE_DEVICES=6 python test_quantize.py -d /home/xzhangga/dataset/kodak \
--data_name kodak --model_name GaussianImage_Cholesky --num_points $num_points --iterations 50000 \
--model_path ./checkpoints_quant/kodak/GaussianImage_Cholesky_50000_$num_points
done
