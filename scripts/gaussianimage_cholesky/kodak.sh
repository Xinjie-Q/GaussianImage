

for num_points in 800 #1000 3000 5000 7000 9000
do
CUDA_VISIBLE_DEVICES=0 python train.py -d /home/xzhangga/dataset/kodak \
--data_name kodak --model_name GaussianImage_Cholesky --num_points $num_points --iterations 50000
done
