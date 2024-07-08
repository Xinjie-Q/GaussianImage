
for num_points in 5000 10000 15000 20000 25000 30000 50000 60000 70000
do
CUDA_VISIBLE_DEVICES=6 python train.py -d /home/xzhangga/dataset/kodak \
--data_name kodak --model_name 3DGS --num_points $num_points --iterations 50000 --sh_degree 3
done

