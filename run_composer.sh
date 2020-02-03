#!/bin/bash
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --mem 128G
#SBATCH --gres=gpu:1
#SBATCH -p emmanuel 
#SBATCH -o composer_cl.stdout
#SBATCH -e composer_cl.stderr
#SBATCH -t 108:00:00
#SBATCH -J composer_cl

source /home/abiyer/rin_env/bin/activate

python composer.py --decomposer saved/decomposer/decomp_cl/state_cl.t7 --shader saved/shader/model.t7 --save_path saved/composer/composer_cl_again \
		   --unlabeled suzanne_train,airplane_train,bunny_train \
		   --labeled motorbike_train,car_train,bottle_train \
		   --val_sets suzanne_val,bunny_val,airplane_val,motorbike_val,car_val \
		   --unlabeled_array shader --labeled_array shader \
		   --transfer 10-normals,reflectance_20-lights --num_epochs 100 --save_model True --data_path ~/intrinsics-network/dataset/output --set_size 19900 \
		   --num_val 3

#--transfer 10-shader_10-normals,reflectance_20-lights
