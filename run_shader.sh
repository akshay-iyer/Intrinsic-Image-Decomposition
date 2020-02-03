#!/bin/bash
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --mem 128G
#SBATCH --gres=gpu:1
#SBATCH -p long 
#SBATCH -o shader.stdout
#SBATCH -e shader.stderr
#SBATCH -t 108:00:00
#SBATCH -J shader

source /home/abiyer/rin_env/bin/activate

python shader.py --data_path ~/intrinsics-network/dataset/output --save_path saved/shader --num_train 10000 --num_val 20 --train_sets motorbike_train,car_train,bottle_train --val_set  motorbike_val,car_val,bottle_val

