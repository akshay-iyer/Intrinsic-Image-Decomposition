#!/bin/bash
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --mem 256G
#SBATCH --gres=gpu:2
#SBATCH -p emmanuel 
#SBATCH -o rin_cl.stdout
#SBATCH -e rin_cl.stderr
#SBATCH -t 108:00:00
#SBATCH -J rin_cl

source /home/abiyer/rin_env/bin/activate

python decomposer.py --data_path ~/intrinsics-network/dataset/output --save_path saved/decomposer/decomp_cl --num_epochs 500 --array shader --num_train 19900 \
		     --num_val 20 --train_sets motorbike_train,car_train,bottle_train --val_set motorbike_val,car_val,bottle_val

