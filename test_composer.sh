#!/bin/bash
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --mem 128G
#SBATCH --gres=gpu:1
#SBATCH -p emmanuel
#SBATCH -o test_composer_160919.stdout
#SBATCH -e test_composer_160919.stderr
#SBATCH -t 108:00:00
#SBATCH -J comp_wound

source /home/abiyer/rin_env/bin/activate

python infer_composer.py
