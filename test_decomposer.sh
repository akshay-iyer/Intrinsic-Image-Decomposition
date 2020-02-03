#!/bin/bash
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --mem 128G
#SBATCH --gres=gpu:1
#SBATCH -p long
#SBATCH -o test_decomposer_160919.stdout
#SBATCH -e test_decomposer_160919.stderr
#SBATCH -t 108:00:00
#SBATCH -J decomp_wound

source /home/abiyer/rin_env/bin/activate

python infer_decomposer.py
