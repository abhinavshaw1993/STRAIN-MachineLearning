#!/bin/bash
#
#SBATCH --mem=60000
#SBATCH --job-name=1-gpu-STRAIN-custom
#SBATCH --partition=1080ti-short
#SBATCH --output=STRAIN-loso_log_reg_loso_mul_gpu-%A.out
#SBATCH --error=STRAIN-loso_log_reg_loso_mul_gpu-%A.err
#SBATCH --gres=gpu:6
#SBATCH --mail-type=ALL
#SBATCH --mail-user=abhinavshaw@umass.edu

# Log the jobid.
echo $SLURM_JOBID - `hostname` >> ~/slurm-jobs.txt

# Chage Dir to SRC.
cd ~/projects/STRAIN-MachineLearning/src/main/train
PYTHONPATH=../ python train_lstm_log_reg_loso_mulgpu.py
