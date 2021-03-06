#!/bin/bash
#
#SBATCH --mem=30000
#SBATCH --job-name=1-gpu-STRAIN-custom
#SBATCH --partition=m40-long
#SBATCH --output=STRAIN-lin_reg-%A.out
#SBATCH --error=STRAIN-lin_reg-%A.err
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=abhinavshaw@umass.edu

# Log the jobid.
echo $SLURM_JOBID - `hostname` >> ~/slurm-jobs.txt

# Chage Dir to SRC.
cd ~/projects/STRAIN-MachineLearning/src/main/train
PYTHONPATH=../ python train_lstm_linear_reg.py


