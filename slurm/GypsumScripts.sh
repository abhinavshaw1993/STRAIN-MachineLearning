#!/bin/bash
#
#SBATCH --mem=30000
#SBATCH --job-name=1-gpu-STRAIN-custom
#SBATCH --partition=m40-short
#SBATCH --output=STRAIN-custom-%A.out
#SBATCH --error=STRAIN-custom-%A.err
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=abhinavshaw@umass.edu

# Log the jobid.
echo $SLURM_JOBID - `hostname` >> ~/slurm-jobs.txt

# Chage Dir to SRC.
cd ~/projects/STRAIN-MachineLearning/src/main
PYTHONPATH=../ python train.py


