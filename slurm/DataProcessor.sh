#!/bin/bash
#
#SBATCH --mem=60000
#SBATCH --job-name=1-gpu-STRAIN-custom
#SBATCH --partition=m40-long
#SBATCH --output=DataProcessor-%A.out
#SBATCH --error=DataProcessor-%A.err
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=abhinavshaw@umass.edu

# Log the jobid.
echo $SLURM_JOBID - `hostname` >> ~/slurm-jobs.txt

# Chage Dir to SRC.
cd ~/projects/STRAIN-MachineLearning/src/main/data_processor
PYTHONPATH=../../ python GenerateFixedSeqNumpy.py
