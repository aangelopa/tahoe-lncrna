#!/bin/bash
#SBATCH --job-name=explore_lnc
#SBATCH --partition=a100
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=explore_lnc_%A_%a.out
#SBATCH --error=explore_lnc_%A_%a.err

module load gcc/14.2.0 python/3.13.8 cuda/12.8.1
source ~/envs/rapids/bin/activate
python ~/Thesis/Code/exploratory_lncrna.py