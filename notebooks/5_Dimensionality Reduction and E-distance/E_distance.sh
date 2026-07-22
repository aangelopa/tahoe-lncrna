#!/bin/bash
#SBATCH --job-name=e_dist
#SBATCH --partition=a100
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --array=0-13
#SBATCH --output=e_dist_%A_%a.out
#SBATCH --error=e_dist_%A_%a.err

module load gcc/14.2.0 python/3.13.8 cuda/12.8.1
source ~/envs/rapids/bin/activate

python ~/Thesis/Code/e_distance.py --plate_idx $SLURM_ARRAY_TASK_ID