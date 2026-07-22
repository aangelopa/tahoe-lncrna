#!/bin/bash
#SBATCH --job-name=leiden3
#SBATCH --nodes=1
#SBATCH --partition=ampere
#SBATCH --gpus=1
#SBATCH --mem=300G
#SBATCH --time=15:00:00
#SBATCH --qos=ampere-extd


source /home/a/aangelopa/envs/scvi02/bin/activate
srun python3 /home/a/aangelopa/Thesis/Code/leiden_3lists_2ndtry.py \
    --job_id $SLURM_ARRAY_TASK_ID