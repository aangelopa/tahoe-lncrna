#!/bin/bash

#SBATCH --job-name=vision
#SBATCH --partition=a100
#SBATCH --time=3-12:00:00
#SBATCH --gpus=1
#SBATCH --mem=300G
#SBATCH --output=leiden3list_%j.out
#SBATCH --error=leiden3list_%j.err



module load gcc/14.2.0 python/3.13.8 cuda/12.8.1
source ~/envs/rapids/bin/activate


python /home/aangelopa/Thesis/Code/leiden_3lists_2ndtry.py
