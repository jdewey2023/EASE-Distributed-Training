#!/bin/bash
#SBATCH --job-name=ease_training
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=5
#SBATCH --ntasks=10
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00
#SBATCH --error=log/%J.err
#SBATCH --output=log/%J.out

export WORLDSIZE=10

srun python3 trainer.py