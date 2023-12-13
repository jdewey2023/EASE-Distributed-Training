#!/bin/bash
#SBATCH --job-name=chunk data
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=42G
#SBATCH --time=01:00:00
#SBATCH --error=log/%J.err
#SBATCH --output=log/%J.out

export WORLDSIZE=1

srun python3 chunker.py