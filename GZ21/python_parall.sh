#!/bin/bash
#SBATCH --output=5seeds-%j.log
#SBATCH --error=5seeds-%j.err
#SBATCH -p spr
#SBATCH --time=2-00:00:00
#SBATCH --nodes=20
#SBATCH --ntasks=40
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=56
#SBATCH --job-name="CNN_5seeds"

module load python3
module load impi
source activate CDS


ibrun -np 40 python -u Threemethods_mpiv06.py
