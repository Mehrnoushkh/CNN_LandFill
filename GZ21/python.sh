#!/bin/bash
#SBATCH --output=sim-%j.log
#SBATCH --error=sim-%j.err
#SBATCH -p spr          # SKX has more memory than SPR
#SBATCH --time=01-10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --job-name="CNN_100"

module load python3              # Load Python module if needed
conda activate CDS
#python box2.py
#python box_tesy.py 
#python maps.py
#python compare_learn.py


python threemethods_modelv02.py
#python  threemethods_model.py

#python train_filled.py
#python  my_modelfor_filled.py
#python  plot_fille.py #plot.py
