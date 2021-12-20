#!/bin/bash

#SBATCH --job-name=Checkthat
#SBATCH --output=logs/checkthat.log
#SBATCH --error=logs/checkthat.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=24CPUNodes
#SBATCH --mem-per-cpu=8000M


source scripts/aven.sh

data_dir="/projets/sig/mullah/nlp/checkthat/"

print("launching ...")
python3 app/main_checkthat.py $data_dir
print("done.")

