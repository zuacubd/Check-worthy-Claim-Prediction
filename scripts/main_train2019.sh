#!/bin/bash

#SBATCH --job-name=Checkthat_train19
#SBATCH --output=logs/checkthat_train19.log
#SBATCH --error=logs/checkthat_train19.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=24CPUNodes
#SBATCH --mem-per-cpu=8000M


#source scripts/aven.sh

data_dir="/projets/sig/mullah/nlp/checkthat19"

echo "launching ..."
python3 app/main_checkthat.py -data_dir $data_dir -p 1
echo "done."

