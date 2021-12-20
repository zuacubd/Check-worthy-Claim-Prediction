#!/bin/bash

data_dir="/projets/sig/mullah/nlp/checkthat18"
fs=0
le=1
pr=1
ev=0

for N in 0 1; do
   for L in 0 1; do
      for C in 0 1; do
         for W in 0 1; do
     	    if [ $((N+L+C+W)) != 0 ]; then
	        features=$N.$L.$C.$W
		sbatch -c 4 -p 24CPUNodes --mem-per-cpu 8000M -o logs/Checkthat_train2018_$features.out -e logs/Checkthat_train2018_$features.err -J ch2018-$features scripts/main_train_features.sh $data_dir $N $L $C $W $fs $le $pr $ev
	    fi
	 done
      done
   done
done

