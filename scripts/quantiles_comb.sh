#!/bin/bash

data_dir="/projets/sig/mullah/nlp/checkthat18"
fs=0
qg=1
le=1
pr=1
ev=0
N=1
L=1
C=1
W=1
for Q1 in 0 1; do
   for Q2 in 0 1; do
      for Q3 in 0 1; do
         for Q4 in 0 1; do
     	    if [ $((Q1+Q2+Q3+Q4)) != 0 ]; then
	        quantiles=$Q1.$Q2.$Q3.$Q4
		sbatch -c 4 -p 24CPUNodes --mem-per-cpu 8000M -o logs/checkthat_train2018_quantiles_$quantiles.out -e logs/checkthat_train2018_quantiles_$quantiles.err -J ch2018-quantiles-$quantiles scripts/main_train_quantiles.sh $data_dir $N $L $C $W $fs $qg $le $pr $ev $Q1 $Q2 $Q3 $Q4
	    fi
	 done
      done
   done
done

