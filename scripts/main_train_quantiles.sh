#!/bin/bash


echo "launching ..."
python3 app/main_checkthat.py -data_dir $1 -N $2 -L $3 -C $4 -W $5 -fs $6 -qg $7 -le $8 -pr $9 -ev ${10} -Q1 ${11} -Q2 ${12} -Q3 ${13} -Q4 ${14}
echo "done."
