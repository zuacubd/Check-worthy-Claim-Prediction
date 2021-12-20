#!/bin/bash


echo "launching ..."
python3 app/main_checkthat.py -data_dir $1 -N $2 -L $3 -C $4 -W $5 -fs $6 -le $7 -pr $8 -ev $9
echo "done."
