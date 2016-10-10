#!/bin/bash

declare -a thresholds=(0.01 0.015 0.02 0.025 0.03 0.04 0.05)

## now loop through the above array
for threshold in "${thresholds[@]}"
do
   cmd="python lista.py --resultFile listaResults.csv --threshold $threshold --iteration 10 --epoch 50"
   echo "Running: " $cmd
   eval $cmd
done
