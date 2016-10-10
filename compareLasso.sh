#!/bin/bash

declare -a thresholds=(0.001 0.002 0.005 0.008 0.01 0.015 0.02 0.025 0.03 0.04 0.05 0.1 0.15 0.2)

## now loop through the above array
for threshold in "${thresholds[@]}"
do
   cmd="python baseline.py --resultFile lassoResults.csv --threshold $threshold --iteration 1000"
   echo "Running: " $cmd
   eval $cmd
done
