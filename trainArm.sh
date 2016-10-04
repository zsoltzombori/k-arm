#!/bin/bash

dict=400
threshold=0.01
lr=0.001
epoch=20

declare -a trainIterations=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
declare -a testIterations=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 15 17 18 19 20 21 22 23 24 25 26 27 28 29 30)

## now loop through the above array
for trainIteration in "${trainIterations[@]}"
do
   cmd="python lista.py --iteration $trainIteration --dict $dict --threshold $threshold --lr $lr --epoch $epoch"
   echo "Training: " $cmd
   eval $cmd
   for testIteration in "${testIterations[@]}"
   do
       cmd="python lista.py --iteration $testIteration --threshold $threshold --dict $dict --weightFile dict/it${trainIteration}_th${threshold}.npz --resultFile results.csv --epoch 0"
       echo "Testing: " $cmd
       eval $cmd
   done
done
