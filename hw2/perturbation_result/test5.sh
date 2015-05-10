#!/bin/bash
for i in {0..5}
do
    for j in {0..3}
    do
        echo $i $j >> result_5
        ./ga 4 6 6 5 5 $i $j < cycle.in.318 >> result_5
    done
done
