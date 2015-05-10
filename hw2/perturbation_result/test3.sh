#!/bin/bash
for i in {0..5}
do
    for j in {0..3}
    do
        echo $i $j >> result_3
        ./ga 4 6 6 3 5 $i $j < cycle.in.318 >> result_3
    done
done
