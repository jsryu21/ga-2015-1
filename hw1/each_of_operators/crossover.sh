#!/bin/bash
funcion_to_fork () {
    for j in {0..29}
    do
        ./ga 4 ${1} 6 5 < cycle.in.101 >> "crossover_${1}"
    done
}

for i in {0..5}
do
    (sleep $((30 * $i)) && funcion_to_fork $i) &
done
