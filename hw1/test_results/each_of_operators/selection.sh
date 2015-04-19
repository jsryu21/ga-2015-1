#!/bin/bash
funcion_to_fork () {
    for j in {0..29}
    do
        ./ga ${1} 6 6 5 < cycle.in.101 >> "selection_${1}"
    done
}

for i in {0..3}
do
    (sleep $((30 * $i)) && funcion_to_fork $i) &
done
