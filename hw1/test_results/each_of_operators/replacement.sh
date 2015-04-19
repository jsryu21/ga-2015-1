#!/bin/bash
funcion_to_fork () {
    for j in {0..29}
    do
        ./ga 4 6 6 ${1} < cycle.in.101 >> "replacement_${1}"
    done
}

for i in {0..4}
do
    (sleep $((30 * $i)) && funcion_to_fork $i) &
done
