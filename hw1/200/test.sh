#!/bin/bash
funcion_to_fork () {
    ./ga 2 0 3 2 < cycle.in.318 >> 318.csv
    ./ga 2 0 3 2 < cycle.in.101 >> 101.csv
    ./ga 2 0 3 2 < cycle.in.11 >> 11.csv
    ./ga 2 0 3 2 < cycle.in.21 >> 21.csv
    ./ga 2 0 3 2 < cycle.in.51 >> 51.csv
}

for i in {0..29}
do
    (sleep $((30 * $i)) && funcion_to_fork) &
done
