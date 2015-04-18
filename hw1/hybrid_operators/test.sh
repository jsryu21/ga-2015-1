#!/bin/bash
funcion_to_fork () {
    ./ga < cycle.in.318 >> hy_318
    ./ga < cycle.in.101 >> hy_101
    ./ga < cycle.in.11 >> hy_11
    ./ga < cycle.in.21 >> hy_21
    ./ga < cycle.in.51 >> hy_51
}

for i in {0..29}
do
    (sleep $((30 * $i)) && funcion_to_fork) &
done
