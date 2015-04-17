#!/bin/bash
i=0
funcion_to_fork () {
    for k in {0..5}
    do
        for p in {0..4}
        do
            echo "$1 $2 $k $p" >> 11_$1_$2_$k_$p
            for q in {0..4}
            do
                ./ga $1 $2 $k $p < cycle.in.11 >> 11_$1_$2_$k_$p
            done
        done
    done
}

for i in {0..3}
do
    for j in {0..5}
    do
        (funcion_to_fork $i $j)&
    done
done
