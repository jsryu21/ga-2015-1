#!/bin/bash
funcion_to_fork () {
    for k in {0..5}
    do
        for p in {0..4}
        do
            echo "$1 $2 $k $p" >> 11_${1}_${2}_${k}_${p}
            for q in {0..4}
            do
                ./ga $1 $2 $k $p < cycle.in.11 >> 11_${1}_${2}_${k}_${p}
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
