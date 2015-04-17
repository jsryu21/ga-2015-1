#!/bin/bash
for i in {0..3}
do
    for j in {0..5}
    do
        for k in {0..5}
        do
            for p in {0..4}
            do
                echo "$i $j $k $p"
                for q in {0..1}
                do
                    ./ga $i $j $k $p < cycle.in.21
                done
            done
        done
    done
done
