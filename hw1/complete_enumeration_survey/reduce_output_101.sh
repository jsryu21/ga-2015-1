#!/bin/bash
for i in {0..3}
do
    for j in {0..5}
    do
        for k in {0..5}
        do
            for p in {0..4}
            do
                f="101_${i}_${j}_${k}_${p}"
                cat ${f} >> 101
            done
        done
    done
done
