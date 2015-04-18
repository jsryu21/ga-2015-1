#!/bin/bash
for i in {0..3}
do
    for j in {0..5}
    do
        for k in {0..5}
        do
            for p in {0..4}
            do
                f="11_${i}_${j}_${k}_${p}"
                cat ${f} >> 11
            done
        done
    done
done
