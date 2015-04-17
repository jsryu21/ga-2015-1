#!/bin/bash
k=0
for i in $(cat 101)
do
    if [ "$k" -lt "4" ]; then
        echo -n "$i "
    fi
    if [ "$k" -eq "106" ]; then
        echo "$i "
        k=-1
    fi
    k=$((k+1))
done
