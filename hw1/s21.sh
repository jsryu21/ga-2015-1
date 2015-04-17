#!/bin/bash
k=0
for i in $(cat 21)
do
    if [ "$k" -lt "4" ]; then
        echo -n "$i "
    fi
    if [ "$k" -eq "26" ]; then
        echo -n "$i "
    fi
    if [ "$k" -eq "49" ]; then
        echo -n "$i "
    fi
    if [ "$k" -eq "72" ]; then
        echo "$i "
        k=-1
    fi
    k=$((k+1))
done
