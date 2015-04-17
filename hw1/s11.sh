#!/bin/bash
k=0
for i in $(cat 11)
do
    if [ "$k" -lt "4" ]; then
        echo -n "$i "
    fi
    if [ "$k" -eq "16" ]; then
        echo -n "$i "
    fi
    if [ "$k" -eq "29" ]; then
        echo -n "$i "
    fi
    if [ "$k" -eq "42" ]; then
        echo -n "$i "
    fi
    if [ "$k" -eq "55" ]; then
        echo -n "$i "
    fi
    if [ "$k" -eq "68" ]; then
        echo "$i "
        k=-1
    fi
    k=$((k+1))
done
