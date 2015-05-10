#!/bin/bash
for i in {0..7}
do
    ./test${i}.sh &
done
