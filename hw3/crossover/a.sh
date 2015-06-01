for j in {0..11}
do
    for i in {1..20}
    do
        echo $j $i
        thorq --add ../ga 100 $j 100 100 100 100 100
    done
done
