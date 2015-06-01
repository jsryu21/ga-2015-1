for j in {0..3}
do
    for i in {1..20}
    do
        echo $j $i
        thorq --add ../ga 100 100 100 100 100 100 $j
    done
done
