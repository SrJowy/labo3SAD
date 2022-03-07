#!/bin/bash

k_min=$1
k_max=$2
p_min=$3
p_max=$4

while [ $k_min -le $k_max ]
do
    p_min=$3
    while [ $p_min -le $p_max ]
    do
        python knn.py -k $k_min -d $p_min -a -c Especie -o f.txt
        let p_min=p_min+1
    done
    let k_min=k_min+2
done

exit 0