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
        python knn_def.py -k $k_min -d $p_min -o f.csv -m uniform
        python knn_def.py -k $k_min -d $p_min -o f.csv -m distance
        let p_min=p_min+1
    done
    let k_min=k_min+2
done

exit 0