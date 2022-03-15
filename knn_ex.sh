#!/bin/bash

k_min=$1
k_max=$2
p_max=$3

while [ $k_min -le $k_max ]
do
    let p_min=1
    while [ $p_min -le $p_max ]
    do
        python knn_def.py -k $k_min -d $p_min -o f.csv -m uniform
        python knn_def.py -k $k_min -d $p_min -o f.csv -m distance
        let p_min=p_min+1
    done
    let k_min=k_min+2
done

exit 0