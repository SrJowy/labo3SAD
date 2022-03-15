#!/bin/bash

min_d=$1
max_d=$2
min_ss=$3
max_ss=$4
min_sl=$5
max_sl=$6

while [ $min_d -le $max_d ]
do
    min_ss=$3
    while [ $min_ss -le $max_ss ]
    do
        min_sl=$5
        while [ $min_sl -le $max_sl ]
        do
            python trees_def.py -m $min_d -s $min_ss -l $min_sl -o fT.csv
            let min_sl=min_sl+1
        done
        let min_ss=min_ss+1
    done
    let min_d=min_d+3
done

exit 0