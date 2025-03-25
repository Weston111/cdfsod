#!/bin/bash

gpu_num=4

datasets=(
dataset1
dataset2
dataset3
)
shots=(
1
5
10
)
for dataset in ${datasets[@]}; do
    for shot in ${shots[@]}; do
        bash tools/dist_train.sh configs/mm_grounding_dino/cdfsod/${dataset}/${dataset}_${shot}shot.py ${gpu_num} --work-dir output/${dataset}/${shot}shot --json-file ./test_results/${dataset}_${shot}shot
    done
done
