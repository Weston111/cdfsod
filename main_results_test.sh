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
        bash tools/dist_test.sh configs/mm_grounding_dino/cdfsod/${dataset}/${dataset}_${shot}shot.py checkpoints/${dataset}_${shot}shot.pth ${gpu_num} --json-file ./test_results/${dataset}_${shot}shot
    done
done
