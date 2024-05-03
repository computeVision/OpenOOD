#!/bin/bash
# sh scripts/adversarial_ood/cifar10.sh

# GPU=1
# CPU=1
# node=73
# jobname=openood

# posthocs=(
#     msp 
#     mds 
#     mds_ensemble 
#     gram 
#     ebo 
#     rmds 
#     gradnorm 
#     react 
#     mls 
#     klm 
#     gmm 
#     vim 
#     knn 
#     dice 
#     rankfeat 
#     ash 
#     she 
#     gen 
#     nnguide 
#     relation 
#     scale
#     )

posthocs=( odin ) 

for iter in "${posthocs[@]}"; do
    echo "$iter"
    python scripts/eval_adversarial_ood.py \
        --id-data cifar10 \
        --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
        --postprocessor "$iter" \
        --save-score --save-csv
done
