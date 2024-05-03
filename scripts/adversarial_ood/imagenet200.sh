#!/bin/bash
# sh scripts/adversarial_ood/imagenet200.sh

# posthocs=(
#     # msp 
#     # mds 
#     # mds_ensemble 
#     # gram 
#     ebo 
#     rmds 
#     gradnorm 
#     react 
#     mls 
#     klm 
#     sem 
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
        --id-data imagenet200 \
        --root ./results/imagenet200_resnet18_224x224_base_e90_lr0.1_default \
        --postprocessor "$iter" \
        --save-score --save-csv
done