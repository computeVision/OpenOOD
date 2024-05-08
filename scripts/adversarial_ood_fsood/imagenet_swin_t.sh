#!/bin/bash
# sh scripts/adversarial_ood/swint_t.sh

# posthocs=(
#     msp 
#     mds 
#     # mds_ensemble 
#     # gram 
#     ebo 
#     rmds 
#     gradnorm 
#     react 
#     mls 
#     klm 
#     # gmm
#     vim 
#     knn 
#     dice 
#     # rankfeat 
#     ash 
#     she 
#     gen 
#     nnguide 
#     relation 
#     scale
#     )

posthocs=( odin ) 

# mds_ensemble --> TypeError: forward() got an unexpected keyword argument 'return_feature_list'
# gram --> TypeError: forward() got an unexpected keyword argument 'return_feature_list'
# rankfeat --> AttributeError: 'Swin_T' object has no attribute 'intermediate_forward'

for iter in "${posthocs[@]}"; do
    echo "$iter"
    python scripts/eval_adversarial_ood_imagenet.py \
        --tvs-pretrained \
        --arch swin-t \
        --postprocessor "$iter" \
        --save-score --save-csv --fsood
done



# # SEM
# python main.py \
# --config configs/datasets/imagenet/imagenet.yml \
# configs/datasets/imagenet/imagenet_adversarial_ood.yml \
# configs/networks/resnet50.yml \
# configs/pipelines/test/test_ood.yml \
# configs/preprocessors/base_preprocessor.yml \
# configs/postprocessors/gmm.yml \
# --num_workers 4 \
# --ood_dataset.image_size 224 \
# --dataset.test.batch_size 256 \
# --dataset.val.batch_size 256 \
# --network.pretrained True \
# --network.checkpoint 'results/checkpoints/imagenet_res50_acc76.10.pth' \
# --merge_option merge