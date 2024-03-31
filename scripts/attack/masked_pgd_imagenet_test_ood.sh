#!/bin/bash
# sh scripts/ood/scale/imagenet_test_ood_scale.sh

############################################
# we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood_imagenet.py

# available architectures:
# resnet50, swin-t, vit-b-16
# ood
# python scripts/attack_ood_imagenet.py \
#     --tvs-pretrained \
#     --arch resnet50 \
#     --postprocessor scale \
#     --save-score --save-csv #--fsood

# full-spectrum ood
python scripts/attack_ood_imagenet.py \
    --tvs-pretrained \
    --arch resnet50 \
    --att masked_pgd
    # --postprocessor scale \
    # --save-score --save-csv --fsood

