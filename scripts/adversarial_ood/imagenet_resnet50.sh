#!/bin/bash
# sh scripts/adversarial_ood/imagenet_resnet50.sh

posthocs=(msp mds mds_ensemble gram ebo rmds gradnorm react mls klm sem vim knn dice rankfeat ash she gen nnguide relation scale)

for iter in "${posthocs[@]}"; do
    echo "$iter"
    python scripts/eval_ood_imagenet.py \
        --tvs-pretrained \
        --arch resnet50 \
        --postprocessor "$iter" \
        --save-score --save-csv
done

