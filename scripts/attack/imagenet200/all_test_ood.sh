#!/bin/bash


python scripts/attack_ood.py \
   --id-data imagenet200 \
   --root ./results/imagenet200_resnet18_224x224_base_e90_lr0.1_default \
   --att pgd \
   --eps "4/255" \
   --batch-size 64

python scripts/attack_ood.py \
   --id-data imagenet200 \
   --root ./results/imagenet200_resnet18_224x224_base_e90_lr0.1_default \
   --att fgsm \
   --eps "4/255" \
   --batch-size 64

python scripts/attack_ood.py \
   --id-data imagenet200 \
   --root ./results/imagenet200_resnet18_224x224_base_e90_lr0.1_default \
   --att df \
   --eps "4/255" \
   --batch-size 64
   

python scripts/attack_ood.py \
   --id-data imagenet200 \
   --root ./results/imagenet200_resnet18_224x224_base_e90_lr0.1_default \
   --att mpgd \
   --eps "1" \
   --batch-size 64