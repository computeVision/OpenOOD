#!/bin/bash
# sh scripts/ood/gram/cifar10_test_ood_gram.sh

############################################
# alternatively, we recommend using the
# new unified, easy-to-use evaluator with
# the example script scripts/eval_ood.py
# especially if you want to get results from
# multiple runs
python scripts/attack_ood.py \
   --id-data cifar10 \
   --root ./results/cifar10_resnet18_32x32_base_e100_lr0.1_default \
   --att masked_pgd \
   --masked-patch-size 8
   