#!/bin/bash

data=${1:-"$VSC_DATA_VO/ImageNet"}
eps=${2:-16}

# All attacks are Linf @ 16/255
# Four settings:
# base: attack baseline models using Square attack
# robust: attack RobustBench models using Square attack
# cs: attack baseline models defended using our CS method with Square attack (black-box setting, no surrogate)
# xfer: attack models with APGD adversarials transferred from unprotected baseline (black-box setting, surrogate)

###################
# Baseline models #
###################

# Attack the baseline models directly using Square.

# ResNet-50
python attack.py -log logs/base/resnet50.json -model resnet50 -weights IMAGENET1K_V2 -results sqlite:///fourier.db -name resnet50 -eps "$eps" -norm Linf -data "$data" -bs 16 -base -trial 273 -iterations 1 -attack square

# Wide-Resnet101
python attack.py -log logs/base/wide_resnet101.json -model wide_resnet101_2 -weights IMAGENET1K_V2 -results sqlite:///fourier.db -name wide_resnet101 -eps "$eps" -norm Linf -data "$data" -bs 16 -base -trial 106 -iterations 1 -attack square

# ViT-B-16
python attack.py -log logs/base/vit_b_16.json -model vit_b_16 -results sqlite:///fourier.db -name vit_b_16 -eps "$eps" -norm Linf -data "$data" -bs 16 -base -trial 52 -iterations 1 -attack square

# Swin-T
python attack.py -log logs/base/swin_t.json -model swin_t -results sqlite:///fourier.db -name swin_t -eps "$eps" -norm Linf -data "$data" -bs 16 -base -trial 275 -iterations 1 -attack square

#################
# RB models     #
#################

# Attack the RobustBench models using Square attack.
# Square attack simulates black-box setting with no direct access to the model (under logs/robust).
# APGD simulates black-box setting with whitebox access to a surrogate model (under logs/xfer).

# Wong2020Fast
python attack.py -log logs/robust/wong2020fast.json -model Wong2020Fast -rb -base -results sqlite:///fourier.db -name resnet50 -eps "$eps" -norm Linf -data "$data" -bs 16 -trial 273 -iterations 1 -attack square
python attack.py -log logs/xfer/wong2020fast.json -model Wong2020Fast -rb -base -results sqlite:///fourier.db -name resnet50 -eps "$eps" -norm Linf -data "$data" -bs 16 -trial 273 -iterations 1 -attack apgd

# Peng2023Robust
python attack.py -log logs/robust/peng2023robust.json -model Peng2023Robust -rb -base -results sqlite:///fourier.db -name wide_resnet101 -eps "$eps" -norm Linf -data "$data" -bs 16 -trial 106 -iterations 1 -attack square
python attack.py -log logs/xfer/peng2023robust.json -model Peng2023Robust -rb -base -results sqlite:///fourier.db -name wide_resnet101 -eps "$eps" -norm Linf -data "$data" -bs 16 -trial 106 -iterations 1 -attack apgd

# Debenedetti2022Light_XCiT-L12
python attack.py -log logs/robust/debenedetti2022light_xcit-l12.json -model Debenedetti2022Light_XCiT-L12 -rb -base -results sqlite:///fourier.db -name vit_b_16 -eps "$eps" -norm Linf -data "$data" -bs 16 -trial 52 -iterations 1 -attack square
python attack.py -log logs/xfer/debenedetti2022light_xcit-l12.json -model Debenedetti2022Light_XCiT-L12 -rb -base -results sqlite:///fourier.db -name vit_b_16 -eps "$eps" -norm Linf -data "$data" -bs 16 -trial 52 -iterations 1 -attack apgd

# Liu2023Comprehensive_Swin-L
python attack.py -log logs/robust/liu2023comprehensive_swin-l.json -model Liu2023Comprehensive_Swin-L -rb -base -results sqlite:///fourier.db -name swin_t -eps "$eps" -norm Linf -data "$data" -bs 16 -trial 275 -iterations 1 -attack square
python attack.py -log logs/xfer/liu2023comprehensive_swin-l.json -model Liu2023Comprehensive_Swin-L -rb -base -results sqlite:///fourier.db -name swin_t -eps "$eps" -norm Linf -data "$data" -bs 16 -trial 275 -iterations 1 -attack apgd

###################
# CS models       #
###################

# Attack the baseline models defended using our CS method.
# Square attack simulates black-box setting with no direct access to the model (under logs/cs).
# APGD simulates black-box setting with whitebox access to a surrogate model (under logs/xfer).

# ResNet-50
python attack.py -log logs/cs/resnet50.json -model resnet50 -weights IMAGENET1K_V2 -results sqlite:///fourier.db -name resnet50 -eps "$eps" -norm Linf -data "$data" -bs 16 -adapt -trial 273 -iterations 1 -attack square
python attack.py -log logs/xfer/resnet50.json -model resnet50 -weights IMAGENET1K_V2 -results sqlite:///fourier.db -name resnet50 -eps "$eps" -norm Linf -data "$data" -bs 16 -trial 273 -iterations 1 -attack apgd

# Wide-Resnet101
python attack.py -log logs/cs/wide_resnet101.json -model wide_resnet101_2 -weights IMAGENET1K_V2 -results sqlite:///fourier.db -name wide_resnet101 -eps "$eps" -norm Linf -data "$data" -bs 16 -adapt -trial 106 -iterations 1 -attack square
python attack.py -log logs/xfer/wide_resnet101.json -model wide_resnet101_2 -weights IMAGENET1K_V2 -results sqlite:///fourier.db -name wide_resnet101 -eps "$eps" -norm Linf -data "$data" -bs 16 -trial 106 -iterations 1 -attack apgd

# ViT-B-16
python attack.py -log logs/cs/vit_b_16.json -model vit_b_16 -results sqlite:///fourier.db -name vit_b_16 -eps "$eps" -norm Linf -data "$data" -bs 16 -adapt -trial 52 -iterations 1 -attack square
python attack.py -log logs/xfer/vit_b_16.json -model vit_b_16 -results sqlite:///fourier.db -name vit_b_16 -eps "$eps" -norm Linf -data "$data" -bs 16 -trial 52 -iterations 1 -attack apgd

# Swin-T
python attack.py -log logs/cs/swin_t.json -model swin_t -results sqlite:///fourier.db -name swin_t -eps "$eps" -norm Linf -data "$data" -bs 16 -adapt -trial 275 -iterations 1 -attack square
python attack.py -log logs/xfer/swin_t.json -model swin_t -results sqlite:///fourier.db -name swin_t -eps "$eps" -norm Linf -data "$data" -bs 16 -trial 275 -iterations 1 -attack apgd
