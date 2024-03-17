#!/bin/bash

# All attacks are Linf @ 16/255

###################
# Baseline models #
###################

# Attack the baseline models directly using Square attack. Should yield 0% robust accuracy.

# ResNet-50
python attack.py -log logs/base/resnet50.json -model resnet50 -weights IMAGENET1K_V2 -results sqlite:///fourier.db -name resnet50 -eps 16 -norm Linf -data "$VSC_DATA_VO/ImageNet" -bs 16 -base -trial 273 -iterations 1 -attack autoattack

# Wide-Resnet101
python attack.py -log logs/base/wide_resnet101.json -model wide_resnet101_2 -weights IMAGENET1K_V2 -results sqlite:///fourier.db -name wide_resnet101 -eps 16 -norm Linf -data "$VSC_DATA_VO/ImageNet" -bs 16 -base -trial 106 -iterations 1 -attack autoattack

# ViT-B-16
python attack.py -log logs/base/vit_b_16.json -model vit_b_16 -results sqlite:///fourier.db -name vit_b_16 -eps 16 -norm Linf -data "$VSC_DATA_VO/ImageNet" -bs 16 -base -trial 52 -iterations 1 -attack autoattack

# Swin-T
python attack.py -log logs/base/swin_t.json -model swin_t -results sqlite:///fourier.db -name swin_t -eps 16 -norm Linf -data "$VSC_DATA_VO/ImageNet" -bs 16 -base -trial 275 -iterations 1 -attack autoattack

#################
# RB models     #
#################

# Attack the RobustBench models using Square attack.

# Wong2020Fast
python attack.py -log logs/robust/wong2020fast.json -model Wong2020Fast -rb -base -results sqlite:///fourier.db -name resnet50 -eps 16 -norm Linf -data "$VSC_DATA_VO/ImageNet" -bs 16 -trial 273 -iterations 1 -attack autoattack

# Peng2023Robust
python attack.py -log logs/robust/peng2023robust.json -model Peng2023Robust -rb -base -results sqlite:///fourier.db -name wide_resnet101 -eps 16 -norm Linf -data "$VSC_DATA_VO/ImageNet" -bs 16 -trial 106 -iterations 1 -attack autoattack

# Debenedetti2022Light_XCiT-L12
python attack.py -log logs/robust/debenedetti2022light_xcit-l12.json -model Debenedetti2022Light_XCiT-L12 -rb -base -results sqlite:///fourier.db -name vit_b_16 -eps 16 -norm Linf -data "$VSC_DATA_VO/ImageNet" -bs 16 -trial 52 -iterations 1 -attack autoattack

# Liu2023Comprehensive_Swin-L
python attack.py -log logs/robust/liu2023comprehensive_swin-l.json -model Liu2023Comprehensive_Swin-L -rb -base -results sqlite:///fourier.db -name swin_t -eps 16 -norm Linf -data "$VSC_DATA_VO/ImageNet" -bs 16 -trial 275 -iterations 1 -attack autoattack

###################
# CS models       #
###################

# Attack the baseline models defended using our CS method.
# Square attack simulates black-box setting with no direct access to the model (under logs/cs).
# AutoAttack simulates black-box setting with whitebox access to a surrogate model (under logs/xfer).

# ResNet-50
python attack.py -log logs/cs/resnet50.json -model resnet50 -weights IMAGENET1K_V2 -results sqlite:///fourier.db -name resnet50 -eps 16 -norm Linf -data "$VSC_DATA_VO/ImageNet" -bs 16 -adapt -trial 273 -iterations 1 -attack autoattack
python attack.py -log logs/xfer/resnet50.json -model resnet50 -weights IMAGENET1K_V2 -results sqlite:///fourier.db -name resnet50 -eps 16 -norm Linf -data "$VSC_DATA_VO/ImageNet" -bs 16 -trial 273 -iterations 1 -attack autoattack

# Wide-Resnet101
python attack.py -log logs/cs/wide_resnet101.json -model wide_resnet101_2 -weights IMAGENET1K_V2 -results sqlite:///fourier.db -name wide_resnet101 -eps 16 -norm Linf -data "$VSC_DATA_VO/ImageNet" -bs 16 -adapt -trial 106 -iterations 1 -attack autoattack
python attack.py -log logs/xfer/wide_resnet101.json -model wide_resnet101_2 -weights IMAGENET1K_V2 -results sqlite:///fourier.db -name wide_resnet101 -eps 16 -norm Linf -data "$VSC_DATA_VO/ImageNet" -bs 16 -trial 106 -iterations 1 -attack autoattack

# ViT-B-16
python attack.py -log logs/cs/vit_b_16.json -model vit_b_16 -results sqlite:///fourier.db -name vit_b_16 -eps 16 -norm Linf -data "$VSC_DATA_VO/ImageNet" -bs 16 -adapt -trial 52 -iterations 1 -attack autoattack
python attack.py -log logs/xfer/vit_b_16.json -model vit_b_16 -results sqlite:///fourier.db -name vit_b_16 -eps 16 -norm Linf -data "$VSC_DATA_VO/ImageNet" -bs 16 -trial 52 -iterations 1 -attack autoattack

# Swin-T
python attack.py -log logs/cs/swin_t.json -model swin_t -results sqlite:///fourier.db -name swin_t -eps 16 -norm Linf -data "$VSC_DATA_VO/ImageNet" -bs 16 -adapt -trial 275 -iterations 1 -attack autoattack
python attack.py -log logs/xfer/swin_t.json -model swin_t -results sqlite:///fourier.db -name swin_t -eps 16 -norm Linf -data "$VSC_DATA_VO/ImageNet" -bs 16 -adapt -trial 275 -iterations 1 -attack autoattack
