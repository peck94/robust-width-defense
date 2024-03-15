#!/bin/bash

###################
# Baseline models #
###################

# ResNet-50
#python attack.py -log logs/resnet50_base.json -model resnet50 -weights IMAGENET1K_V2 -results sqlite:///fourier.db -name resnet50 -eps 16 -norm Linf -data "$VSC_DATA_VO/ImageNet" -bs 4 -base -trial 273 -iterations 1 -attack autoattack

# Wide-Resnet101
#python attack.py -log logs/wide_resnet101_base.json -model wide_resnet101_2 -results sqlite:///fourier.db -name wide_resnet101 -eps 16 -norm Linf -data "$VSC_DATA_VO/ImageNet" -bs 4 -base -trial 106 -iterations 1 -attack autoattack

# Wong2020Fast
#python attack.py -log logs/wong2020fast_base.json -model Wong2020Fast -rb -base -weights IMAGENET1K_V2 -results sqlite:///fourier.db -name resnet50 -eps 16 -norm Linf -data "$VSC_DATA_VO/ImageNet" -bs 4 -base -trial 273 -iterations 1 -attack autoattack

# ViT-B-16
python attack.py -weights IMAGENET1K_V1 -log logs/vit_b_16_base.json -model vit_b_16 -results sqlite:///fourier.db -name vit_b_16 -eps 16 -norm Linf -data "$VSC_DATA_VO/ImageNet" -bs 4 -base -trial 52 -iterations 1 -attack autoattack

# Swin-T
python attack.py -weights IMAGENET1K_V1 -log logs/swin_t_base.json -model swin_t -results sqlite:///fourier.db -name swin_t -eps 16 -norm Linf -data "$VSC_DATA_VO/ImageNet" -bs 4 -base -trial 275 -iterations 1 -attack autoattack
