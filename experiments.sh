#!/bin/bash

###################
# Baseline models #
###################

# ResNet-50
python attack.py -log logs/resnet50.json -model resnet50 -weights IMAGENET1K_V2 -results sqlite:///fourier.db -name resnet50 -eps 16 -norm Linf -data "$VSC_DATA_VO/ImageNet" -bs 4 -adapt -trial 273 -iterations 1 -attack autoattack

# Wide-Resnet101
python attack.py -log logs/wide_resnet101.json -model wide_resnet101_2 -results sqlite:///fourier.db -name wide_resnet101 -eps 16 -norm Linf -data "$VSC_DATA_VO/ImageNet" -bs 4 -adapt -trial 106 -iterations 1 -attack autoattack

# ViT-B-16
python attack.py -log logs/vit_b_16.json -model vit_b_16 -weights IMAGENET1K_V2 -results sqlite:///fourier.db -name vit_b_16 -eps 16 -norm Linf -data "$VSC_DATA_VO/ImageNet" -bs 4 -adapt -trial 52 -iterations 1 -attack autoattack

# Swin-T
python attack.py -log logs/swin_t.json -model swin_t -weights IMAGENET1K_V2 -results sqlite:///fourier.db -name swin_t -eps 16 -norm Linf -data "$VSC_DATA_VO/ImageNet" -bs 4 -adapt -trial 275 -iterations 1 -attack autoattack

#################
# Robust models #
#################

# Wong2020Fast
python attack.py -log logs/wong2020fast.json -model Wong2020Fast -rb -base -weights IMAGENET1K_V2 -results sqlite:///fourier.db -name resnet50 -eps 16 -norm Linf -data "$VSC_DATA_VO/ImageNet" -bs 4 -adapt -trial 273 -iterations 1 -attack autoattack

# Peng2023Robust
python attack.py -log logs/peng2023robust.json -model Peng2023Robust -rb -base -results sqlite:///fourier.db -name wide_resnet101 -eps 16 -norm Linf -data "$VSC_DATA_VO/ImageNet" -bs 4 -adapt -trial 106 -iterations 1 -attack autoattack

# Debenedetti2022Light_XCiT-L12
python attack.py -log logs/debenedetti2022light_xcit-l12.json -model Debenedetti2022Light_XCiT-L12 -rb -base -results sqlite:///fourier.db -name vit_b_16 -eps 16 -norm Linf -data "$VSC_DATA_VO/ImageNet" -bs 4 -adapt -trial 52 -iterations 1 -attack autoattack

# Liu2023Comprehensive_Swin-L
python attack.py -log logs/liu2023comprehensive_swin-l.json -model Liu2023Comprehensive_Swin-L -rb -base -results sqlite:///fourier.db -name swin_t -eps 16 -norm Linf -data "$VSC_DATA_VO/ImageNet" -bs 4 -adapt -trial 275 -iterations 1 -attack autoattack

#################
# Combo models  #
#################

# Wong2020Fast
python attack.py -log logs/wong2020fast_cs.json -model Wong2020Fast -rb -weights IMAGENET1K_V2 -results sqlite:///fourier.db -name resnet50 -eps 16 -norm Linf -data "$VSC_DATA_VO/ImageNet" -bs 4 -adapt -trial 273 -iterations 1 -attack autoattack

# Peng2023Robust
python attack.py -log logs/peng2023robust_cs.json -model Peng2023Robust -rb -results sqlite:///fourier.db -name wide_resnet101 -eps 16 -norm Linf -data "$VSC_DATA_VO/ImageNet" -bs 4 -adapt -trial 106 -iterations 1 -attack autoattack

# Debenedetti2022Light_XCiT-L12
python attack.py -log logs/debenedetti2022light_xcit-l12_cs.json -model Debenedetti2022Light_XCiT-L12 -rb -results sqlite:///fourier.db -name vit_b_16 -eps 16 -norm Linf -data "$VSC_DATA_VO/ImageNet" -bs 4 -adapt -trial 52 -iterations 1 -attack autoattack

# Liu2023Comprehensive_Swin-L
python attack.py -log logs/liu2023comprehensive_swin-l_cs.json -model Liu2023Comprehensive_Swin-L -rb -results sqlite:///fourier.db -name swin_t -eps 16 -norm Linf -data "$VSC_DATA_VO/ImageNet" -bs 4 -adapt -trial 275 -iterations 1 -attack autoattack
