#!/bin/bash

# ResNet-50
#python attack.py -log logs/resnet50_simba.json -model resnet50 -weights IMAGENET1K_V2 -results sqlite:///fourier.db -name resnet50 -eps 4 -norm Linf -data "$VSC_DATA_VO/ImageNet" -bs 1 -adapt -trial 273 -iterations 1 -attack simba -timeout 120 -softmax
#python attack.py -log logs/resnet50_autoattack.json -model resnet50 -weights IMAGENET1K_V2 -results sqlite:///fourier.db -name resnet50 -eps 4 -norm Linf -data "$VSC_DATA_VO/ImageNet" -bs 4 -adapt -trial 273 -iterations 1 -attack autoattack

# Wide-Resnet101
python attack.py -log logs/wide_resnet101_autoattack.json -model wide_resnet101_2 -results sqlite:///fourier.db -name wide_resnet101 -eps 4 -norm Linf -data "$VSC_DATA_VO/ImageNet" -bs 4 -adapt -trial 106 -iterations 1 -attack autoattack
python attack.py -log logs/wide_resnet101_autoattack.json -model wide_resnet101_2 -results sqlite:///fourier.db -name wide_resnet101 -eps 4 -norm Linf -data "$VSC_DATA_VO/ImageNet" -bs 4 -adapt -trial 106 -iterations 1 -attack simba -softmax
