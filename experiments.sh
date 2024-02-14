#!/bin/bash

# ResNet-50
python attack.py -model resnet50 -weights IMAGENET1K_V2 -results sqlite:///fourier.db -name resnet50 -eps 4 -norm Linf -data "$VSC_DATA_VO/ImageNet" -bs 32 -adapt -trial 273 -iterations 1 -attack autoattack -timeout 120
