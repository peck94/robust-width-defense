#!/bin/bash

# ResNet-50
python attack.py -log logs/resnet50_simba.json -model resnet50 -weights IMAGENET1K_V2 -results sqlite:///fourier.db -name resnet50 -eps 4 -norm Linf -data "$VSC_DATA_VO/ImageNet" -bs 1 -adapt -trial 273 -iterations 1 -attack simba -timeout 120 -softmax
