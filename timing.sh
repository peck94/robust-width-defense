#!/bin/bash

data=${1:-"$VSC_DATA_VO/ImageNet"}

# WideResNet101 + CS
python timing.py -model wide_resnet101_2 -weights IMAGENET1K_V2 -results sqlite:///fourier.db -name wide_resnet101 -data "$data" -bs 16 -trial 106 -cs

# WideResNet101 + RS
python timing.py -model wide_resnet101_2 -weights IMAGENET1K_V2 -results sqlite:///fourier.db -name wide_resnet101 -data "$data" -bs 16 -trial 106 -rs

# ResNet50 + CS
python timing.py -model resnet50 -weights IMAGENET1K_V2 -results sqlite:///fourier.db -name resnet50 -data "$data" -bs 16 -trial 273 -cs

# ResNet50 + RS
python timing.py -model resnet50 -weights IMAGENET1K_V2 -results sqlite:///fourier.db -name resnet50 -data "$data" -bs 16 -trial 273 -rs
