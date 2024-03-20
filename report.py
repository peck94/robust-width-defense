import argparse

import json

import numpy as np

import matplotlib.pyplot as plt

from utils import Logger

from pathlib import Path

from glob import glob

from braceexpand import braceexpand

MAPPING = {
    'wong2020fast': 'Wong et al. (2020)',
    'peng2023robust': 'Peng et al. (2023)',
    'liu2023comprehensive_swin-l': 'Liu et al. (2023)',
    'debenedetti2022light_xcit-l12': 'Debenedetti et al. (2023)',

    'resnet50': 'ResNet50',
    'swin_t': 'Swin Transformer',
    'vit_b_16': 'Vision Transformer',
    'wide_resnet101': 'WRN-101-2'
}

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('log', type=str, help='location of log files')
    parser.add_argument('-out', type=str, help='save plot to this location')

    args = parser.parse_args()

    # load data files
    files = []
    for d in braceexpand(args.log):
        files += glob(f'{d}/*.json')
    if len(files) == 0:
        raise FileNotFoundError(args.log)

    # parse results
    results = {}
    for filename in files:
        model_name = filename.split('/')[-1].split('.')[0]
        results[model_name] = {
            'eps': [],
            'orig_acc': [],
            'adv_acc': []
        }

        with open(filename, 'r') as f:
            logger = Logger(filename)
        
        experiments = logger.get_experiments(sort=True)
        for experiment in experiments:
            results[model_name]['eps'].append(experiment['eps'])
            results[model_name]['orig_acc'].append(experiment['orig_acc'].mean)
            results[model_name]['adv_acc'].append(experiment['adv_acc'].mean)
    
    # plot results
    plt.clf()
    plt.ylim(0, 1)
    for model_name in results:
        label = MAPPING[model_name]
        data = results[model_name]
        plt.plot(data['eps'], data['adv_acc'], label=label, marker='o')
    plt.legend()
    plt.tight_layout()

    # save or show
    if args.out:
        plt.savefig(args.out)
    else:
        plt.show()
