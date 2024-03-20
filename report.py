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
    plt.clf()
    plt.ylim(0, 1)

    results = {}
    X_axis = np.arange(len(files))
    names = []
    for i, filename in enumerate(files):
        # load data
        model_name = filename.split('/')[-1].split('.')[0]
        names.append(MAPPING[model_name])
        results[model_name] = {
            'eps': [],
            'orig_acc': [],
            'adv_acc': []
        }

        with open(filename, 'r') as f:
            logger = Logger(filename)
        
        # plot experiments
        experiments = logger.get_experiments(sort=True)
        width = .1
        start = -2*width

        plt.bar(X_axis[i] + start, experiments[0]['orig_acc'].mean, width, yerr=experiments[0]['orig_acc'].sem, color='cyan')
        for j, experiment in enumerate(experiments):
            rect = plt.bar(X_axis[i] + start + (j + 1)*width, experiment['adv_acc'].mean, width, yerr=experiment['adv_acc'].sem, color='red')[0]
            height = rect.get_height() + experiment['adv_acc'].sem
            plt.text(rect.get_x() + rect.get_width() / 2.0, height, experiment['eps'], ha='center', va='bottom')
    
    plt.xticks(X_axis, names, rotation=45, ha='right')
    plt.tight_layout()

    # save or show
    if args.out:
        plt.savefig(args.out)
    else:
        plt.show()
