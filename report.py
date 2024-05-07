import argparse

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

from utils import Logger

from glob import glob

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
ORDER = [
    'wide_resnet101', 'resnet50', 'vit_b_16', 'swin_t',
    'liu2023comprehensive_swin-l', 'debenedetti2022light_xcit-l12', 'peng2023robust', 'wong2020fast'
]

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('log', type=str, help='location of log files', nargs='+')
    parser.add_argument('-norm', type=str, choices=['Linf', 'L2'], default='Linf', help='norm to report')
    parser.add_argument('-out', type=str, help='save plot to this location')

    args = parser.parse_args()

    # load data files
    files = []
    for d in args.log:
        files += glob(f'{d}/*.json')
    if len(files) == 0:
        raise FileNotFoundError(args.log)
    files = [filename for filename in files if filename.split('/')[-1].split('.')[0] in MAPPING]

    # parse results
    epses = set()
    for i, filename in enumerate(files):
        # load data
        model_name = filename.split('/')[-1].split('.')[0]
        with open(filename, 'r') as f:
            logger = Logger(filename)
        
        experiments = logger.get_experiments(sort=True, norm=args.norm)
        for item in experiments:
            epses.add(item['eps'])
    epses = sorted(list(epses))

    results = {
        'Standard': []
    }
    raw_data = {}
    for eps in epses:
        results[f'eps = {eps}'] = [pd.NA for _ in files]
    model_names = []
    for i, filename in enumerate(files):
        # load data
        model_name = filename.split('/')[-1].split('.')[0]
        model_names.append(MAPPING[model_name])
        with open(filename, 'r') as f:
            logger = Logger(filename)
        
        experiments = logger.get_experiments(sort=True, norm=args.norm)
        raw_data[model_name] = experiments
        results['Standard'].append(f"{experiments[0]['orig_acc'].mean:.2%} +- {experiments[0]['orig_acc'].sem:.2%}")
        for item in experiments:
            results[f"eps = {item['eps']}"][i] = f"{item['adv_acc'].mean:.2%} +- {item['adv_acc'].sem:.2%}"
    
    # show table
    print(pd.DataFrame(results, index=model_names))

    # plot results
    plt.clf()
    plt.ylim(0, 1)

    names = [model_name for model_name in ORDER if model_name in raw_data]
    X_axis = np.arange(len(names))
    for i, model_name in enumerate(names):
        experiments = raw_data[model_name]
        width = .2
        start = -2*width

        plt.bar(X_axis[i] + start, experiments[0]['orig_acc'].mean, width, yerr=experiments[0]['orig_acc'].sem, color='cyan')
        for j in range(len(experiments)):
            rect = plt.bar(X_axis[i] + start + (j + 1)*width, experiments[j]['adv_acc'].mean, width, yerr=experiments[j]['adv_acc'].sem, color='red')[0]
            height = rect.get_height() + experiments[j]['adv_acc'].sem
            plt.text(rect.get_x() + rect.get_width() / 2.0, height, experiments[j]['eps'], ha='center', va='bottom',
                     fontfamily='monospace', fontsize='x-small', fontweight='bold')
    
    plt.xticks(X_axis, [MAPPING[name] for name in names], rotation=45, ha='right')

    cyan_patch = mpatches.Patch(color='cyan', label='standard')
    red_patch = mpatches.Patch(color='red', label='robust')
    plt.legend(handles=[cyan_patch, red_patch])

    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.tight_layout()

    # save or show
    if args.out:
        plt.savefig(args.out)
    else:
        plt.show()
