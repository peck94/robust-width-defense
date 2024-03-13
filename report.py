import argparse

import json

import numpy as np

from utils import Welford

from pathlib import Path

from tabulate import tabulate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('log', type=str, help='log file')

    args = parser.parse_args()

    orig_acc, adv_acc = Welford(), Welford()
    if Path(args.log).exists():
        with open(args.log, 'r') as log:
            data = json.load(log)
            orig_acc.from_json(data['orig_acc'])
            adv_acc.from_json(data['adv_acc'])
    else:
        raise FileNotFoundError(args.log)
    
    orig_mean, orig_var = orig_acc.values
    orig_err = 1.96 * np.sqrt(orig_var / orig_acc.count)

    adv_mean, adv_var = adv_acc.values
    adv_err = 1.96 * np.sqrt(adv_var / adv_acc.count)

    print(tabulate([
        ['Standard', f'{orig_mean:.2%}', f'{orig_err:.2%}'],
        ['Robust', f'{adv_mean:.2%}', f'{adv_err:.2%}']
    ], headers=['Setting', 'Accuracy', 'Error']))
