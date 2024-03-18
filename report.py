import argparse

import json

import numpy as np

import matplotlib.pyplot as plt

from utils import Welford

from pathlib import Path

from tabulate import tabulate

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
    'wide_resnet101': 'WideResNet101'
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('log', type=str, help='location of log files')
    parser.add_argument('-latex', action='store_true', default=False, help='output LaTeX code')
    parser.add_argument('-plot', action='store_true', default=False, help='plot scores')

    args = parser.parse_args()

    if args.plot:
        files = []
        for d in braceexpand(args.log):
            files += glob(f'{d}/*.json')
        if len(files) == 0:
            raise FileNotFoundError(args.log)

        names = []
        orig_accs, adv_accs = [], []
        orig_errs, adv_errs = [], []
        for filename in files:
            with open(filename, 'r') as f:
                data = json.load(f)
                w1, w2 = Welford(), Welford()
                w1.from_json(data['orig_acc'])
                w2.from_json(data['adv_acc'])

                orig_mean, orig_var = w1.values
                orig_err = 1.96 * np.sqrt(orig_var / w1.count)

                adv_mean, adv_var = w2.values
                adv_err = 1.96 * np.sqrt(adv_var / w2.count)

                orig_accs.append(orig_mean)
                orig_errs.append(orig_err)

                adv_accs.append(adv_mean)
                adv_errs.append(adv_err)

                names.append(MAPPING[filename.split('/')[-1].split('.')[0]])
        
        X_axis = np.arange(len(orig_accs))
        idx = np.argsort(adv_accs)[::-1]
  
        plt.bar(X_axis - 0.2, np.array(orig_accs)[idx], 0.4, label='standard', yerr=orig_errs)
        plt.bar(X_axis + 0.2, np.array(adv_accs)[idx], 0.4, label='robust', yerr=adv_errs)
        
        plt.xticks(X_axis, np.array(names)[idx], rotation=45, ha='right')
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
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

        if args.latex:
            print(f'{100*orig_mean:.2f}\\% $\\pm$ {100*orig_err:.2f}\\% & {100*adv_mean:.2f}\\% $\\pm$ {100*adv_err:.2f}\\%')
        else:
            print(tabulate([
                ['Standard', f'{orig_mean:.2%}', f'{orig_err:.2%}'],
                ['Robust', f'{adv_mean:.2%}', f'{adv_err:.2%}']
            ], headers=['Setting', 'Accuracy', 'Error']))
