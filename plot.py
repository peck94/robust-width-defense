import argparse

import optuna

import matplotlib.pyplot as plt

import numpy as np

from tqdm import tqdm

from matplotlib.lines import Line2D

COLORS = {
    'wavelet': 'red',
    'dtcwt': 'green',
    'fourier': 'blue',
    'shearlet': 'orange'
}

def plot_info(trial, xt, yt):
    x = trial.values[0]
    y = trial.values[1]
    params = trial.params
    if 'levels' in params:
        levels = params['levels']
    else:
        levels = params['scales']
    plt.plot([x, xt], [y, yt], color='black', linewidth=1)
    plt.text(xt, yt,
             f'Method: {params["method"]}\nScales: {levels}\nThreshold: {params["mu"]:.2f}\nSubsampling: {params["q"]:.2%}\nIterations: {params["iterations"]}',
             backgroundcolor='black', color='cyan', fontsize='medium', fontweight='bold', fontfamily='monospace')


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', type=str, default='cs-test', help='study name')
    parser.add_argument('-results', type=str, default='sqlite:///results.db', help='results database')
    parser.add_argument('-method', type=str, default=None, help='filter method')

    args = parser.parse_args()

    # load study
    study = optuna.load_study(study_name=args.name, storage=args.results)
    
    # load trials
    obj1s, obj2s, css = [], [], []
    for trial in tqdm(study.trials):
        if trial.values:
            if args.method is None or args.method == trial.params['method']:
                obj1s.append(trial.values[0])
                obj2s.append(trial.values[1])
                css.append(COLORS[trial.params['method']])
    
    best1s, best2s = [], []
    for trial in tqdm(study.best_trials):
        if trial.values:
            if args.method is None or args.method == trial.params['method']:
                best1s.append(trial.values[0])
                best2s.append(trial.values[1])
    idx = np.argsort(best1s)
    best1s = np.array(best1s)[idx]
    best2s = np.array(best2s)[idx]
    
    # plot trials
    fig, ax = plt.subplots()
    plt.plot([0, 1], [0, 1], color='black', ls=':', alpha=.5)
    plt.plot(best1s, best2s, color='black', linewidth=2)
    plt.scatter(obj1s, obj2s, color=css, s=4)
    plt.xlabel('robust accuracy')
    plt.ylabel('standard accuracy')
    plt.xlim((0, 1))
    plt.ylim((0, 1))

    plot_info(study.best_trials[idx[0]], .1, .9)
    plot_info(study.best_trials[idx[len(study.best_trials) // 3]], .6, .9)
    plot_info(study.best_trials[idx[len(study.best_trials) // 2]], .4, .1)
    plot_info(study.best_trials[idx[-1]], .8, .5)

    if args.method:
        ax.legend(handles=[Line2D([0], [0], color='w', markerfacecolor=COLORS[args.method], marker='o', label=args.method)])
    else:
        ax.legend(handles=[Line2D([0], [0], color='w', markerfacecolor=COLORS[c], marker='o', label=c) for c in COLORS])
    plt.tight_layout()
    plt.savefig(f'plots/{args.name}.pdf')
