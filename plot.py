import argparse

import optuna

import matplotlib.pyplot as plt

from tqdm import tqdm

from matplotlib.lines import Line2D

COLORS = {
    'wavelet': 'red',
    'dtcwt': 'green',
    'fourier': 'blue',
    'shearlet': 'orange'
}

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', type=str, default='cs-test', help='study name')
    parser.add_argument('-results', type=str, default='sqlite:///results.db', help='results database')

    args = parser.parse_args()

    # load study
    study = optuna.load_study(study_name=args.name, storage=args.results)
    
    # load other trials
    obj1s, obj2s, css = [], [], []
    for trial in tqdm(study.trials):
        if trial.values:
            obj1s.append(trial.values[0])
            obj2s.append(trial.values[1])
            css.append(COLORS[trial.params['method']])
    
    # plot trials
    fig, ax = plt.subplots()
    plt.title(args.name)
    plt.scatter(obj1s, obj2s, color=css, s=4)
    plt.xlabel('robust accuracy')
    plt.ylabel('standard accuracy')
    plt.xlim((0, 1))
    plt.ylim((0, 1))

    ax.legend(handles=[Line2D([0], [0], color='w', markerfacecolor=COLORS[c], marker='o', label=c) for c in COLORS])
    plt.savefig(f'plots/{args.name}.pdf')
