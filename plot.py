import argparse

import optuna

import matplotlib.pyplot as plt

import numpy as np

from sklearn.isotonic import IsotonicRegression

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

    # load Pareto front
    obj1, obj2, cs = [], [], []
    for trial in tqdm(study.best_trials):
        obj1.append(trial.values[0])
        obj2.append(trial.values[1])
        cs.append(COLORS[trial.params['method']])
    
    # load other trials
    obj1s, obj2s, css = [], [], []
    for trial in tqdm(study.trials):
        if trial.values:
            obj1s.append(trial.values[0])
            obj2s.append(trial.values[1])
            css.append(COLORS[trial.params['method']])
    
    # plot trials
    isoreg = IsotonicRegression(y_min=0, y_max=1, increasing=False, out_of_bounds='clip').fit(obj1, obj2)
    p = np.polyfit(obj1, obj2, deg=1)
    u1, u2 = np.max(obj1), np.argmax(obj1)

    x1s = np.linspace(0, u1, 100)
    ys = isoreg.predict(x1s)

    x2s = np.linspace(u1, 1, 100)
    zs = np.polyval(p, x2s)
    
    fig, ax = plt.subplots()
    plt.title(args.name)
    plt.scatter(obj1s, obj2s, color=css)
    plt.plot(x1s, ys, ls='--', color='black')
    plt.plot(x2s, zs, ls='--', color='black')
    plt.xlabel('robust accuracy')
    plt.ylabel('standard accuracy')
    plt.xlim((0, 1))
    plt.ylim((0, 1))

    ax.legend(handles=[Line2D([0], [0], color=COLORS[c], marker='o', label=c) for c in COLORS])
    plt.savefig(f'plots/{args.name}.pdf')
