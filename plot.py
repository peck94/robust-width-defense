import argparse

import optuna

import matplotlib.pyplot as plt

import numpy as np

from sklearn.isotonic import IsotonicRegression

from tqdm import tqdm

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', type=str, default='cs-test', help='study name')
    parser.add_argument('-results', type=str, default='sqlite:///results.db', help='results database')

    args = parser.parse_args()

    # load study
    study = optuna.load_study(study_name=args.name, storage=args.results)

    # load Pareto front
    obj1, obj2 = [], []
    for trial in tqdm(study.best_trials):
        obj1.append(trial.values[0])
        obj2.append(trial.values[1])
    
    # load other trials
    obj1s, obj2s = [], []
    for trial in tqdm(study.trials):
        if trial.values:
            obj1s.append(trial.values[0])
            obj2s.append(trial.values[1])
    
    # plot trials
    isoreg = IsotonicRegression(y_min=0, y_max=1, increasing=False, out_of_bounds='clip').fit(obj1, obj2)

    xs = np.linspace(0, 1, 100)
    ys = isoreg.predict(xs)
    
    plt.title(args.name)
    plt.scatter(obj1s, obj2s, color='black')
    plt.scatter(obj1, obj2, color='red')
    plt.plot(xs, ys, ls='--', color='blue')
    plt.xlabel('robust accuracy')
    plt.ylabel('standard accuracy')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.savefig(f'plots/{args.name}.pdf')
