import argparse

import optuna

import matplotlib.pyplot as plt

import numpy as np

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', type=str, default='cs-test', help='study name')
    parser.add_argument('-results', type=str, default='sqlite:///results.db', help='results database')

    args = parser.parse_args()

    # load study
    study = optuna.load_study(study_name=args.name, storage=args.results)

    # plot Pareto front
    obj1, obj2 = [], []
    for trial in study.best_trials:
        obj1.append(trial.values[0])
        obj2.append(trial.values[1])
    
    p = np.polyfit(obj1, obj2, deg=2)
    xs = np.linspace(0, 1, 100)
    ys = np.polyval(p, xs)
    
    plt.title(args.name)
    plt.scatter(obj1, obj2, color='red')
    plt.plot(xs, ys, ls='--', color='blue')
    plt.xlabel('robust accuracy')
    plt.ylabel('standard accuracy')
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.savefig(f'plots/{args.name}.pdf')
