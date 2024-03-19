import optuna

import json

import numpy as np

import argparse

import warnings

import torch
import torchvision
import torchvision.transforms as transforms

import robustbench as rb

from smoother import Smoother

from tqdm import tqdm

from attacks import AutoAttack

from reconstruction import Reconstruction

from tabulate import tabulate

from utils import Logger

from pathlib import Path

def main(args):
    # perform checks
    if args.attack == 'simba' and not args.softmax:
        warnings.warn('This attack expects probabilities. Consider passing the -softmax flag.', RuntimeWarning)
    if args.attack == 'autoattack' and args.softmax:
        warnings.warn('This attack expects logits. Consider removing the -softmax flag.', RuntimeWarning)

    # get device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Device: {device}')

    # load data
    tfs = [transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()]
    if not args.rb:
        tfs.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    imagenet_data = torchvision.datasets.ImageNet(args.data, split='val',
                                              transform=transforms.Compose(tfs))
    indices = np.random.permutation(len(imagenet_data))[:args.count]
    subset_data = torch.utils.data.Subset(imagenet_data, indices)
    data_loader = torch.utils.data.DataLoader(subset_data, batch_size=args.bs, shuffle=True, num_workers=1)
    
    # load parameters
    study = optuna.load_study(study_name=args.name, storage=args.results)
    trial = study.trials[args.trial]
    reconstructor = Reconstruction(**trial.params, device=device)
    print(f'Loaded study with parameters: {trial.params}')

    # load model
    if args.rb:
        model = rb.utils.load_model(args.model, dataset='imagenet', threat_model=args.norm).to(device)
    else:
        model = torchvision.models.get_model(args.model, weights=args.weights).to(device)
    model.eval()

    # attack the model
    if args.base:
        defense = model
    else:
        defense = Smoother(model, reconstructor, args.iterations, verbose=False, softmax=args.softmax).to(device)

    adversary = AutoAttack(args, model, defense)

    # load log
    logger = Logger(args.log)
    orig_acc, adv_acc = logger.get_experiment(args)

    # perform attacks
    progbar = tqdm(data_loader)
    for x_batch, y_batch in progbar:
        try:
            x_adv = adversary.generate(x_batch.detach(), y_batch.detach())
            
            with torch.no_grad():
                y_pred_orig = defense(x_batch.detach().to(device)).cpu().detach().numpy()
                y_pred = defense(x_adv.detach().to(device)).cpu().detach().numpy()

            orig_acc.update_all(y_pred_orig.argmax(axis=1) == y_batch.numpy())
            adv_acc.update_all(y_pred.argmax(axis=1) == y_batch.numpy())

            progbar.set_postfix({'standard': orig_acc.values[0], 'robust': adv_acc.values[0]})
            logger.set_experiment(args, orig_acc, adv_acc)
        except RuntimeError as e:
            print(e)

    print()

    orig_mean, orig_var = orig_acc.values
    orig_err = 1.96 * np.sqrt(orig_var / orig_acc.count)

    adv_mean, adv_var = adv_acc.values
    adv_err = 1.96 * np.sqrt(adv_var / adv_acc.count)

    print(tabulate([
        ['Standard', f'{orig_mean:.2%}', f'{orig_err:.2%}'],
        ['Robust', f'{adv_mean:.2%}', f'{adv_err:.2%}']
    ], headers=['Setting', 'Accuracy', 'Error']))

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, default='resnet50', help='model name')
    parser.add_argument('-weights', type=str, default='IMAGENET1K_V1', help='model weights')
    parser.add_argument('-count', type=int, default=1000, help='number of test samples')
    parser.add_argument('-name', type=str, default='cs-test', help='study name')
    parser.add_argument('-results', type=str, default='sqlite:///results.db', help='results database')
    parser.add_argument('-eps', type=int, default=4, help='perturbation budget')
    parser.add_argument('-norm', type=str, default='Linf', help='threat model')
    parser.add_argument('-data', type=str, default='/scratch/jpeck/imagenet', help='ImageNet path')
    parser.add_argument('-bs', type=int, default=16, help='batch size')
    parser.add_argument('-adapt', action='store_true', default=False, help='perform adaptive attack')
    parser.add_argument('-trial', type=int, default=0, help='Optuna trial to load')
    parser.add_argument('-iterations', type=int, default=10, help='number of iterations of smoothing')
    parser.add_argument('-rb', action='store_true', default=False, help='use RobustBench models')
    parser.add_argument('-attack', choices=['square', 'apgd'], help='adversarial attack to run')
    parser.add_argument('-softmax', action='store_true', default=False, help='predict softmax probabilities')
    parser.add_argument('-log', type=str, default='output.json', help='output log')
    parser.add_argument('-base', action='store_true', default=False, help='do not apply any defense')
    parser.add_argument('-overwrite', action='store_true', default=False, help='overwrite completed experiments')

    args = parser.parse_args()

    main(args)
