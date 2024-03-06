import optuna

import json

import numpy as np

import argparse

import warnings

import asyncio

import torch
import torchvision
import torchvision.transforms as transforms

import robustbench as rb

from smoother import Smoother

from tqdm import tqdm

from attacks import AutoAttack, SimBA

from reconstruction import Reconstruction

from tabulate import tabulate

from utils import Welford

from pathlib import Path

async def main(args):
    # perform checks
    if args.attack == 'simba' and not args.softmax:
        warnings.warn('This attack expects probabilities. Consider passing the -softmax flag.', RuntimeWarning)
    if args.attack == 'autoattack' and args.softmax:
        warnings.warn('This attack expects logits. Consider removing the -softmax flag.', RuntimeWarning)

    # get device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Device: {device}')

    # load data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    imagenet_data = torchvision.datasets.ImageNet(args.data, split='val',
                                              transform=transforms.Compose([
                                                  transforms.Resize(256),
                                                  transforms.CenterCrop(224),
                                                  transforms.ToTensor(),
                                                  normalize]))
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
    defense = Smoother(model, reconstructor, args.iterations, verbose=False, softmax=args.softmax).to(device)

    if args.attack == 'autoattack':
        adversary = AutoAttack(args, model, defense)
    elif args.attack == 'simba':
        adversary = SimBA(args, model, defense)

    # load log
    orig_acc = Welford()
    adv_acc = Welford()
    if Path(args.log).exists():
        with open(args.log, 'r') as log:
            data = json.load(log)
            orig_acc.from_json(data['orig_acc'])
            adv_acc.from_json(data['adv_acc'])

    # perform attacks
    progbar = tqdm(data_loader)
    loop = asyncio.get_event_loop()
    timeout = 0
    for x_batch, y_batch in progbar:
        try:
            x_adv = await asyncio.wait_for(loop.run_in_executor(None, lambda: adversary.generate(x_batch.detach(), y_batch.detach())), timeout=args.timeout)
            
            with torch.no_grad():
                y_pred_orig = defense(x_batch.detach().to(device)).cpu().detach().numpy()
                y_pred = defense(x_adv.detach().to(device)).cpu().detach().numpy()

            orig_acc.update_all(y_pred_orig.argmax(axis=1) == y_batch.numpy())
            adv_acc.update_all(y_pred.argmax(axis=1) == y_batch.numpy())

            progbar.set_postfix({'orig_acc': orig_acc.values[0], 'adv_rec_acc': adv_acc.values[0], 'timeout': timeout})
            with open(args.log, 'w') as log:
                orig_mean, orig_var = orig_acc.values
                orig_err = 1.96 * np.sqrt(orig_var / orig_acc.count)

                adv_mean, adv_var = adv_acc.values
                adv_err = 1.96 * np.sqrt(adv_var / adv_acc.count)

                print(json.dumps({
                    'orig_acc': orig_acc.to_json(),
                    'adv_acc': adv_acc.to_json()
                }, sort_keys=True, indent=4), file=log)
        except asyncio.TimeoutError:
            print('Timed out.')
            timeout += 1
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
    parser.add_argument('-weights', type=str, default='IMAGENET1K_V2', help='model weights')
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
    parser.add_argument('-attack', choices=['autoattack', 'simba'], help='adversarial attack to run')
    parser.add_argument('-softmax', action='store_true', default=False, help='predict softmax probabilities')
    parser.add_argument('-timeout', default=None, type=int, help='per-batch timeout for attacks (in seconds)')
    parser.add_argument('-log', type=str, default='output.json', help='output log')

    args = parser.parse_args()

    asyncio.run(main(args))
