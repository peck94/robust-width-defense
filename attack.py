import optuna

import numpy as np

import argparse

import torch
import torchvision
import torchvision.transforms as transforms

from smoother import Smoother

from tqdm import tqdm

from autoattack import AutoAttack

from reconstruction import Reconstruction

from tabulate import tabulate

from utils import Welford

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, default='resnet50', help='model name')
    parser.add_argument('-weights', type=str, default='IMAGENET1K_V2', help='model weights')
    parser.add_argument('-name', type=str, default='cs-test', help='study name')
    parser.add_argument('-results', type=str, default='sqlite:///results.db', help='results database')
    parser.add_argument('-eps', type=int, default=4, help='perturbation budget')
    parser.add_argument('-norm', type=str, default='Linf', help='threat model')
    parser.add_argument('-data', type=str, default='/scratch/jpeck/imagenet', help='ImageNet path')
    parser.add_argument('-bs', type=int, default=16, help='batch size')
    parser.add_argument('-adapt', action='store_true', default=False, help='perform adaptive attack')
    parser.add_argument('-trial', type=int, default=0, help='Optuna trial to load')

    args = parser.parse_args()

    # get device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Device: {device}')

    # load data
    imagenet_data = torchvision.datasets.ImageNet(args.data, split='val',
                                              transform=transforms.Compose([
                                                  transforms.Resize(256),
                                                  transforms.CenterCrop(224),
                                                  transforms.ToTensor()]))
    indices = np.random.permutation(len(imagenet_data))[:5000]
    subset_data = torch.utils.data.Subset(imagenet_data, indices)
    data_loader = torch.utils.data.DataLoader(subset_data, batch_size=args.bs, shuffle=True, num_workers=1)
    
    # load parameters
    study = optuna.load_study(study_name=args.name, storage=args.results)
    trial = study.trials[args.trial]
    reconstructor = Reconstruction(**trial.params, device=device)
    print(f'Loaded study with parameters: {trial.params}')

    # load model
    model = torch.hub.load('pytorch/vision', args.model, weights=args.weights).to(device)

    # attack the model
    defense = Smoother(model, reconstructor).to(device)
    if args.adapt:
        adversary = AutoAttack(defense, norm=args.norm, eps=args.eps/255, version='rand')
    else:
        adversary = AutoAttack(model, norm=args.norm, eps=args.eps/255, version='standard')

    orig_acc = Welford()
    adv_acc = Welford()
    progbar = tqdm(data_loader)
    for x_batch, y_batch in progbar:
        x_adv = adversary.run_standard_evaluation(x_batch.to(device), y_batch.to(device), bs=x_batch.shape[0])

        with torch.no_grad():
            y_pred_orig = defense(x_batch.to(device)).cpu().detach().numpy()
            y_pred = defense(x_adv).cpu().detach().numpy()

        orig_acc.update_all(y_pred_orig.argmax(axis=1) == y_batch.numpy())
        adv_acc.update_all(y_pred.argmax(axis=1) == y_batch.numpy())

        progbar.set_postfix({'orig_acc': orig_acc.values[0], 'adv_rec_acc': adv_acc.values[0]})

    print()

    orig_mean, orig_var = orig_acc.values
    orig_err = 1.96 * np.sqrt(orig_var / orig_acc.count)

    adv_mean, adv_var = adv_acc.values
    adv_err = 1.96 * np.sqrt(adv_var / adv_acc.count)

    print(tabulate([
        ['Standard', f'{orig_mean:.2%}', f'{orig_err:.2%}'],
        ['Robust', f'{adv_mean:.2%}', f'{adv_err:.2%}']
    ], headers=['Setting', 'Accuracy', 'Error']))
