import optuna

import numpy as np

import argparse

import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms

import robustbench as rb

from reconstruction import Reconstruction

from tqdm import tqdm

from autoattack import AutoAttack

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', type=str, default='cs-test', help='study name')
    parser.add_argument('-results', type=str, default='sqlite:///results.db', help='results database')
    parser.add_argument('-data', type=str, help='ImageNet path')
    parser.add_argument('-bs', type=int, default=16, help='batch size')
    parser.add_argument('-trial', type=int, default=0, help='Optuna trial to load')
    parser.add_argument('-model', type=str, default='resnet50', help='model name')
    parser.add_argument('-weights', type=str, default='IMAGENET1K_V2', help='model weights')
    parser.add_argument('-rb', action='store_true', default=False, help='use RobustBench models')
    parser.add_argument('-norm', default='Linf', choices=['L2', 'Linf'], help='perturbation norm')

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
    data_loader = torch.utils.data.DataLoader(imagenet_data, batch_size=args.bs, shuffle=True, num_workers=1)
    
    # load parameters
    study = optuna.load_study(study_name=args.name, storage=args.results)
    trial = study.trials[args.trial]
    reconstructor = Reconstruction(**trial.params, device=device)
    print(f'Loaded study with parameters: {trial.params}')

    # load model
    if args.rb:
        model = rb.utils.load_model(args.model, dataset='imagenet', threat_model=args.norm).to(device).eval()
    else:
        model = torchvision.models.get_model(args.model, weights=args.weights).to(device).eval()

    # determine initial robustness
    adversary = AutoAttack(model, norm=args.norm, version='custom')
    adversary.attacks_to_run = ['fab']
    taus = []
    for images, labels in tqdm(data_loader):
        x_adv = adversary.run_standard_evaluation(images, labels, bs=args.bs)
        taus.append(np.sqrt(np.square(images - x_adv).sum(axis=[1, 2, 3])))
    taus = np.concatenate(taus)
    print(f'tau = {taus.mean():.2f}')

    # compute expected sparsity defect
    total_bound = 0
    total = 0
    progbar = tqdm(data_loader)
    for x_batch, y_batch in progbar:
        total_bound += reconstructor.certify(x_batch.to(device)).item()
        total += x_batch.shape[0]
        progbar.set_postfix({'bound': total_bound / total})
    defect = total_bound / total
    print(f'defect = {defect:.2f}')

    # compute certified radius
    n = np.prod(x_adv.shape[1:])
    eta = .01
    sigma = 1.
    rs = .5 * (taus * np.sqrt(n) - defect / (eta * sigma**2))
    print(f'radius = {rs.mean():.2f}')

    plt.hist(rs, bins='auto')
    plt.xlabel('radius')
    plt.ylabel('count')
    plt.savefig(f'{args.model}_radii.pdf')
