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

def get_radius(taus, defects, n, eta, sigma):
    return np.maximum(0, .5 * (taus * np.sqrt(n) - defects / (eta * sigma**2)))

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
    parser.add_argument('-output', default='.', help='output data directory')
    parser.add_argument('-plot', action='store_true', default=False, help='only plot existing results')
    parser.add_argument('-norm', default='Linf', choices=['L2', 'Linf'], help='threat model')
    parser.add_argument('-eta', type=float, default=.01, help='eta value')
    parser.add_argument('-sigma', type=float, default=1, help='upper frame bound')
    parser.add_argument('-eps', default=2, type=float, help='perturbation bound')
    parser.add_argument('-acc', default=1, type=float, help='baseline accuracy')
    parser.add_argument('-simplify', default=False, action='store_true', help='use simplified bound')

    args = parser.parse_args()

    # get device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Device: {device}')

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

    if not args.plot:
        # load data
        imagenet_data = torchvision.datasets.ImageNet(args.data, split='val',
                                                transform=transforms.Compose([
                                                    transforms.Resize(256),
                                                    transforms.CenterCrop(224),
                                                    transforms.ToTensor()]))
        data_loader = torch.utils.data.DataLoader(imagenet_data, batch_size=args.bs, shuffle=True, num_workers=1)

        # determine initial robustness and expected sparsity defect
        adversary = AutoAttack(model, norm=args.norm, version='custom', eps=args.eps, device=device)
        adversary.attacks_to_run = ['fab-t']

        taus = []
        defects, errs = [], []
        progbar = tqdm(data_loader)
        for images, labels in progbar:
            x_adv = adversary.run_standard_evaluation(images, labels, bs=args.bs)

            delta = (images.numpy() - x_adv.numpy()).reshape([images.shape[0], -1])
            if args.norm == 'L2':
                taus.append(np.linalg.norm(delta, axis=1))
            else:
                taus.append(abs(delta).max(axis=1))

            mu, err = reconstructor.certify(images.to(device))
            defects.append(mu)
            errs.append(err)

            np.savez(f'{args.output}/{args.model}_cert.npz',
                    taus=np.concatenate(taus),
                    defects=np.concatenate(defects),
                    errs=np.concatenate(errs))
        taus = np.concatenate(taus)
        defects = np.concatenate(defects)
        errs = np.concatenate(errs)
    else:
        data = np.load(f'{args.output}/{args.model}_cert.npz')
        taus = data['taus']
        defects = data['defects']
        errs = data['errs']

    # compute certified radius
    n = 224*224*3

    if args.simplify:
        rs = np.sqrt(n) * taus / 2
    else:
        rs = get_radius(taus, defects, n, args.eta, args.sigma)
    print(f'Accuracy: {args.acc:.2%}')
    print(f'tau = {taus.mean():.2f}')
    print(f'defect = {defects.mean():.2f} +- {errs.mean():.2f}')
    print(f'radius: {rs.mean():.2f}')

    np.savez(f'{args.output}/{args.model}_radii.npz', rs=rs)

    zs = np.linspace(0, np.quantile(rs, .95), 1000)
    qs = np.array([(rs >= z).mean() for z in zs])
    ts = np.array([(taus >= z).mean() for z in zs])

    ax = plt.subplot()
    ax.plot(zs, args.acc * qs, label='robust')
    ax.plot(zs, args.acc * ts, label='original')
    ax.set_xlabel('epsilon')
    ax.set_ylabel('certified accuracy')
    ax.set_ylim(0, 1)
    plt.legend()
    plt.savefig(f'{args.output}/{args.model}_radii.pdf')
