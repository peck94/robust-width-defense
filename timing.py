import optuna

import numpy as np

import argparse

import torch
import torchvision
import torchvision.transforms as transforms

import robustbench as rb

import time

from smoother import Smoother

from tqdm import tqdm

from reconstruction import Reconstruction

from tabulate import tabulate

from utils import Welford

from art.estimators.certification.randomized_smoothing import PyTorchRandomizedSmoothing

def main(args):
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

    if args.cs:
        model = Smoother(model, reconstructor).to(device)
    if args.rs:
        model = PyTorchRandomizedSmoothing(
            model,
            loss=torch.nn.CrossEntropyLoss(),
            input_shape=[3, 224, 224],
            nb_classes=1000)

    # time the model
    progbar = tqdm(data_loader)
    base_time = Welford()
    for x_batch, _ in progbar:
        if hasattr(model, 'predict'):
            start = time.time()
            model.predict(x_batch)
            end = time.time()
        else:
            x_gpu = x_batch.detach().to(device)
            start = time.time()
            model(x_gpu)
            end = time.time()
        base_time.update(end - start)

        progbar.set_postfix({'base': base_time.values[0]})

    print()

    suffix = ''
    if args.cs:
        suffix = '+ CS'
    if args.rs:
        suffix = '+ RS'

    print(tabulate([
        [f'{args.model} {suffix}', base_time.values[0], base_time.sem],
    ], headers=['Model', 'Mean', 'Error']))

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, default='resnet50', help='model name')
    parser.add_argument('-weights', type=str, default='IMAGENET1K_V1', help='model weights')
    parser.add_argument('-count', type=int, default=1000, help='number of test samples')
    parser.add_argument('-name', type=str, default='cs-test', help='study name')
    parser.add_argument('-results', type=str, default='sqlite:///results.db', help='results database')
    parser.add_argument('-data', type=str, help='ImageNet path')
    parser.add_argument('-bs', type=int, default=16, help='batch size')
    parser.add_argument('-trial', type=int, default=0, help='Optuna trial to load')
    parser.add_argument('-rb', action='store_true', default=False, help='use RobustBench models')
    parser.add_argument('-norm', type=str, default='Linf', help='threat model')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-cs', action='store_true', default=False, help='apply robust width defense')
    group.add_argument('-rs', action='store_true', default=False, help='apply randomized smoothing')

    args = parser.parse_args()

    main(args)
