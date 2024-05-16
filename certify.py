import optuna

import numpy as np

import argparse

import torch
import torchvision
import torchvision.transforms as transforms

from reconstruction import Reconstruction

from tqdm import tqdm

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', type=str, default='cs-test', help='study name')
    parser.add_argument('-results', type=str, default='sqlite:///results.db', help='results database')
    parser.add_argument('-data', type=str, help='ImageNet path')
    parser.add_argument('-bs', type=int, default=16, help='batch size')
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

    # compute bounds
    total_bound = 0
    total = 0
    progbar = tqdm(data_loader)
    for x_batch, y_batch in progbar:
        total_bound += reconstructor.certify(x_batch.to(device)).item()
        total += x_batch.shape[0]
        progbar.set_postfix({'bound': total_bound / total})
    
    print(f'Bound: {total_bound / total:.2f}')
