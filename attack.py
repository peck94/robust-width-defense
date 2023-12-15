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

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, default='resnet50', help='model name')
    parser.add_argument('-weights', type=str, default='IMAGENET1K_V2', help='model weights')
    parser.add_argument('-name', type=str, default='cs-test', help='study name')
    parser.add_argument('-results', type=str, default='sqlite:///results.db', help='results database')
    parser.add_argument('-eps', type=int, default=4, help='perturbation budget')
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
        adversary = AutoAttack(defense, norm='Linf', eps=args.eps/255, version='rand')
    else:
        adversary = AutoAttack(model, norm='Linf', eps=args.eps/255, version='standard')

    total = 0
    orig_acc = 0
    adv_rec_acc = 0
    progbar = tqdm(data_loader)
    for x_batch, y_batch in progbar:
        x_adv = adversary.run_standard_evaluation(x_batch.to(device), y_batch.to(device), bs=x_batch.shape[0])

        with torch.no_grad():
            y_pred_orig = defense(x_batch.to(device)).cpu().detach().numpy()
            y_pred = defense(x_adv).cpu().detach().numpy()

        orig_acc += (y_pred_orig.argmax(axis=1) == y_batch.numpy()).sum()
        adv_rec_acc += (y_pred.argmax(axis=1) == y_batch.numpy()).sum()
        total += x_batch.shape[0]

        progbar.set_postfix({'orig_acc': orig_acc/total, 'adv_rec_acc': adv_rec_acc/total})
