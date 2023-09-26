import optuna

import argparse

import torch
import torchvision
import torchvision.transforms as transforms

from utils import Wrapper

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
    
    # load best parameters
    study = optuna.load_study(study_name=args.name, storage=args.results)
    reconstructor = Reconstruction(**study.best_params)
    print(f'Loaded study with parameters: {study.best_params}')

    # load model
    model = torch.hub.load('pytorch/vision', args.model, weights=args.weights).to(device)

    # attack the model
    if args.adapt:
        defense = Wrapper(model, reconstructor).to(device)
        adversary = AutoAttack(defense, norm='Linf', eps=args.eps/255, version='rand')
    else:
        adversary = AutoAttack(model, norm='Linf', eps=args.eps/255, version='standard')

    total = 0
    orig_acc = 0
    adv_rec_acc = 0
    progbar = tqdm(data_loader)
    for x_batch, y_batch in progbar:
        x_adv = adversary.run_standard_evaluation(x_batch.to(device), y_batch.to(device), bs=x_batch.shape[0])

        if args.adapt:
            y_pred_orig = defense(x_batch.to(device)).cpu().detach().numpy()
            y_pred = defense(x_adv).cpu().detach().numpy()
        else:
            x_orig = reconstructor.generate(x_batch.to(device))
            x_rec = reconstructor.generate(x_adv)

            y_pred_orig = model(x_orig.float()).cpu().detach().numpy()
            y_pred = model(x_rec.float()).cpu().detach().numpy()

        orig_acc += (y_pred_orig.argmax(axis=1) == y_batch.numpy()).sum()
        adv_rec_acc += (y_pred.argmax(axis=1) == y_batch.numpy()).sum()
        total += x_batch.shape[0]

        progbar.set_postfix({'orig_acc': orig_acc/total, 'adv_rec_acc': adv_rec_acc/total})
