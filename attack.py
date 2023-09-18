import optuna

import argparse

import torch
import torchvision
import torchvision.transforms as transforms

from utils import Wrapper, generate_reconstructions, normalize

from tqdm import tqdm

from autoattack import AutoAttack

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', type=str, default='cs-test', help='study name')
    parser.add_argument('-results', type=str, default='sqlite:///results.db', help='results database')
    parser.add_argument('-eps', type=int, default=4, help='perturbation budget')
    parser.add_argument('-data', type=str, default='/scratch/jpeck/imagenet', help='ImageNet path')
    parser.add_argument('-version', type=str, default='standard', help='AutoAttack version')
    parser.add_argument('-bs', type=int, default=16, help='batch size')

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
    print(f'Loaded study with parameters: {study.best_params}')

    # load model
    weights = torchvision.models.Swin_B_Weights.DEFAULT
    preprocess = weights.transforms(antialias=True)
    raw_model = torchvision.models.swin_b(weights=weights)
    model = Wrapper(preprocess, raw_model).to(device)

    # attack the model
    adversary = AutoAttack(model, norm='Linf', eps=args.eps/255, version=args.version)
    total = 0
    orig_acc = 0
    adv_rec_acc = 0
    progbar = tqdm(data_loader)
    for x_batch, y_batch in progbar:
        x_adv = adversary.run_standard_evaluation(x_batch.to(device), y_batch.to(device), bs=x_batch.shape[0])
        x_rec = generate_reconstructions(normalize(x_adv), **study.best_params)
        x_orig = generate_reconstructions(normalize(x_batch.to(device)), **study.best_params)

        y_pred_orig = model(x_orig).cpu().detach().numpy()
        y_pred = model(x_rec).cpu().detach().numpy()

        orig_acc += (y_pred_orig.argmax(axis=1) == y_batch.numpy()).sum()
        adv_rec_acc += (y_pred.argmax(axis=1) == y_batch.numpy()).sum()
        total += x_batch.shape[0]

        progbar.set_postfix({'orig_acc': orig_acc/total, 'adv_rec_acc': adv_rec_acc/total})
