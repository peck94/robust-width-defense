import numpy as np

import argparse

import optuna

import torch
import torchvision
import torchvision.transforms as transforms

from art.attacks.evasion import AutoProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier

from reconstruction import Reconstruction

from tqdm import tqdm

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, default='resnet50', help='model name')
    parser.add_argument('-weights', type=str, default='IMAGENET1K_V2', help='model weights')
    parser.add_argument('-name', type=str, default='cs-test', help='study name')
    parser.add_argument('-results', type=str, default='sqlite:///results.db', help='results database')
    parser.add_argument('-trials', type=int, default=1000, help='number of trials')
    parser.add_argument('-eps', type=int, default=4, help='perturbation budget')
    parser.add_argument('-version', type=str, default='standard', help='AutoAttack version')
    parser.add_argument('-bs', type=int, default=16, help='batch size')
    parser.add_argument('-max-batches', type=int, default=64, help='maximum number of batches')
    parser.add_argument('-data', type=str, default='/scratch/jpeck/imagenet', help='ImageNet path')
    parser.add_argument('-iterations', type=int, default=10, help='AutoPGD iterations')

    args = parser.parse_args()

    # get device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Device: {device}')

    # load model
    model = torch.hub.load('pytorch/vision', args.model, weights=args.weights).to(device)

    classifier = PyTorchClassifier(
        model=model,
        loss=torch.nn.CrossEntropyLoss(),
        input_shape=(3, 224, 224),
        nb_classes=1000,
        clip_values=(0, 1)
    )

    # load data
    imagenet_data = torchvision.datasets.ImageNet(args.data, split='val',
                                              transform=transforms.Compose([
                                                  transforms.Resize(256),
                                                  transforms.CenterCrop(224),
                                                  transforms.ToTensor()]))
    data_loader = torch.utils.data.DataLoader(imagenet_data, batch_size=args.bs, shuffle=True, num_workers=1)

    # define the objective function
    def objective(trial):
        Reconstruction.initialize_trial(trial)

        method = Reconstruction.get_method(trial.params['method'])
        method.initialize_trial(trial)

        reconstructor = Reconstruction(**trial.params, device=device, eps=args.eps/255)

        print(f'Running trial with params: {trial.params}')

        adv_rec_acc = 0
        orig_rec_acc = 0
        total = 0
        attack = AutoProjectedGradientDescent(estimator=classifier,
                                              norm=np.inf,
                                              eps=args.eps/255,
                                              max_iter=10,
                                              nb_random_init=5,
                                              loss_type='difference_logits_ratio')
        progbar = tqdm(data_loader, total=args.max_batches)
        for step, (x_batch, y_batch) in enumerate(progbar):
            x_orig = reconstructor.generate(x_batch.float().to(device)).float()
            x_adv = torch.from_numpy(attack.generate(x=x_batch.numpy(), y=y_batch.numpy())).float().to(device)
            x_rec = reconstructor.generate(x_adv).float()

            if x_orig.shape[1] < 3:
                raise optuna.TrialPruned()

            y_pred_orig = model(x_orig).cpu().detach().numpy()
            orig_rec_acc += (y_pred_orig.argmax(axis=1) == y_batch.numpy()).sum()

            y_pred_rec = model(x_rec).cpu().detach().numpy()
            adv_rec_acc += (y_pred_rec.argmax(axis=1) == y_batch.numpy()).sum()

            total += x_batch.shape[0]

            progbar.set_postfix({'adv_rec_acc': adv_rec_acc/total, 'orig_rec_acc': orig_rec_acc/total})

            if adv_rec_acc/total < .1 or orig_rec_acc/total < .1:
                raise optuna.TrialPruned()

            if step >= args.max_batches - 1:
                break
        
        return adv_rec_acc/total, orig_rec_acc/total
    
    # start the study
    study = optuna.create_study(study_name=args.name, storage=args.results, load_if_exists=True, directions=['maximize', 'maximize'])
    study.optimize(objective, n_trials=args.trials)
