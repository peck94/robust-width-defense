import numpy as np

import argparse

import optuna

import pywt

import torch
import torchvision
import torchvision.transforms as transforms

from art.attacks.evasion import AutoProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier

from utils import generate_reconstructions, normalize, Wrapper

from tqdm import tqdm

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-name', type=str, default='cs-test', help='study name')
    parser.add_argument('-results', type=str, default='sqlite:///results.db', help='results database')
    parser.add_argument('-eps', type=int, default=4, help='perturbation budget')
    parser.add_argument('-data', type=str, default='/scratch/jpeck/imagenet', help='ImageNet path')
    parser.add_argument('-version', type=str, default='standard', help='AutoAttack version')
    parser.add_argument('-bs', type=int, default=16, help='batch size')
    parser.add_argument('-trials', type=int, default=100, help='number of trials')

    args = parser.parse_args()

    # get device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Device: {device}')

    # load model
    weights = torchvision.models.Swin_B_Weights.DEFAULT
    preprocess = weights.transforms(antialias=True)
    raw_model = torchvision.models.swin_b(weights=weights)
    model = Wrapper(preprocess, raw_model).to(device)

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
        method = trial.suggest_categorical('wavelet', pywt.wavelist())
        undersample_rate = trial.suggest_float('undersample_rate', 0.25, 1)
        levels = trial.suggest_int('levels', 1, 10)
        lam = trial.suggest_float('lam', 0, 1)
        lam_decay = trial.suggest_float('lam_decay', 0.9, 1)

        adv_rec_acc = 0
        total = 0
        max_batches = 100
        attack = AutoProjectedGradientDescent(estimator=classifier, eps=args.eps/255, norm=np.inf)
        progbar = tqdm(data_loader, total=max_batches)
        try:
            for step, (x_batch, y_batch) in enumerate(progbar):
                x_adv = torch.from_numpy(attack.generate(x=x_batch.numpy(), y=y_batch.numpy()))
                x_rec = generate_reconstructions(normalize(x_adv), undersample_rate, method, levels, lam, lam_decay)

                y_pred_rec = model(x_rec.to(device)).cpu().detach().numpy()
                adv_rec_acc += (y_pred_rec.argmax(axis=1) == y_batch.numpy()).sum()
                total += x_batch.shape[0]

                progbar.set_postfix({'adv_rec_acc': adv_rec_acc/total})

                trial.report(adv_rec_acc/total, step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                if step >= max_batches - 1:
                    break
        except:
            raise optuna.TrialPruned()
        
        return adv_rec_acc/total
    
    # start the study
    study = optuna.create_study(study_name=args.name, storage=args.result, load_if_exists=True, direction='maximize')
    study.optimize(objective, n_trials=args.trials)
