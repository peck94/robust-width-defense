# Robust width adversarial defense

This repository contains the code for our paper, *Robust width adversarial defense*.

## Setup

Most of the dependencies are described in the `environment.yml` file. You can setup this environment using Anaconda:

```console
conda env create -f environment.yml
conda activate sparse-ds
```

There is an additional dependency in the form of a custom library we created specifically for this project, called `pytorch_shearlets`. This is merely a (partial) PyTorch port of [pyShearLab](https://github.com/stefanloock/pyshearlab). Once the conda environment has been set up and activated, you can install the provided `pytorch_shearlets` wheel using pip:

```console
pip install pytorch_shearlets-0.0.1-py3-none-any.whl
```

This should take care of all dependencies.

## Hyperparameter optimization

The hyperparameter optimization is carried out by the `main.py` Python script. For our experiments, we saved the results of the Optuna trials to a local SQLite database called `fourier.db`. The results can be viewed using the Optuna dashboard:

```console
optuna-dashboard sqlite:///fourier.db
```

## Reproducing experiments

You can reproduce our experiments by running the appropriate script:

```console
bash experiments.sh
```

This will run all of the experiments one by one. Individual adversarial attacks can be carried out using the `attack.py` Python script.
