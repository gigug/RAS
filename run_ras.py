"""
Test script for RAS (Ranked Activation Shift) on the OpenOOD benchmark.

Downloads checkpoints and datasets automatically, then evaluates RAS on:
- CIFAR-10:     ResNet18_32x32,  seeds 0,1,2
- CIFAR-100:    ResNet18_32x32,  seeds 0,1,2
- ImageNet-200: ResNet18_224x224, seeds 0,1,2
- ImageNet-1K:  ResNet50 (torchvision pretrained)

Datasets are downloaded automatically by OpenOOD's Evaluator.
ImageNet-1K requires manual setup (see README.md for instructions).

Usage:
    python run_ras.py                          # run all experiments
    python run_ras.py --id-data cifar10        # run only CIFAR-10
    python run_ras.py --id-data imagenet       # run only ImageNet-1K
"""
import argparse
import os
import sys
import pickle
import traceback

import numpy as np
import pandas as pd
import torch

from torchvision.models import ResNet50_Weights
from torch.hub import load_state_dict_from_url

from openood.evaluation_api import Evaluator
from openood.evaluation_api.datasets import DATA_INFO
from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.networks.resnet18_224x224 import ResNet18_224x224
from openood.networks.resnet50 import ResNet50

# Removing covariate-shifted ID datasets from OpenOOD config to avoid
# downloading/loading corruption data (e.g. CIFAR-10-C, ImageNet-C).
for _dataset_cfg in DATA_INFO.values():
    _dataset_cfg['csid'] = {'datasets': []}

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from src.ras_postprocessor import RASPostprocessor
from src.ras_net import RASNet
from src.utils import setup_checkpoints, CKPT_DIRS

# Device setup
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cpu':
    print('WARNING: CUDA not available, running on CPU. This will be slow.')

# Paths
DATA_ROOT = os.path.join(ROOT_DIR, 'data')
CONFIG_ROOT = os.path.join(ROOT_DIR, 'configs')
CKPT_ROOT = os.path.join(ROOT_DIR, 'results')
RESULTS_DIR = os.path.join(ROOT_DIR, 'run_results')


def get_experiments(id_data_list):
    """Build the experiment list based on requested datasets."""
    experiments = []

    for id_data in id_data_list:
        if id_data in ('cifar10', 'cifar100', 'imagenet200'):
            model_class = {
                'cifar10': ResNet18_32x32,
                'cifar100': ResNet18_32x32,
                'imagenet200': ResNet18_224x224,
            }[id_data]
            num_classes = {'cifar10': 10, 'cifar100': 100, 'imagenet200': 200}[id_data]

            for seed in [0, 1, 2]:
                experiments.append({
                    'id_name': id_data,
                    'model_class': model_class,
                    'num_classes': num_classes,
                    'ckpt': os.path.join(
                        CKPT_ROOT, CKPT_DIRS[id_data],
                        f's{seed}', 'best.ckpt'),
                    'seed': seed,
                    'preprocessor': None,
                })

        elif id_data == 'imagenet':
            experiments.append({
                'id_name': 'imagenet',
                'model_class': ResNet50,
                'num_classes': 1000,
                'ckpt': '__torchvision__',
                'seed': 0,
                'preprocessor': '__torchvision__',
            })

    return experiments


def load_backbone(exp):
    """Load backbone model with checkpoint."""
    backbone = exp['model_class'](num_classes=exp['num_classes'])

    if exp['ckpt'] == '__torchvision__':
        weights = ResNet50_Weights.IMAGENET1K_V1
        backbone.load_state_dict(load_state_dict_from_url(weights.url))
        preprocessor = weights.transforms()
    else:
        backbone.load_state_dict(
            torch.load(exp['ckpt'], map_location='cpu'))
        preprocessor = None

    backbone.to(DEVICE)
    backbone.eval()
    return backbone, preprocessor


def run_experiment(exp, ras_postprocessor, batch_size):
    """Run a single RAS evaluation experiment."""
    id_name = exp['id_name']
    seed = exp['seed']
    print(f'\n{"="*70}')
    print(f'[RAS] {id_name} | seed={seed}')
    print(f'{"="*70}')

    backbone, preprocessor = load_backbone(exp)
    net = RASNet(backbone)

    evaluator = Evaluator(
        net,
        id_name=id_name,
        data_root=DATA_ROOT,
        config_root=CONFIG_ROOT,
        preprocessor=preprocessor,
        postprocessor=ras_postprocessor,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
    )

    metrics = evaluator.eval_ood()
    print(metrics)

    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_file = os.path.join(
        RESULTS_DIR, f'{id_name}_ras_seed{seed}.pickle')
    with open(result_file, 'wb') as f:
        pickle.dump(metrics, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Saved to {result_file}')

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate RAS on OpenOOD')
    parser.add_argument('--id-data', type=str, default='all',
                        choices=['all', 'cifar10', 'cifar100',
                                 'imagenet200', 'imagenet'])
    parser.add_argument('--batch-size', type=int, default=200)
    args = parser.parse_args()

    if args.id_data == 'all':
        id_data_list = ['cifar10', 'cifar100', 'imagenet200', 'imagenet']
    else:
        id_data_list = [args.id_data]

    # Download checkpoints (datasets are downloaded automatically by OpenOOD)
    print('Setting up checkpoints...')
    setup_checkpoints(id_data_list, CKPT_ROOT)

    # Build experiment list
    experiments = get_experiments(id_data_list)

    # Instantiate RAS postprocessor
    ras_postprocessor = RASPostprocessor(None)

    # Run experiments
    all_results = {}
    for exp in experiments:
        key = f"{exp['id_name']}_seed{exp['seed']}"
        try:
            all_results[key] = run_experiment(exp, ras_postprocessor, args.batch_size)
        except Exception as e:
            print(f'!!! Error: {key}: {e}')
            traceback.print_exc()
            all_results[key] = None

    # Summary with mean +/- std per dataset
    print(f'\n{"="*70}')
    print('SUMMARY')
    print(f'{"="*70}')

    for id_data in id_data_list:
        dataset_results = [
            v for k, v in all_results.items()
            if k.startswith(id_data) and v is not None
        ]
        if not dataset_results:
            print(f'\n{id_data}: ALL FAILED')
            continue

        if len(dataset_results) > 1:
            stacked = np.stack([m.to_numpy() for m in dataset_results], axis=0)
            mean = np.mean(stacked, axis=0)
            std = np.std(stacked, axis=0)
            summary = []
            for i in range(len(mean)):
                row = []
                for j in range(mean.shape[1]):
                    row.append(u'{:.2f} \u00B1 {:.2f}'.format(
                        mean[i, j], std[i, j]))
                summary.append(row)
            df = pd.DataFrame(summary,
                              index=dataset_results[0].index,
                              columns=dataset_results[0].columns)
            print(f'\n{id_data} (mean \u00B1 std over {len(dataset_results)} seeds):')
            print(df)
        else:
            print(f'\n{id_data}:')
            print(dataset_results[0])


if __name__ == '__main__':
    main()
