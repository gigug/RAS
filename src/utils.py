"""Utility functions for downloading checkpoints and setting up datasets."""
import os
import zipfile

import torch
from torchvision.models import ResNet50_Weights
from torch.hub import load_state_dict_from_url

from openood.networks.resnet18_32x32 import ResNet18_32x32
from openood.networks.resnet18_224x224 import ResNet18_224x224
from openood.networks.resnet50 import ResNet50


# Google Drive IDs for v1.5 checkpoints
CKPT_GDRIVE_IDS = {
    'cifar10_res18_v1.5': '1byGeYxM_PlLjT72wZsMQvP6popJeWBgt',
    'cifar100_res18_v1.5': '1s-1oNrRtmA0pGefxXJOUVRYpaoAML0C-',
    'imagenet200_res18_v1.5': '1ddVmwc8zmzSjdLUO84EuV4Gz1c7vhIAs',
}

# Checkpoint directory names after unzipping
CKPT_DIRS = {
    'cifar10': 'cifar10_resnet18_32x32_base_e100_lr0.1_default',
    'cifar100': 'cifar100_resnet18_32x32_base_e100_lr0.1_default',
    'imagenet200': 'imagenet200_resnet18_224x224_base_e90_lr0.1_default',
}


def download_checkpoint(ckpt_key, store_path):
    """Download and extract a checkpoint from Google Drive if not present."""
    import gdown

    os.makedirs(store_path, exist_ok=True)

    # Check if already downloaded
    expected_contents = os.listdir(store_path) if os.path.exists(store_path) else []
    if any(item.startswith(ckpt_key.replace('_v1.5', '').replace('_res18', '_resnet18_32x32').replace('_res50', '_resnet50'))
           for item in expected_contents):
        print(f'Checkpoint {ckpt_key} already present, skipping download.')
        return

    print(f'Downloading checkpoint: {ckpt_key}...')
    dl_path = store_path if store_path.endswith('/') else store_path + '/'
    gdown.download(id=CKPT_GDRIVE_IDS[ckpt_key], output=dl_path)

    zip_path = os.path.join(store_path, ckpt_key + '.zip')
    if os.path.exists(zip_path):
        print(f'Extracting {zip_path}...')
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(store_path)
        os.remove(zip_path)
    print(f'Checkpoint {ckpt_key} ready.')


def setup_checkpoints(id_data_list, ckpt_root):
    """Download all required checkpoints."""
    needed = set()
    for id_data in id_data_list:
        if id_data == 'cifar10':
            needed.add('cifar10_res18_v1.5')
        elif id_data == 'cifar100':
            needed.add('cifar100_res18_v1.5')
        elif id_data == 'imagenet200':
            needed.add('imagenet200_res18_v1.5')
        # imagenet uses torchvision pretrained weights, no download needed

    for ckpt_key in needed:
        download_checkpoint(ckpt_key, ckpt_root)


def get_experiments(id_data_list, ckpt_root):
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
                        ckpt_root, CKPT_DIRS[id_data],
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


def load_backbone(exp, device):
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

    backbone.to(device)
    backbone.eval()
    return backbone, preprocessor
