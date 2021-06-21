from deepscm.experiments import morphomnist_reversed_arrows  # noqa: F401
from deepscm.experiments.morphomnist_reversed_arrows.base_experiment import EXPERIMENT_REGISTRY, MODEL_REGISTRY
# from morphomnist_reversed_arrows.base_experiment import EXPERIMENT_REGISTRY, MODEL_REGISTRY

import torch

import sys
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import pyro
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import pandas as pd
import inspect
from collections import OrderedDict
from functools import partial
import torch

import traceback
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

torch.autograd.set_grad_enabled(False);

from deepscm.datasets.morphomnist import MorphoMNISTLike
from deepscm.submodules.morphomnist.morphomnist import io, morpho, perturb
from deepscm.morphomnist import measure
import multiprocessing

def measure_image(x, threshold=0.5, use_progress_bar=True):
    imgs = x.detach().cpu().numpy()[:, 0]

    with multiprocessing.Pool() as pool:
        measurements = measure.measure_batch(imgs, threshold=threshold, pool=pool)

    def get_intensity(imgs, threshold):

        img_min, img_max = imgs.min(axis=(1, 2), keepdims=True), imgs.max(axis=(1, 2), keepdims=True)
        mask = (imgs >= img_min + (img_max - img_min) * threshold)

        return np.array([np.median(i[m]) for i, m in zip(imgs, mask)])

    return measurements['thickness'].values, get_intensity(imgs, threshold)

def prep_data(batch):
    x = batch['image']
    thickness = batch['thickness'].unsqueeze(-1).float()
    intensity = batch['intensity'].unsqueeze(-1).float()
    x = x.float()
    x = x.unsqueeze(1)
    return {'x': x, 'thickness': thickness, 'intensity': intensity}

def get_counterfactuals_conds(x, intensity, thickness):
    _int_lower = intensity - 75
    _int_lower[_int_lower < 0] = 0
    return {
        'do(i + 75)': {'intensity': intensity + 75},
        'do(i - 75)': {'intensity': _int_lower},
        'do(t + 2)': {'thickness': thickness + 2},
        'do(t - 2)': {'thickness': thickness - 1.5},
    }

import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--version', type=int)
parser.add_argument('--exp', type=str)
parser.add_argument('--data_path', type=str)
parser.add_argument('--log_path', type=str)
args = parser.parse_args()
v = args.version
exp = args.exp
MNIST_DATA_PATH = args.data_path
BASE_LOG_PATH = args.log_path

try: 
    checkpoint_path= f'{BASE_LOG_PATH}/{exp}/version_{v}/'

    base_path = os.path.join(checkpoint_path, 'checkpoints')
    checkpoint_path = os.path.join(base_path, os.listdir(base_path)[0])

    ckpt = torch.load(checkpoint_path, map_location=torch.device('cuda'))
    hparams = ckpt['hyper_parameters']
    
    model_class = MODEL_REGISTRY[hparams['model']]

    model_params = {
        k: v for k, v in hparams.items() if (
            k in inspect.signature(model_class.__init__).parameters
            or k in k in inspect.signature(model_class.__bases__[0].__init__).parameters
            or k in k in inspect.signature(model_class.__bases__[0].__bases__[0].__init__).parameters
        )
    }
    
    new_state_dict = OrderedDict()

    for key, value in ckpt['state_dict'].items():
        new_key = key.replace('pyro_model.', '')
        new_state_dict[new_key] = value
        
    loaded_model = model_class(**model_params)
    loaded_model.load_state_dict(new_state_dict)
    
    for p in loaded_model._buffers.keys():
        if 'norm' in p:
            setattr(loaded_model, p, getattr(loaded_model, p))
            
    loaded_model.eval()
except Exception as e:
    print(e)
    traceback.print_exc()


test_data = MorphoMNISTLike(MNIST_DATA_PATH, train=False, columns=['thickness', 'intensity'])
# thickened_images = io.load_idx(f'{MNIST_DATA_PATH}t10k-images-idx3-ubyte-thick.gz')
# thinned_images = io.load_idx(f'{MNIST_DATA_PATH}t10k-images-idx3-ubyte-thin.gz')

from collections import defaultdict

batch_size = 512
data_len = len(test_data)
batches = data_len // batch_size
model_name = exp  # 'ConditionalReversedVISEM'

recons = []
print('Recons')
for i in range(batches + 1):
    print(i)
    lb = i * batch_size
    if i == batches:
        ub = data_len  # len(test_data)
    else:
        ub = i * batch_size + batch_size
    batch = prep_data(test_data[lb:ub])
    _recons = loaded_model.reconstruct(**batch, num_particles=8)
    # cfs['mae'] += np.abs((recons - batch['x'])).reshape(ub - lb, -1).mean(axis=1).sum()
    recons.append(_recons)

# cfs_dict = f'{exp}_version_{v}_recon'
np.save(f'{exp}_version_{v}_recon', np.concatenate(recons))

print(model_name)

# cfs = defaultdict(float)
conds = [
    'do(i + 75)',
    'do(i - 75)',
    'do(t + 2)',
    'do(t - 2)',
]
for k in conds:
    cfs = []
    for i in range(batches + 1):
        print(i)
        lb = i * batch_size
        if i == batches:
            ub = data_len  # len(test_data)
        else:
            ub = i * batch_size + batch_size
        batch = prep_data(test_data[lb:ub])   
        cf_conds = get_counterfactuals_conds(batch['x'], batch['intensity'], batch['thickness'])
        cfs_ = loaded_model.counterfactual(
            obs=batch, condition=cf_conds[k], num_particles=4)
        cfs.append(cfs_['x'].numpy())
    np.save(f'{exp}_main_version_{v}_{k}', np.concatenate(cfs))

        # hparams['num_sample_particles'])
        # cfs_intensity_2 = loaded_model.counterfactual(
        #     obs=batch, condition=cf_conds['do(i - 75)'], num_particles=hparams['num_sample_particles'])
        # cfs_thickness_1 = loaded_model.counterfactual(
        #     obs=batch, condition=cf_conds['do(t + 2)'], num_particles=hparams['num_sample_particles'])
        # cfs_thickness_2 = loaded_model.counterfactual(
        #     obs=batch, condition=cf_conds['do(t - 2)'], num_particles=hparams['num_sample_particles'])
        
        # cfs_comb = np.vstack([
        #     cfs_intensity_1['x'],
        #     cfs_intensity_2['x'],
        #     cfs_thickness_1['x'],
        #     cfs_thickness_2['x']
        # ])
        # 
        # preds = measure_image(torch.tensor(cfs_comb))
        # 
        # thickness_pred = preds[0].reshape(-1, ub - lb)
        # intensity_pred = preds[1].reshape(-1, ub - lb)
        # 
        # int_mae_1 = np.abs(cf_conds['do(i + 75)']['intensity'].flatten() - np.nan_to_num(intensity_pred[0])).sum()
        # int_mae_2 = np.abs(cf_conds['do(i - 75)']['intensity'].flatten() - np.nan_to_num(intensity_pred[1])).sum()
        # thick_mae_1_measure = np.abs(cf_conds['do(t + 2)']['thickness'].flatten() - np.nan_to_num(thickness_pred[2])).sum()
        # thick_mae_2_measure = np.abs(cf_conds['do(t - 2)']['thickness'].flatten() - np.nan_to_num(thickness_pred[3])).sum()
        # 
        # cfs['do(i + 75)'] += int_mae_1
        # cfs['do(i - 75)'] += int_mae_2
        # cfs['do(t + 2)_measure'] += thick_mae_1_measure
        # cfs['do(t - 2)_measure'] += thick_mae_2_measure
        # 
        # t_1 = cfs_thickness_1['x'].squeeze(1).numpy()
        # t_2 = cfs_thickness_2['x'].squeeze(1).numpy()
        # thick_mae_1 = (thickened_images[lb:ub] - t_1).reshape(ub - lb, -1).sum() / 255
        # thick_mae_2 = (thinned_images[lb:ub] - t_2).reshape(ub - lb, -1).sum() / 255
        # cfs['do(t + 2)'] += thick_mae_1
        # cfs['do(t - 2)'] += thick_mae_2
        # 
        # # print(int_mae_1, int_mae_2, thick_mae_1_measure, thick_mae_2_measure, thick_mae_1, thick_mae_2)

for k in cfs.keys():
    cfs[k] /= data_len

cfs_dict = f'{exp}_version_{v}'
np.save(cfs_dict, cfs)

print(cfs)
