ROOT_PATH = '../../../../'
UKBB_DATA_PATH = ROOT_PATH + 'assets/data/ukbb_meshes/'
BASE_LOG_PATH = ROOT_PATH + 'medical_mesh_experiments/SVIExperiment'

import sys
import os

sys.path.append(ROOT_PATH)
import torch
gpu = True
if gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device('cuda')
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    device = torch.device('cpu')
    
from tqdm import tqdm, trange
import traceback
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import inspect
from collections import OrderedDict
from itertools import product
from functools import partial

import numpy as np
import pandas as pd
import pyro
torch.autograd.set_grad_enabled(False);
import seaborn as sns
import pyvista as pv
pv.set_plot_theme("document")

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300

from PIL import Image

from deepscm.experiments.medical_meshes import ukbb  # noqa: F401
from deepscm.experiments.medical_meshes.base_experiment import EXPERIMENT_REGISTRY, MODEL_REGISTRY

data = [
    (1026586, {
        'sex': 1,
        'age': 42,
        'brain_volume': 1256030,
        'stem_volume': 28976,
        'hipp_volume': 4127,
        'stem_template': pv.read('ref_meshes/T1_first-BrStem_first_1026586.vtk'),
        'hipp_template': pv.read('ref_meshes/T1_first-R_Hipp_first_1026586.vtk'),
        'stem_x': np.array(pv.read('ref_meshes/T1_first-BrStem_first_1026586.vtk').points),
        'hipp_x': np.array(pv.read('ref_meshes/T1_first-R_Hipp_first_1026586.vtk').points),
    }),
    (2654886, {
        'sex': 0,
        'age': 40,
        'brain_volume': 1009720,
        'stem_volume': 17963,
        'hipp_volume': 2990,
        'stem_template': pv.read('ref_meshes/T1_first-BrStem_first_2654886.vtk'),
        'hipp_template': pv.read('ref_meshes/T1_first-R_Hipp_first_2654886.vtk'),
        'stem_x': np.array(pv.read('ref_meshes/T1_first-BrStem_first_2654886.vtk').points),
        'hipp_x': np.array(pv.read('ref_meshes/T1_first-R_Hipp_first_2654886.vtk').points),
    }),
    (5163634, {
        'sex': 0,
        'age': 57,
        'brain_volume': 1245880,
        'stem_volume': 24459,
        'hipp_volume': 3717,
        'stem_template': pv.read('ref_meshes/T1_first-BrStem_first_5163634.vtk'),
        'hipp_template': pv.read('ref_meshes/T1_first-R_Hipp_first_5163634.vtk'),
        'stem_x': np.array(pv.read('ref_meshes/T1_first-BrStem_first_5163634.vtk').points),
        'hipp_x': np.array(pv.read('ref_meshes/T1_first-R_Hipp_first_5163634.vtk').points),
    }),
    (5862049, {
        'sex': 1,
        'age': 66,
        'brain_volume': 1162210,
        'stem_volume': 22799,
        'hipp_volume': 3941,
        'stem_template': pv.read('ref_meshes/T1_first-BrStem_first_5862049.vtk'),
        'hipp_template': pv.read('ref_meshes/T1_first-R_Hipp_first_5862049.vtk'),
        'stem_x': np.array(pv.read('ref_meshes/T1_first-BrStem_first_5862049.vtk').points),
        'hipp_x': np.array(pv.read('ref_meshes/T1_first-BrStem_first_5862049.vtk').points),
    }),
]

var_name = {'structure_volume': 'v', 'brain_volume': 'b', 'sex': 's', 'age': 'a'}
value_fmt = {
    'structure_volume': lambda s: rf'{float(s)/1000:.4g}\,\mathrm{{ml}}',
    'brain_volume': lambda s: rf'{float(s)/1000:.4g}\,\mathrm{{ml}}',
    'age': lambda s: rf'{int(s):d}\,\mathrm{{y}}',
    'sex': lambda s: '{}'.format(['\mathrm{female}', '\mathrm{male}'][int(s)])
}


def fmt_intervention(intervention):
    if isinstance(intervention, str):
        var, value = intervention[3:-1].split('=')
        return f"$do({var_name[var]}={value_fmt[var](value)})$"
    else:
        all_interventions = ',\n'.join([f'${var_name[k]}={value_fmt[k](v)}$' for k, v in intervention.items()])
        return f"do({all_interventions})"

def prep_data(input_dict, stem=True):
    if stem:
        x = torch.tensor(input_dict['stem_x']).unsqueeze(0).float().to(device)
        structure_volume = torch.tensor([[input_dict['stem_volume']]]).float().to(device)
        template = input_dict['stem_template']
    else:
        x = torch.tensor(input_dict['hipp_x']).unsqueeze(0).float().to(device)
        structure_volume = torch.tensor([[input_dict['hipp_volume']]]).float().to(device)
        template = input_dict['hipp_template']
    age = torch.tensor([[input_dict['age']]]).float().to(device)
    sex = torch.tensor([[input_dict['sex']]]).float().to(device)
    brain_volume = torch.tensor([[input_dict['brain_volume']]]).float().to(device)

    data = {
        'x': x,
        'age': age,
        'sex': sex,
        'structure_volume': structure_volume,
        'brain_volume': brain_volume
    }
    return template, data

def load_model(base_log_path=BASE_LOG_PATH, exp='ConditionalVISEM', version='version_87'):
    checkpoint_path = f'{base_log_path}/{exp}/{version}/'
    base_path = os.path.join(checkpoint_path, 'checkpoints')
    checkpoint_path = os.path.join(base_path, os.listdir(base_path)[0])

    ckpt = torch.load(checkpoint_path, map_location=device)
    hparams = ckpt['hyper_parameters']
    # hparams['gpu'] = 1 if gpu else 0
    
    model_class = MODEL_REGISTRY[hparams['model']]

    model_params = {
        k: v for k, v in hparams.items() if (
            k in inspect.signature(model_class.__init__).parameters
            or k in k in inspect.signature(model_class.__bases__[0].__init__).parameters
            or k in k in inspect.signature(model_class.__bases__[0].__bases__[0].__init__).parameters
        )
    }
    # model_params['gpu'] = 1 if gpu else 0

    new_state_dict = OrderedDict()

    for key, value in ckpt['state_dict'].items():
        new_key = key.replace('pyro_model.', '')
        new_state_dict[new_key] = value

    loaded_model = model_class(**model_params)
    loaded_model.load_state_dict(new_state_dict)
    loaded_model.to(device)

    for p in loaded_model._buffers.keys():
        if 'norm' in p:
            setattr(loaded_model, p, getattr(loaded_model, p))

    loaded_model.eval()

    def sample_pgm(num_samples, model):
        with pyro.plate('observations', num_samples):
            return model.pgm_model()

    model = partial(sample_pgm, model=loaded_model)
    
    return loaded_model, model


stem_loaded_model, _ = load_model(version='version_116')
hipp_loaded_model, _ = load_model(version='version_120')

def interactive_plot(stem=True):
    rot = (0, 0, 260)
    
    loaded_model = stem_loaded_model if stem else hipp_loaded_model
    
    def template_with_points(template, points):
        _template = template.copy()
        _template.points = points
        return _template

    def rot_template(template, rot):
        template.rotate_x(rot[0])
        template.rotate_y(rot[1])
        template.rotate_z(rot[2])

    def plot_mesh(plotter, x, y, title, mesh_arr, diff, template):
        plotter.subplot(x, y)
        plotter.add_title(title, font_size=8)
        mesh = template_with_points(template, mesh_arr)
        rot_template(mesh, rot)
        mesh['distance'] = diff
        plotter.add_mesh(
            mesh,
            scalars='distance',
            cmap='seismic',
            clim=[-5, 5],
            smooth_shading=True,
            flip_scalars=True,
            show_scalar_bar=False
        )
    
    def plot_intervention(intervention, idx, num_samples=1):
        fig, ax = plt.subplots(1, 4, figsize=(10, 2.5), gridspec_kw=dict(wspace=0, hspace=0))
        lim = 0

        record = data[idx]
        template, orig_data = prep_data(record[1], stem)
        x_test = orig_data['x']

        pyro.clear_param_store()
        cond = {k: torch.tensor([[v]]).to(device) for k, v in intervention.items()}
        counterfactual = loaded_model.counterfactual(orig_data, cond, num_samples)

        x = counterfactual['x']

        diff = (x - x_test).squeeze().mean(axis=1).cpu().numpy()
        x_test = x_test.squeeze(0).cpu().numpy()
        x = x.squeeze(0).cpu().numpy()
        
        plotter = pv.Plotter(
            shape=(1, 1),
            window_size=(256, 256),
            border=False,
            lighting='light_kit',
            off_screen=True,
            notebook=False,
        )
        plot_mesh(plotter, 0, 0, '', x_test, np.zeros_like(diff), template)
        plotter.show(screenshot='orig_interactive.png')
        
        plotter = pv.Plotter(
            shape=(1, 1),
            window_size=(256, 256),
            border=False,
            lighting='light_kit',
            off_screen=True,
            notebook=False,
        )
        plot_mesh(plotter, 0, 0, '', x, np.zeros_like(diff), template)
        plotter.show(screenshot='do_interactive.png')
        
        plotter = pv.Plotter(
            shape=(1, 1),
            window_size=(256, 256),
            border=False,
            lighting='light_kit',
            off_screen=True,
            notebook=False,
        )
        plot_mesh(plotter, 0, 0, '', x, diff, template)
        plotter.show(screenshot='do_difference.png')

        ax[1].set_title('Original')
        ax[1].imshow(Image.open('orig_interactive.png', 'r'))

        ax[2].set_title(fmt_intervention(intervention))
        ax[2].imshow(Image.open('do_interactive.png', 'r'))

        ax[3].set_title('Difference')
        ax[3].imshow(Image.open('do_difference.png', 'r'))

        for axi in ax:
            axi.axis('off')
            axi.xaxis.set_major_locator(plt.NullLocator())
            axi.yaxis.set_major_locator(plt.NullLocator())

        att_str = '$s={sex}$\n$a={age}$\n$b={brain_volume}$\n$v={structure_volume}$'.format(
            **{att: value_fmt[att](orig_data[att].item())
            for att in ('sex', 'age', 'brain_volume', 'structure_volume')}
        )

        ax[0].text(0.5, 0.5, att_str, horizontalalignment='center',
            verticalalignment='center', transform=ax[0].transAxes,
            fontsize=mpl.rcParams['axes.titlesize']
        )

        plt.show()
    
    from ipywidgets import interactive, IntSlider, FloatSlider, HBox, VBox, Checkbox, Dropdown

    def plot(image, age, sex, brain_volume, structure_volume, do_age, do_sex, do_brain_volume, do_structure_volume):
        intervention = {}
        if do_age:
            intervention['age'] = age
        if do_sex:
            intervention['sex'] = sex
        if do_brain_volume:
            intervention['brain_volume'] = brain_volume * 1000.
        if do_structure_volume:
            intervention['structure_volume'] = structure_volume * 1000.

        plot_intervention(intervention, image)

    w = interactive(
        plot,
        image=IntSlider(min=0, max=3, description='Image #'),
        age=FloatSlider(min=30., max=90., step=1., continuous_update=False, description='Age'),
        do_age=Checkbox(description='do(age)'),
        sex=Dropdown(options=[('female', 0.), ('male', 1.)], description='Sex'),
        do_sex=Checkbox(description='do(sex)'),
        brain_volume=FloatSlider(min=800., max=1600., step=10., continuous_update=False, description='Brain Volume (ml):', style={'description_width': 'initial'}),
        do_brain_volume=Checkbox(description='do(brain_volume)'),
        structure_volume=FloatSlider(min=11., max=110., step=1., continuous_update=False, description='Structure Volume (ml):', style={'description_width': 'initial'}),
        do_structure_volume=Checkbox(description='do(structure_volume)'),
    )

    ui = VBox([w.children[0], VBox([HBox([w.children[i + 1], w.children[i + 5]]) for i in range(4)]), w.children[-1]])

    display(ui)

    w.update()