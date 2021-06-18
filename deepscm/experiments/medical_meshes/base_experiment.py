import pyro

from pyro.nn import PyroModule, pyro_method

from pyro.distributions import TransformedDistribution
from pyro.infer.reparam.transform import TransformReparam
from torch.distributions import Independent

# from deepscm.datasets.medical.ukbb import UKBBDataset
from pyro.distributions.transforms import ComposeTransform, SigmoidTransform, AffineTransform

import torchvision.utils
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import os
from functools import partial

from coma.utils import transforms
from coma.datasets.ukbb_meshdata import UKBBMeshDataset, VerticesDataLoader


EXPERIMENT_REGISTRY = {}
MODEL_REGISTRY = {}


class BaseSEM(PyroModule):
    def __init__(self):  # , preprocessing: str = 'realnvp', downsample: int = -1):
        super().__init__()

        # self.downsample = downsample
        # self.preprocessing = preprocessing

    # def _get_preprocess_transforms(self):
    #     alpha = 0.05
    #     num_bits = 8

    #     if self.preprocessing == 'glow':
    #         # Map to [-0.5,0.5]
    #         a1 = AffineTransform(-0.5, (1. / 2 ** num_bits))
    #         preprocess_transform = ComposeTransform([a1])
    #     elif self.preprocessing == 'realnvp':
    #         # Map to [0,1]
    #         a1 = AffineTransform(0., (1. / 2 ** num_bits))

    #         # Map into unconstrained space as done in RealNVP
    #         a2 = AffineTransform(alpha, (1 - alpha))

    #         s = SigmoidTransform()

    #         preprocess_transform = ComposeTransform([a1, a2, s.inv])

    #     return preprocess_transform

    @pyro_method
    def pgm_model(self):
        raise NotImplementedError()

    @pyro_method
    def model(self):
        raise NotImplementedError()

    @pyro_method
    def pgm_scm(self, *args, **kwargs):
        def config(msg):
            if isinstance(msg['fn'], TransformedDistribution):
                return TransformReparam()
            else:
                return None

        return pyro.poutine.reparam(self.pgm_model, config=config)(*args, **kwargs)

    @pyro_method
    def scm(self, *args, **kwargs):
        def config(msg):
            if isinstance(msg['fn'], TransformedDistribution):
                return TransformReparam()
            else:
                return None

        return pyro.poutine.reparam(self.model, config=config)(*args, **kwargs)

    @pyro_method
    def sample(self, n_samples=1):
        with pyro.plate('observations', n_samples):
            samples = self.model()

        return (*samples,)

    @pyro_method
    def sample_scm(self, n_samples=1):
        with pyro.plate('observations', n_samples):
            samples = self.scm()

        return (*samples,)

    @pyro_method
    def infer_e_x(self, *args, **kwargs):
        raise NotImplementedError()

    @pyro_method
    def infer_exogeneous(self, **obs):
        # assuming that we use transformed distributions for everything:
        cond_sample = pyro.condition(self.sample, data=obs)
        cond_trace = pyro.poutine.trace(cond_sample).get_trace(obs['x'].shape[0])

        output = {}
        for name, node in cond_trace.nodes.items():
            if 'fn' not in node.keys():
                continue

            fn = node['fn']
            if isinstance(fn, Independent):
                fn = fn.base_dist
            if isinstance(fn, TransformedDistribution):
                output[name + '_base'] = ComposeTransform(fn.transforms).inv(node['value'])

        return output

    @pyro_method
    def infer(self, **obs):
        raise NotImplementedError()

    @pyro_method
    def counterfactual(self, obs, condition=None):
        raise NotImplementedError()

    @classmethod
    def add_arguments(cls, parser):
        # parser.add_argument('--preprocessing', default='realnvp', type=str, help="type of preprocessing (default: %(default)s)", choices=['realnvp', 'glow'])
        # parser.add_argument('--downsample', default=-1, type=int, help="downsampling factor (default: %(default)s)")
        return parser


class BaseCovariateExperiment(pl.LightningModule):
    def __init__(self, hparams, pyro_model: BaseSEM):
        super().__init__()

        self.pyro_model = pyro_model

        hparams.experiment = self.__class__.__name__
        hparams.model = pyro_model.__class__.__name__
        self.hparams = hparams
        self.train_batch_size = hparams.train_batch_size
        self.test_batch_size = hparams.test_batch_size

        if hasattr(hparams, 'num_sample_particles'):
            self.pyro_model._gen_counterfactual = partial(
                self.pyro_model.counterfactual,
                num_particles=self.hparams.num_sample_particles
            )
        else:
            self.pyro_model._gen_counterfactual = self.pyro_model.counterfactual
        
        self.seed = 42

        # if hparams.validate:
        import random

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.autograd.set_detect_anomaly(self.hparams.validate)
        pyro.enable_validation()

    def prepare_data(self):
        # Preprocessor
        preprocessor = transforms.get_transforms()
        
        # Load Dataset
        mesh_path = self.hparams.data_dir
        cache_path = self.hparams.cache_dir 
        split = self.hparams.train_test_split
        substructures = [self.hparams.brain_substructure]

        # TODO: Remove hardcoding for UDIs in UKBB CSV
        substructure_to_udi = {
            'BrStem': '25025-2.0', 
            'L_Thal': '25011-2.0',
            'L_Caud': '25013-2.0',
            'L_Puta': '25015-2.0',
            'L_Pall': '25017-2.0',
            'L_Hipp': '25019-2.0',
            'L_Amyg': '25021-2.0',
            'L_Accu': '25023-2.0',
            'R_Thal': '25012-2.0',
            'R_Caud': '25014-2.0',
            'R_Puta': '25016-2.0',
            'R_Pall': '25018-2.0',
            'R_Hipp': '25020-2.0',
            'R_Amyg': '25022-2.0',
            'R_Accu': '25024-2.0',
        }

        feature_name_map = {
            '31-0.0': 'sex',
            '21003-0.0': 'age',
            substructure_to_udi[self.hparams.brain_substructure]: 'structure_volume',  # Brain Stem
            '25010-2.0': 'brain_volume',  # Unnormalised brain volume from UKBB
        }

        csv_path = self.hparams.csv_path
        metadata_df = pd.read_csv(csv_path)

        reload_mesh_path = self.hparams.reload_mesh_path

        total_train_dataset = UKBBMeshDataset(
            mesh_path,
            substructures=substructures,
            split=split,
            train=True,
            transform=preprocessor,
            reload_path=reload_mesh_path,
            features_df=metadata_df,
            feature_name_map=feature_name_map,
            cache_path=cache_path,
        )
        test_dataset = UKBBMeshDataset(
            mesh_path,
            substructures=substructures,
            split=split,
            train=False,
            transform=preprocessor,
            reload_path=reload_mesh_path,
            features_df=metadata_df,
            feature_name_map=feature_name_map,
            cache_path=cache_path,
        )
        
        val_split = self.hparams.train_val_split
        total_train_length = len(total_train_dataset)
        val_length = int(val_split * total_train_length)
        train_length = total_train_length - val_length
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            total_train_dataset,
            lengths=[train_length, val_length],
            generator=torch.Generator().manual_seed(self.seed),
        )

        self.ukbb_train = train_dataset
        self.ukbb_val = val_dataset
        self.ukbb_test = test_dataset

        self.torch_device = self.trainer.root_gpu if self.trainer.on_gpu else self.trainer.root_device

        # Ranges for plotting
        train_metrics = pd.concat([
            self.ukbb_train.dataset.get_features_by_index(i)
            for i in self.ukbb_train.indices
        ])
        age_tensor = torch.tensor(train_metrics.age.values).float()
        structure_volume_tensor = torch.tensor(train_metrics.structure_volume.values).float()
        brain_volume_tensor = torch.tensor(train_metrics.brain_volume.values).float()

        brain_volumes = 800000. + 300000 * torch.arange(3, dtype=torch.float, device=self.torch_device)
        self.brain_volume_range = brain_volumes.repeat(3).unsqueeze(1)

        min_vol = structure_volume_tensor.min()
        max_vol = structure_volume_tensor.max()
        intervals = (max_vol - min_vol) / 2
        structure_volumes = min_vol + intervals * torch.arange(3, dtype=torch.float, device=self.torch_device)
        self.structure_volume_range = structure_volumes.repeat_interleave(3).unsqueeze(1)

        self.z_range = torch.randn([1, self.hparams.latent_dim], device=self.torch_device, dtype=torch.float).repeat((9, 1))

        # Age and volume metrics
        self.pyro_model.age_flow_lognorm_loc = (
            age_tensor.log().mean().to(self.torch_device).float()
        )
        self.pyro_model.age_flow_lognorm_scale = (
            age_tensor.log().std().to(self.torch_device).float()
        )

        self.pyro_model.structure_volume_flow_lognorm_loc = (
            structure_volume_tensor.log().mean().to(self.torch_device).float()
        )
        self.pyro_model.structure_volume_flow_lognorm_scale = (
            structure_volume_tensor.log().std().to(self.torch_device).float()
        )

        self.pyro_model.brain_volume_flow_lognorm_loc = (
            brain_volume_tensor.log().mean().to(self.torch_device).float()
        )
        self.pyro_model.brain_volume_flow_lognorm_scale = (
            brain_volume_tensor.log().std().to(self.torch_device).float()
        )

        print(f'set structure_volume_flow_lognorm {self.pyro_model.structure_volume_flow_lognorm.loc} +/- {self.pyro_model.structure_volume_flow_lognorm.scale}')  # noqa: E501
        print(f'set age_flow_lognorm {self.pyro_model.age_flow_lognorm.loc} +/- {self.pyro_model.age_flow_lognorm.scale}')
        print(f'set brain_volume_flow_lognorm {self.pyro_model.brain_volume_flow_lognorm.loc} +/- {self.pyro_model.brain_volume_flow_lognorm.scale}')

    def configure_optimizers(self):
        pass

    def train_dataloader(self):
        return VerticesDataLoader(
            self.ukbb_train,
            batch_size=self.train_batch_size,
            shuffle=False
        )

    def val_dataloader(self):
        self.val_loader = VerticesDataLoader(
            self.ukbb_val,
            batch_size=self.test_batch_size,
            shuffle=False
        )
        return self.val_loader

    def test_dataloader(self):
        self.test_loader = VerticesDataLoader(
            self.ukbb_test,
            batch_size=self.test_batch_size,
            shuffle=False
        )
        return self.test_loader

    def forward(self, *args, **kwargs):
        pass

    def prep_batch(self, batch):
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        raise NotImplementedError()

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError()

    def validation_epoch_end(self, outputs):
        outputs = self.assemble_epoch_end_outputs(outputs)

        metrics = {('val/' + k): v for k, v in outputs.items()}

        # if self.current_epoch % self.hparams.sample_mesh_interval == 0:
        self.sample_images()

        self.log_dict(metrics)

    def test_epoch_end(self, outputs):
        print('Assembling outputs')
        outputs = self.assemble_epoch_end_outputs(outputs)

        samples = outputs.pop('samples')

        sample_trace = pyro.poutine.trace(self.pyro_model.sample).get_trace(self.hparams.test_batch_size)
        samples['unconditional_samples'] = {
            'x': sample_trace.nodes['x']['value'].cpu(),
            'brain_volume': sample_trace.nodes['brain_volume']['value'].cpu(),
            'structure_volume': sample_trace.nodes['structure_volume']['value'].cpu(),
            'age': sample_trace.nodes['age']['value'].cpu(),
            'sex': sample_trace.nodes['sex']['value'].cpu()
        }

        cond_data = {
            'brain_volume': self.brain_volume_range,
            'structure_volume': self.structure_volume_range,
            'z': self.z_range
        }
        cond_data = {
            'brain_volume': self.brain_volume_range.repeat(self.hparams.test_batch_size, 1),
            'structure_volume': self.structure_volume_range.repeat(self.hparams.test_batch_size, 1),
            'z': torch.randn([self.hparams.test_batch_size, self.hparams.latent_dim], device=self.torch_device, dtype=torch.float).repeat_interleave(9, 0)
        }
        sample_condition = pyro.condition(self.pyro_model.sample, data=cond_data)
        sample_trace = pyro.poutine.trace(sample_condition).get_trace(9 * self.hparams.test_batch_size)
        samples['conditional_samples'] = {
            'x': sample_trace.nodes['x']['value'].cpu(),
            'brain_volume': sample_trace.nodes['brain_volume']['value'].cpu(),
            'structure_volume': sample_trace.nodes['structure_volume']['value'].cpu(),
            'age': sample_trace.nodes['age']['value'].cpu(),
            'sex': sample_trace.nodes['sex']['value'].cpu()
        }

        print(f'Got samples: {tuple(samples.keys())}')

        metrics = {('test/' + k): v for k, v in outputs.items()}

        for k, v in samples.items():
            p = os.path.join(self.trainer.logger.experiment.log_dir, f'{k}.pt')
            print(f'Saving samples for {k} to {p}')
            torch.save(v, p)

        p = os.path.join(self.trainer.logger.experiment.log_dir, 'metrics.pt')
        torch.save(metrics, p)

        self.log_dict(metrics)

    def assemble_epoch_end_outputs(self, outputs):
        num_items = len(outputs)

        def handle_row(batch, assembled=None):
            if assembled is None:
                assembled = {}

            for k, v in batch.items():
                if k not in assembled.keys():
                    if isinstance(v, dict):
                        assembled[k] = handle_row(v)
                    elif isinstance(v, float):
                        assembled[k] = v
                    elif np.prod(v.shape) == 1:
                        assembled[k] = v.cpu()
                    else:
                        assembled[k] = v.cpu()
                else:
                    if isinstance(v, dict):
                        assembled[k] = handle_row(v, assembled[k])
                    elif isinstance(v, float):
                        assembled[k] += v
                    elif np.prod(v.shape) == 1:
                        assembled[k] += v.cpu()
                    else:
                        assembled[k] = torch.cat([assembled[k], v.cpu()], 0)

            return assembled

        assembled = {}
        for _, batch in enumerate(outputs):
            assembled = handle_row(batch, assembled)

        for k, v in assembled.items():
            if (hasattr(v, 'shape') and np.prod(v.shape) == 1) or isinstance(v, float):
                assembled[k] /= num_items

        return assembled

    def get_counterfactual_conditions(self, batch):
        counterfactuals = {
            'do(brain_volume=800000)': {'brain_volume': torch.ones_like(batch['brain_volume']) * 800000},
            'do(brain_volume=1200000)': {'brain_volume': torch.ones_like(batch['brain_volume']) * 1200000},
            'do(brain_volume=1600000)': {'brain_volume': torch.ones_like(batch['brain_volume']) * 1600000},
            'do(structure_volume=10000)': {'structure_volume': torch.ones_like(batch['structure_volume']) * 10000},
            'do(structure_volume=50000)': {'structure_volume': torch.ones_like(batch['structure_volume']) * 50000},
            'do(structure_volume=110000)': {'structure_volume': torch.ones_like(batch['structure_volume']) * 110000},
            'do(age=40)': {'age': torch.ones_like(batch['age']) * 40},
            'do(age=60)': {'age': torch.ones_like(batch['age']) * 60},
            'do(age=80)': {'age': torch.ones_like(batch['age']) * 80},
            'do(age=100)': {'age': torch.ones_like(batch['age']) * 100},
            'do(age=120)': {'age': torch.ones_like(batch['age']) * 120},
            'do(sex=0)': {'sex': torch.zeros_like(batch['sex'])},
            'do(sex=1)': {'sex': torch.ones_like(batch['sex'])},
            'do(brain_volume=800000, structure_volume=224)': {'brain_volume': torch.ones_like(batch['brain_volume']) * 800000,
                                                              'structure_volume': torch.ones_like(batch['structure_volume']) * 110000},
            'do(brain_volume=1600000, structure_volume=10000)': {'brain_volume': torch.ones_like(batch['brain_volume']) * 1600000,
                                                                 'structure_volume': torch.ones_like(batch['structure_volume']) * 10000}
        }

        return counterfactuals

    def build_test_samples(self, batch):
        samples = {}
        reconstruction = self.pyro_model.reconstruct(
            **batch,
            num_particles=self.hparams.num_sample_particles
        )
        samples['reconstruction'] = {'x': reconstruction}

        counterfactuals = self.get_counterfactual_conditions(batch)

        for name, condition in counterfactuals.items():
            samples[name] = self.pyro_model._gen_counterfactual(obs=batch, condition=condition)

        return samples

    def log_meshes(self, tag, verts, faces, colors, normalize=True, save_img=False, **kwargs):
        if save_img:
            p = os.path.join(self.trainer.logger.experiment.log_dir, f'{tag}.png')
            torchvision.utils.save_image(imgs, p)
        camera_config = {
	    'cls': 'PerspectiveCamera',
            'fov': 75,
            'aspect': 0.9,
        }
        material_config = {
            'cls': 'MeshDepthMaterial',
            'wireframe': True,
        }
        config_dict = {
            'material': material_config,
            'camera': camera_config,
        }
        self.logger.experiment.add_mesh(
            tag,
            vertices=verts,
            colors=colors,
            faces=faces,
            global_step=self.current_epoch,
            config_dict=config_dict,
        )

    def get_batch(self, loader):
        batch = next(iter(self.val_loader))
        # if self.trainer.on_gpu:
        #    batch = self.trainer.accelerator_backend.to_device(batch, self.torch_device)
        return batch

    def log_kdes(self, tag, data, save_img=False):
        def np_val(x):
            return x.cpu().numpy().squeeze() if isinstance(x, torch.Tensor) else x.squeeze()

        fig, ax = plt.subplots(1, len(data), figsize=(5 * len(data), 3), sharex=True, sharey=True)
        for i, (name, covariates) in enumerate(data.items()):
            try:
                if len(covariates) == 1:
                    (x_n, x), = tuple(covariates.items())
                    sns.kdeplot(x=np_val(x), ax=ax[i], shade=True, thresh=0.05)
                elif len(covariates) == 2:
                    (x_n, x), (y_n, y) = tuple(covariates.items())
                    sns.kdeplot(x=np_val(x), y=np_val(y), ax=ax[i], shade=True, thresh=0.05)
                    ax[i].set_ylabel(y_n)
                else:
                    raise ValueError(f'got too many values: {len(covariates)}')
            except np.linalg.LinAlgError:
                print(f'got a linalg error when plotting {tag}/{name}')

            ax[i].set_title(name)
            ax[i].set_xlabel(x_n)

        sns.despine()

        if save_img:
            p = os.path.join(self.trainer.logger.experiment.log_dir, f'{tag}.png')
            plt.savefig(p, dpi=300)

        self.logger.experiment.add_figure(tag, fig, self.current_epoch)

    def build_reconstruction(self, x, age, sex, structure_volume, brain_volume, tag='reconstruction'):
        obs = {
            'x': x,
            'age': age,
            'sex': sex,
            'structure_volume': structure_volume,
            'brain_volume': brain_volume
        }
        recon = self.pyro_model.reconstruct(
            **obs,
            num_particles=self.hparams.num_sample_particles
        )

        face = self.pyro_model.template.face.T

        # TODO: Add difference colours
        for i in range(x.shape[0]):
            _tag = f'{tag}/age:{age[i]}, sex:{sex[i]}, sv:{structure_volume[i]}, bv:{brain_volume[i]}'
            self.log_side_by_side_meshes(_tag, x[i], recon[i], face, self.current_epoch)

        self.logger.experiment.add_scalar(
            f'{tag}/mse',
            torch.mean(torch.square(x - recon).sum((1, 2))),
            self.current_epoch,
        )

    def log_side_by_side_meshes(self, tag, mesh_1, mesh_2, face, step):
        # TODO: Move all wiremesh settings out 
        camera_config = {
	    'cls': 'PerspectiveCamera',
            'fov': 75,
            'aspect': 0.9,
        }
        material_config = {
            'cls': 'MeshDepthMaterial',
            'wireframe': True,
        }
        config_dict = {
            'material': material_config,
            'camera': camera_config,
        }
        # print(tag)
        self.logger.experiment.add_mesh(
            tag,
            vertices=torch.cat([mesh_1.unsqueeze(0), mesh_2.unsqueeze(0)], 0),
            faces=face.unsqueeze(0).repeat_interleave(2, 0),
            global_step=step,
            config_dict=config_dict,
        )

    def build_counterfactual(self, obs, conditions, absolute=None):
        _required_data = ('x', 'age', 'sex', 'structure_volume', 'brain_volume')
        assert set(obs.keys()) == set(_required_data), 'got: {}'.format(tuple(obs.keys()))

        # meshes = [obs['x']]
        # TODO: decide which kde's to plot in which configuration
        if absolute == 'brain_volume':
            sampled_kdes = {'orig': {'structure_volume': obs['structure_volume']}}
        elif absolute == 'structure_volume':
            sampled_kdes = {'orig': {'brain_volume': obs['brain_volume']}}
        else:
            sampled_kdes = {'orig': {
                'brain_volume': obs['brain_volume'],
                'structure_volume': obs['structure_volume']
            }}

        meshes = []
        for name, data in conditions.items():
            counterfactual = self.pyro_model._gen_counterfactual(obs=obs, condition=data)

            counter = counterfactual['x']
            meshes.append(counter)

            sampled_brain_volume = counterfactual['brain_volume']
            sampled_structure_volume = counterfactual['structure_volume']
            if absolute == 'brain_volume':
                sampled_kdes[name] = {'structure_volume': sampled_structure_volume}
            elif absolute == 'structure_volume':
                sampled_kdes[name] = {'brain_volume': sampled_brain_volume}
            else:
                sampled_kdes[name] = {
                    'brain_volume': sampled_brain_volume,
                    'structure_volume': sampled_structure_volume
                }

        return meshes, sampled_kdes 

    def sample_images(self):
        with torch.no_grad():
            # TODO: redo all this....
            vis_meshes = 4
            faces = self.pyro_model.template.face.T \
                .unsqueeze(0) \
                .repeat_interleave(vis_meshes, 0)

            sample_trace = pyro.poutine.trace(self.pyro_model.sample) \
                .get_trace(self.hparams.test_batch_size)

            samples = sample_trace.nodes['x']['value']
            sampled_brain_volume = sample_trace.nodes['brain_volume']['value']
            sampled_structure_volume = sample_trace.nodes['structure_volume']['value']
            
            self.log_meshes('samples', samples.data[:vis_meshes], faces, None)

            cond_data = {
                'brain_volume': self.brain_volume_range,
                'structure_volume': self.structure_volume_range,
                'z': self.z_range
            }
            samples, *_ = pyro.condition(self.pyro_model.sample, data=cond_data)(9)
            self.log_meshes('cond_samples', samples.data[:vis_meshes], faces, None,)

            # TODO: Get the indices which I have choosen from the notebook
            obs_batch = self.prep_batch(self.get_batch(self.val_loader))

            kde_data = {
                'batch': {
                    'brain_volume': obs_batch['brain_volume'],
                    'structure_volume': obs_batch['structure_volume']
                },
                'sampled': {
                    'brain_volume': sampled_brain_volume,
                    'structure_volume': sampled_structure_volume
                },
            }
            self.log_kdes('sample_kde', kde_data, save_img=True)

            exogeneous = self.pyro_model.infer(**obs_batch)

            for (tag, val) in exogeneous.items():
                self.logger.experiment.add_histogram(tag, val, self.current_epoch)

            # Batch for visualisations
            obs_batch = {k: v for k, v in obs_batch.items()}

            ####
            # Sanity Check - plotting inputs
            ###
            self.log_meshes('input', obs_batch['x'][:vis_meshes], faces, None)

            ####
            # Reconstructions - plotted beside corresponding input
            ###
            self.build_reconstruction(**{
                k: v[:vis_meshes] for k, v in obs_batch.items()
            })

            ######
            # Plotting Age Counterfactuals
            ######
            # TODO: Wider age ranges
            conditions = {
                '40': {'age': torch.zeros_like(obs_batch['age']) + 40},
                '60': {'age': torch.zeros_like(obs_batch['age']) + 60},
                '80': {'age': torch.zeros_like(obs_batch['age']) + 80}
            }
            meshes, sampled_kdes = self.build_counterfactual(
                obs=obs_batch,
                conditions=conditions,
            )

            self.log_kdes(f'do(age=x)_sampled', sampled_kdes, save_img=True)

            for i in range(vis_meshes):
                cond_meshes = [meshes[c][i].unsqueeze(0) for c in range(len(conditions))]
                plot_meshes = [obs_batch['x'][i].unsqueeze(0)] + cond_meshes
                tag = f'do(age=x) age:{obs_batch["age"][i]} sex:{obs_batch["sex"][i]} sv:{obs_batch["structure_volume"][i]} bv:{obs_batch["brain_volume"][i]}' 
                self.log_meshes(tag, torch.cat(plot_meshes, 0), faces, None)

            ######
            # Plotting Sex Counterfactuals
            ######
            conditions = {
                '0': {'sex': torch.zeros_like(obs_batch['sex'])},
                '1': {'sex': torch.ones_like(obs_batch['sex'])},
            }
            meshes, sampled_kdes = self.build_counterfactual(
                obs=obs_batch,
                conditions=conditions,
            )

            self.log_kdes(f'do(sex=x)_sampled', sampled_kdes, save_img=True)

            for i in range(vis_meshes):
                plot_meshes = [meshes[c][i].unsqueeze(0) for c in range(len(conditions))]
                tag = f'do(sex=x) age:{obs_batch["age"][i]} sex:{obs_batch["sex"][i]} sv:{obs_batch["structure_volume"][i]} bv:{obs_batch["brain_volume"][i]}' 
                self.log_meshes(tag, torch.cat(plot_meshes, 0), faces, None)

            """
            ######
            # Plotting Brain Volume Counterfactuals
            ######
            conditions = {
                '800000': {'brain_volume': torch.zeros_like(obs_batch['brain_volume']) + 800000},
                '1100000': {'brain_volume': torch.zeros_like(obs_batch['brain_volume']) + 1100000},
                '1400000': {'brain_volume': torch.zeros_like(obs_batch['brain_volume']) + 1400000},
                '1600000': {'brain_volume': torch.zeros_like(obs_batch['brain_volume']) + 1600000}
            }
            self.build_counterfactual(
                'do(brain_volume=x)',
                obs=obs_batch,
                conditions=conditions,
                absolute='brain_volume'
            )

            # TODO: Based on which structure is being modelled 
            conditions = {
                '10000': {'structure_volume': torch.zeros_like(obs_batch['structure_volume']) + 10000},
                '25000': {'structure_volume': torch.zeros_like(obs_batch['structure_volume']) + 25000},
                '50000': {'structure_volume': torch.zeros_like(obs_batch['structure_volume']) + 50000},
                '75000': {'structure_volume': torch.zeros_like(obs_batch['structure_volume']) + 75000},
                '110000': {'structure_volume': torch.zeros_like(obs_batch['structure_volume']) + 110000}
            }
            self.build_counterfactual(
                'do(structure_volume=x)',
                obs=obs_batch,
                conditions=conditions,
                absolute='structure_volume'
            )
            """

    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--data_dir', default="/vol/biomedic3/bglocker/brainshapes", type=str, help="data dir (default: %(default)s)")  # noqa: E501
        parser.add_argument('--csv_path', default="/vol/biomedic3/bglocker/brainshapes/ukb21079_extracted.csv", type=str, help="csv metadata dir (default: %(default)s)")  # noqa: E501
        parser.add_argument('--cache_dir', default="/vol/bitbucket/rrr2417/deepscm_data_cache", type=str, help="location to cache dataset metadata (default: %(default)s)")  # noqa: E501
        parser.add_argument('--sample_mesh_interval', default=1, type=int, help="interval in which to sample and log meshes (default: %(default)s)")
        parser.add_argument('--train_batch_size', default=10, type=int, help="train batch size (default: %(default)s)")
        parser.add_argument('--test_batch_size', default=50, type=int, help="train batch size (default: %(default)s)")
        parser.add_argument('--train_test_split', default=0.8, type=float, help="train-test split (default: %(default)s)")
        parser.add_argument('--train_val_split', default=0.1, type=float, help="fraction of train set to allocate as val (default: %(default)s)")
        substructures = [
            'BrStem', 'L_Accu', 'L_Amyg', 'L_Caud', 'L_Hipp',
            'L_Pall', 'L_Puta', 'L_Thal', 'R_Accu', 'R_Amyg',
            'R_Caud', 'R_Hipp', 'R_Pall', 'R_Puta', 'R_Thal',
        ]
        parser.add_argument(
            '--brain_substructure',
            default='BrStem',
            choices=substructures,
            help="Substructure to model (default: %(default)s)"
        )
        parser.add_argument('--validate', default=False, action='store_true', help="whether to validate (default: %(default)s)")
        parser.add_argument('--lr', default=1e-4, type=float, help="lr of deep part (default: %(default)s)")
        parser.add_argument('--pgm_lr', default=5e-3, type=float, help="lr of pgm (default: %(default)s)")
        parser.add_argument('--reload_mesh_path', default=True, type=bool, help="reload mesh metdata from csv path (default: %(default)s)")
        parser.add_argument('--l2', default=0., type=float, help="weight decay (default: %(default)s)")
        parser.add_argument('--use_amsgrad', default=False, action='store_true', help="use amsgrad? (default: %(default)s)")
        return parser
