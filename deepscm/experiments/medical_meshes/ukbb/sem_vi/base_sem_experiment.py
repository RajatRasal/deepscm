import pyro

from typing import Mapping, List

from pyro.infer import SVI, TraceGraph_ELBO
from pyro.nn import pyro_method
from pyro.optim import Adam
from torch.distributions import Independent

import torch
from pyro.distributions.torch_transform import ComposeTransformModule
from pyro.distributions.transforms import (
    ComposeTransform, AffineTransform, ExpTransform, Spline
)
from pyro.distributions import (
    LowRankMultivariateNormal, MultivariateNormal, Normal, TransformedDistribution
)
# from deepscm.arch.medical import Decoder, Encoder
from deepscm.distributions.transforms.reshape import ReshapeTransform
from deepscm.distributions.transforms.affine import LowerCholeskyAffine

from deepscm.distributions.deep import (
    DeepMultivariateNormal, DeepIndepNormal, DeepLowRankMultivariateNormal
)
from coma.models import init_coma_pooling
from coma.models.components import Encoder, Decoder 
from coma.datasets.ukbb_meshdata import get_data_from_polydata

import numpy as np
import pyvista as pv

from deepscm.experiments.medical_meshes.base_experiment import (
    BaseCovariateExperiment, BaseSEM, EXPERIMENT_REGISTRY, MODEL_REGISTRY  # noqa: F401
)


class CustomELBO(TraceGraph_ELBO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.trace_storage = {'model': None, 'guide': None}

    def _get_trace(self, model, guide, args, kwargs):
        model_trace, guide_trace = super()._get_trace(model, guide, args, kwargs)

        self.trace_storage['model'] = model_trace
        self.trace_storage['guide'] = guide_trace

        return model_trace, guide_trace


class Lambda(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class BaseVISEM(BaseSEM):
    context_dim = 0

    def __init__(self, latent_dim: int = 8, logstd_init: float = -5, in_channels: int = 3,
        filters: List[int] = [32, 32, 32, 64], pooling_factor: int = 4, cheb_k: int = 10,
        decoder_type: str = 'normal', decoder_cov_rank: int = 10, 
        template_path: str = '/vol/biomedic3/bglocker/brainshapes/5026976/T1_first-BrStem_first.vtk',
        gpu: int = 1, img_shape: List[int] = [642, 3], **kwargs
    ):
        super().__init__(**kwargs)

        # TODO: Remove hard coding
        self.img_shape = tuple(img_shape)

        self.latent_dim = latent_dim
        self.logstd_init = logstd_init

        self.in_channels = in_channels
        self.filters = filters 
        self.depth = len(self.filters)
        self.cheb_k = cheb_k 
        self.decoder_type = decoder_type
        self.decoder_cov_rank = decoder_cov_rank
        self.pooling_factor = pooling_factor

        # Get template brain mesh 
        template_polydata = pv.read(template_path)
        self.template = get_data_from_polydata(template_polydata)

        # CoMA pooling parameters
        # TODO: Remove this hack?
        device = torch.device('cuda' if gpu else 'cpu')
        edge_index_list, down_transform_list, up_transform_list = init_coma_pooling(
            self.template, self.pooling_factor, self.depth, device,
        )

        # Graph Decoder
        decoder_in_dim = self.latent_dim + self.context_dim
        decoder = Decoder(
            in_channels=self.in_channels,
            out_channels=self.filters,
            latent_channels=decoder_in_dim,
            edge_index=edge_index_list,
            down_transform=down_transform_list,
            up_transform=up_transform_list,
            K=self.cheb_k, n_blocks=1,
        )
        seq = torch.nn.Sequential(
            decoder,
            Lambda(lambda x: x.view(x.shape[0], -1))
        )
        decoder_out_dim = np.prod(self.img_shape)
        out_dim = np.prod(self.img_shape)
        out_dist_args = [seq, decoder_out_dim, out_dim]

        if 'lowrank' in self.decoder_type:
            out_dist_args.append(decoder_cov_rank)

        decoders = {
            'normal': DeepIndepNormal,
            'multivariate_gaussian': DeepMultivariateNormal,
            'sharedvar_multivariate_gaussian': DeepMultivariateNormal,
            'lowrank_multivariate_gaussian': DeepLowRankMultivariateNormal, 
            'sharedvar_lowrank_multivariate_gaussian': DeepLowRankMultivariateNormal,
        }

        self.decoder = decoders[self.decoder_type](*out_dist_args)

        if self.decoder_type == 'sharedvar_multivariate_gaussian':
            torch.nn.init.zeros_(self.decoder.logdiag_head.weight)
            self.decoder.logdiag_head.weight.requires_grad = False

            torch.nn.init.zeros_(self.decoder.lower_head.weight)
            self.decoder.lower_head.weight.requires_grad = False

            torch.nn.init.normal_(self.decoder.logdiag_head.bias, self.logstd_init, 1e-1)
            self.decoder.logdiag_head.bias.requires_grad = True
        elif self.decoder_type == 'sharedvar_lowrank_multivariate_gaussian':
            torch.nn.init.zeros_(self.decoder.logdiag_head.weight)
            self.decoder.logdiag_head.weight.requires_grad = False

            torch.nn.init.zeros_(self.decoder.factor_head.weight)
            self.decoder.factor_head.weight.requires_grad = False

            torch.nn.init.normal_(self.decoder.logdiag_head.bias, self.logstd_init, 1e-1)
            self.decoder.logdiag_head.bias.requires_grad = True

        # encoder parts
        self.encoder = Encoder(
            in_channels=self.in_channels,
            out_channels=self.filters,
            latent_channels=self.latent_dim,
            edge_index=edge_index_list,
            down_transform=down_transform_list,
            up_transform=up_transform_list,
            K=self.cheb_k, n_blocks=1,
        )

        latent_layers = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim + self.context_dim, self.latent_dim),
            torch.nn.ReLU()
        )
        self.latent_encoder = DeepIndepNormal(
            latent_layers,
            self.latent_dim,
            self.latent_dim,
        )

        # priors
        self.register_buffer('age_base_loc', torch.zeros([1, ], requires_grad=False))
        self.register_buffer('age_base_scale', torch.ones([1, ], requires_grad=False))

        self.sex_logits = torch.nn.Parameter(torch.zeros([1, ]))

        self.register_buffer('structure_volume_base_loc', torch.zeros([1, ], requires_grad=False))
        self.register_buffer('structure_volume_base_scale', torch.ones([1, ], requires_grad=False))

        self.register_buffer('brain_volume_base_loc', torch.zeros([1, ], requires_grad=False))
        self.register_buffer('brain_volume_base_scale', torch.ones([1, ], requires_grad=False))

        self.register_buffer('z_loc', torch.zeros([latent_dim, ], requires_grad=False))
        self.register_buffer('z_scale', torch.ones([latent_dim, ], requires_grad=False))

        self.register_buffer('x_base_loc', torch.zeros(self.img_shape, requires_grad=False))
        self.register_buffer('x_base_scale', torch.ones(self.img_shape, requires_grad=False))

        self.register_buffer('age_flow_lognorm_loc', torch.zeros([], requires_grad=False))
        self.register_buffer('age_flow_lognorm_scale', torch.ones([], requires_grad=False))

        self.register_buffer('structure_volume_flow_lognorm_loc', torch.zeros([], requires_grad=False))
        self.register_buffer('structure_volume_flow_lognorm_scale', torch.ones([], requires_grad=False))

        self.register_buffer('brain_volume_flow_lognorm_loc', torch.zeros([], requires_grad=False))
        self.register_buffer('brain_volume_flow_lognorm_scale', torch.ones([], requires_grad=False))

        # age flow
        self.age_flow_components = ComposeTransformModule([
            Spline(1)
        ])
        self.age_flow_lognorm = AffineTransform(
            loc=self.age_flow_lognorm_loc.item(),
            scale=self.age_flow_lognorm_scale.item()
        )
        self.age_flow_constraint_transforms = ComposeTransform([
            self.age_flow_lognorm,
            ExpTransform()
        ])
        self.age_flow_transforms = ComposeTransform([
            self.age_flow_components,
            self.age_flow_constraint_transforms
        ])

        # other flows shared components
        self.structure_volume_flow_lognorm = AffineTransform(
            loc=self.structure_volume_flow_lognorm_loc.item(),
            scale=self.structure_volume_flow_lognorm_scale.item()
        )  # noqa: E501
        self.structure_volume_flow_constraint_transforms = ComposeTransform([
            self.structure_volume_flow_lognorm,
            ExpTransform()
        ])

        self.brain_volume_flow_lognorm = AffineTransform(
            loc=self.brain_volume_flow_lognorm_loc.item(),
            scale=self.brain_volume_flow_lognorm_scale.item()
        )
        self.brain_volume_flow_constraint_transforms = ComposeTransform([
            self.brain_volume_flow_lognorm,
            ExpTransform()
        ])

    def __setattr__(self, name, value):
        super().__setattr__(name, value)

        if name == 'age_flow_lognorm_loc':
            self.age_flow_lognorm.loc = self.age_flow_lognorm_loc.item()
        elif name == 'age_flow_lognorm_scale':
            self.age_flow_lognorm.scale = self.age_flow_lognorm_scale.item()
        elif name == 'structure_volume_flow_lognorm_loc':
            self.structure_volume_flow_lognorm.loc = self.structure_volume_flow_lognorm_loc.item()
        elif name == 'structure_volume_flow_lognorm_scale':
            self.structure_volume_flow_lognorm.scale = self.structure_volume_flow_lognorm_scale.item()
        elif name == 'brain_volume_flow_lognorm_loc':
            self.brain_volume_flow_lognorm.loc = self.brain_volume_flow_lognorm_loc.item()
        elif name == 'brain_volume_flow_lognorm_scale':
            self.brain_volume_flow_lognorm.scale = self.brain_volume_flow_lognorm_scale.item()

    def _get_preprocess_transforms(self):
        return super()._get_preprocess_transforms().inv

    def _get_transformed_x_dist(self, latent):
        x_pred_dist = self.decoder.predict(latent)
        x_base_dist = Normal(self.x_base_loc, self.x_base_scale).to_event(2)

        reshape_transform = ReshapeTransform(self.img_shape, (np.prod(self.img_shape), ))

        if isinstance(x_pred_dist, MultivariateNormal) or isinstance(x_pred_dist, LowRankMultivariateNormal):
            transform = LowerCholeskyAffine(x_pred_dist.loc, x_pred_dist.scale_tril)
        elif isinstance(x_pred_dist, Independent) or isinstance(x_pred_dist, Normal):
            x_pred_dist = x_pred_dist.base_dist
            transform = AffineTransform(x_pred_dist.loc, x_pred_dist.scale, 2)

        return TransformedDistribution(
            x_base_dist,
            ComposeTransform([reshape_transform, transform, reshape_transform.inv])
        )

    @pyro_method
    def guide(self, x, age, sex, structure_volume, brain_volume):
        raise NotImplementedError()

    @pyro_method
    def svi_guide(self, x, age, sex, structure_volume, brain_volume):
        self.guide(x, age, sex, structure_volume, brain_volume)

    @pyro_method
    def svi_model(self, x, age, sex, structure_volume, brain_volume):
        with pyro.plate('observations', x.shape[0]):
            pyro.condition(
                self.model,
                data={
                    'x': x,
                    'sex': sex,
                    'age': age,
                    'structure_volume': structure_volume,
                    'brain_volume': brain_volume
                })()

    @pyro_method
    def infer_z(self, *args, **kwargs):
        return self.guide(*args, **kwargs)

    @pyro_method
    def infer(self, **obs):
        _required_data = ('x', 'sex', 'age', 'structure_volume', 'brain_volume')
        assert set(obs.keys()) == set(_required_data), 'got: {}'.format(tuple(obs.keys()))

        z = self.infer_z(**obs)

        exogeneous = self.infer_exogeneous(z=z, **obs)
        exogeneous['z'] = z

        return exogeneous

    @pyro_method
    def reconstruct(self, x, age, sex, structure_volume, brain_volume, num_particles: int = 1):
        obs = {'x': x, 'sex': sex, 'age': age, 'structure_volume': structure_volume, 'brain_volume': brain_volume}
        z_dist = pyro.poutine.trace(self.guide).get_trace(**obs).nodes['z']['fn']

        recons = []
        for _ in range(num_particles):
            z = pyro.sample('z', z_dist)
            recon, *_ = pyro.poutine.condition(
                self.sample,
                data={
                    'sex': sex,
                    'age': age,
                    'structure_volume': structure_volume,
                    'brain_volume': brain_volume,
                    'z': z
                })(x.shape[0])
            recons += [recon]
        return torch.stack(recons).mean(0)

    @pyro_method
    def counterfactual(self, obs: Mapping, condition: Mapping = None, num_particles: int = 1):
        _required_data = ('x', 'sex', 'age', 'structure_volume', 'brain_volume')
        assert set(obs.keys()) == set(_required_data), 'got: {}'.format(tuple(obs.keys()))

        z_dist = pyro.poutine.trace(self.guide).get_trace(**obs).nodes['z']['fn']

        counterfactuals = []
        for _ in range(num_particles):
            z = pyro.sample('z', z_dist)

            exogeneous = self.infer_exogeneous(z=z, **obs)
            exogeneous['z'] = z
            # condition on sex if sex isn't included in 'do' as it's a root node and we don't have the exogeneous noise for it yet...
            if 'sex' not in condition.keys():
                exogeneous['sex'] = obs['sex']

            counter = pyro.poutine.do(pyro.poutine.condition(self.sample_scm, data=exogeneous), data=condition)(obs['x'].shape[0])
            counterfactuals += [counter]
        return {k: v for k, v in zip(('x', 'z', 'sex', 'age', 'structure_volume', 'brain_volume'), (torch.stack(c).mean(0) for c in zip(*counterfactuals)))}

    @classmethod
    def add_arguments(cls, parser):
        parser = super().add_arguments(parser)

        parser.add_argument(
            '--latent_dim',
            default=8,
            type=int,
            help="latent dimension of model (default: %(default)s)"
        )
        parser.add_argument(
            '--logstd_init',
            default=-5,
            type=float,
            help="init of logstd (default: %(default)s)"
        )
        parser.add_argument(
            '--in_channels',
            default=3,
            type=int,
            help="input dimensions (default: %(default)s)"
        )
        parser.add_argument(
            '--filters',
            default=[16, 32, 64, 128],
            nargs='+',
            type=int,
            help="filters at each layer (default: %(default)s)"
        )
        parser.add_argument(
            '--pooling_factor',
            default=2,
            type=int,
            help="pooling factor (default: %(default)s)"
        )
        parser.add_argument(
            '--cheb_k',
            default=10,
            type=int,
            help="hops in ChebConv (default: %(default)s)"
        )
        parser.add_argument(
            '--decoder_type', default='normal', help="var type (default: %(default)s)",
            choices=['normal', 'sharedvar_multivariate_gaussian', 'multivariate_gaussian',
                     'sharedvar_lowrank_multivariate_gaussian', 'lowrank_multivariate_gaussian'])
        parser.add_argument(
            '--decoder_cov_rank',
            default=10,
            type=int,
            help="rank for lowrank cov approximation (requires lowrank decoder) (default: %(default)s)"
        )  # noqa: E501
        parser.add_argument(
            '--template_path',
            default='/vol/biomedic3/bglocker/brainshapes/5026976/T1_first-BrStem_first.vtk',
            type=str,
            help="Path for a mesh"
        )
        parser.add_argument(
            '--img_shape',
            default=[642, 3],
            nargs='+',
            type=int,
            help="Input shape size (default: %(default)s)"
        )

        return parser


class SVIExperiment(BaseCovariateExperiment):
    def __init__(self, hparams, pyro_model: BaseSEM):
        super().__init__(hparams, pyro_model)

        self.svi_loss = CustomELBO(num_particles=hparams.num_svi_particles)

        self._build_svi()

    def _build_svi(self, loss=None):
        def per_param_callable(module_name, param_name):
            params = {
                'eps': 1e-5,
                'amsgrad': self.hparams.use_amsgrad,
                'weight_decay': self.hparams.l2
            }
            if 'flow_components' in module_name or 'sex_logits' in param_name:
                params['lr'] = self.hparams.pgm_lr
            else:
                params['lr'] = self.hparams.lr

            print(f'building opt for {module_name} - {param_name} with p: {params}')
            return params

        if loss is None:
            loss = self.svi_loss

        if self.hparams.use_cf_guide:
            def guide(*args, **kwargs):
                return self.pyro_model.counterfactual_guide(
                    *args, **kwargs, counterfactual_type=self.hparams.cf_elbo_type
                )
            self.svi = SVI(self.pyro_model.svi_model, guide, Adam(per_param_callable), loss)
        else:
            self.svi = SVI(self.pyro_model.svi_model, self.pyro_model.svi_guide, Adam(per_param_callable), loss)
        self.svi.loss_class = loss

    def backward(self, *args, **kwargs):
        pass  # No loss to backpropagate since we're using Pyro's optimisation machinery

    def print_trace_updates(self, batch):
        with torch.no_grad():
            print('Traces:\n' + ('#' * 10))

            guide_trace = pyro.poutine.trace(self.pyro_model.svi_guide).get_trace(**batch)
            model_trace = pyro.poutine.trace(pyro.poutine.replay(self.pyro_model.svi_model, trace=guide_trace)).get_trace(**batch)

            guide_trace = pyro.poutine.util.prune_subsample_sites(guide_trace)
            model_trace = pyro.poutine.util.prune_subsample_sites(model_trace)

            model_trace.compute_log_prob()
            guide_trace.compute_score_parts()

            print(f'model: {model_trace.nodes.keys()}')
            for name, site in model_trace.nodes.items():
                if site["type"] == "sample":
                    fn = site['fn']
                    if isinstance(fn, Independent):
                        fn = fn.base_dist
                    print(f'{name}: {fn} - {fn.support}')
                    log_prob_sum = site["log_prob_sum"]
                    is_obs = site["is_observed"]
                    print(f'model - log p({name}) = {log_prob_sum} | obs={is_obs}')
                    if torch.isnan(log_prob_sum):
                        value = site['value'][0]
                        conc0 = fn.concentration0
                        conc1 = fn.concentration1

                        print(f'got:\n{value}\n{conc0}\n{conc1}')

                        raise Exception()

            print(f'guide: {guide_trace.nodes.keys()}')

            for name, site in guide_trace.nodes.items():
                if site["type"] == "sample":
                    fn = site['fn']
                    if isinstance(fn, Independent):
                        fn = fn.base_dist
                    print(f'{name}: {fn} - {fn.support}')
                    entropy = site["score_parts"].entropy_term.sum()
                    is_obs = site["is_observed"]
                    print(f'guide - log q({name}) = {entropy} | obs={is_obs}')

    def get_trace_metrics(self, batch):
        metrics = {}

        model = self.svi.loss_class.trace_storage['model']
        guide = self.svi.loss_class.trace_storage['guide']

        metrics['log p(x)'] = model.nodes['x']['log_prob'].mean()
        metrics['log p(age)'] = model.nodes['age']['log_prob'].mean()
        metrics['log p(sex)'] = model.nodes['sex']['log_prob'].mean()
        metrics['log p(structure_volume)'] = model.nodes['structure_volume']['log_prob'].mean()
        metrics['log p(brain_volume)'] = model.nodes['brain_volume']['log_prob'].mean()
        metrics['p(z)'] = model.nodes['z']['log_prob'].mean()
        metrics['q(z)'] = guide.nodes['z']['log_prob'].mean()
        metrics['log p(z) - log q(z)'] = metrics['p(z)'] - metrics['q(z)']

        return metrics

    def __series_to_batched_tensor(self, series):
        return torch.tensor(series.values).float().view(series.shape[0], -1).to(self.torch_device)

    def prep_batch(self, batch):
        # TODO: Batch will be a tuple with a dataframe + batch of graphs
        x = batch.x.float().to(self.torch_device)
        features = batch.features
        age = self.__series_to_batched_tensor(features.age)
        sex = self.__series_to_batched_tensor(features.sex)
        structure_volume = self.__series_to_batched_tensor(features.structure_volume)
        brain_volume = self.__series_to_batched_tensor(features.brain_volume)

        # if self.training:
        #     x += torch.rand_like(x)

        return {
            'x': x,
            'age': age,
            'sex': sex,
            'structure_volume': structure_volume,
            'brain_volume': brain_volume
        }
        return outs

    def training_step(self, batch, batch_idx):
        batch = self.prep_batch(batch)

        # if self.hparams.validate:
        # print('Validation:')
        # self.print_trace_updates(batch)

        loss = self.svi.step(**batch)

        metrics = self.get_trace_metrics(batch)

        if np.isnan(loss):
            self.logger.experiment.add_text('nan', f'nand at {self.current_epoch}:\n{metrics}')
            raise ValueError('loss went to nan with metrics:\n{}'.format(metrics))

        tensorboard_logs = {('train/' + k): v for k, v in metrics.items()}
        tensorboard_logs['train/loss'] = loss

        self.log_dict(tensorboard_logs)

        return torch.Tensor([loss])

    def validation_step(self, batch, batch_idx):
        batch = self.prep_batch(batch)

        loss = self.svi.evaluate_loss(**batch)

        metrics = self.get_trace_metrics(batch)

        return {'loss': loss, **metrics}

    def test_step(self, batch, batch_idx):
        batch = self.prep_batch(batch)

        loss = self.svi.evaluate_loss(**batch)

        metrics = self.get_trace_metrics(batch)

        samples = self.build_test_samples(batch)

        return {'loss': loss, **metrics, 'samples': samples}

    @classmethod
    def add_arguments(cls, parser):
        parser = super().add_arguments(parser)

        parser.add_argument(
            '--num_svi_particles',
            default=1,
            type=int,
            help="number of particles to use for ELBO (default: %(default)s)"
        )
        parser.add_argument(
            '--num_sample_particles',
            default=32,
            type=int,
            help="number of particles to use for MC sampling (default: %(default)s)"
        )
        parser.add_argument(
            '--use_cf_guide',
            default=False,
            action='store_true',
            help="whether to use counterfactual guide (default: %(default)s)"
        )
        parser.add_argument(
            '--cf_elbo_type',
            default=-1,
            choices=[-1, 0, 1, 2],
            help="-1: randomly select per batch, 0: shuffle thickness, 1: shuffle intensity, 2: shuffle both (default: %(default)s)"
        )

        return parser


EXPERIMENT_REGISTRY[SVIExperiment.__name__] = SVIExperiment
