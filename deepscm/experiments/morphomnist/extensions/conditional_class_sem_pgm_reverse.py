import torch
import pyro

from pyro.nn import pyro_method
from pyro.distributions import Normal, TransformedDistribution, Categorical
from pyro.distributions.transforms import (
    ComposeTransform, ExpTransform, Spline, SigmoidTransform,
    conditional_spline
)
from pyro.distributions.torch_transform import ComposeTransformModule
from pyro.distributions.conditional import ConditionalTransformedDistribution
from deepscm.distributions.transforms.affine import LearnedAffineTransform
from pyro.nn import DenseNN

from .base_sem_experiment import BaseVISEM, MODEL_REGISTRY


class ConditionalClassReversedVISEM(BaseVISEM):
    # For each value that is concatentated into the latent vector
    context_dim = 3

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # print(kwargs)

        # Learned affine flow for intensity (Normal)
        # self.intensity_flow_components = LearnedAffineTransform()  # context_nn=intensity_net, event_dim=0)
        # self.intensity_flow_constraint_transforms = ComposeTransform([SigmoidTransform(), self.intensity_flow_norm])
        # self.intensity_flow_transforms = [self.intensity_flow_components, self.intensity_flow_constraint_transforms]
        self.intensity_flow_components = ComposeTransformModule([LearnedAffineTransform(), Spline(1)])
        self.intensity_flow_constraint_transforms = ComposeTransform([SigmoidTransform(), self.intensity_flow_norm])
        self.intensity_flow_transforms = [self.intensity_flow_components, self.intensity_flow_constraint_transforms]

        # Conditional Spline flow for thickness (Gamma)
        # self.thickness_flow_components = ComposeTransformModule([Spline(1)])
        self.thickness_flow_components = conditional_spline(1, 1, count_bins=8, bound=3., order='linear')
        self.thickness_flow_constraint_transforms = ComposeTransform([self.thickness_flow_lognorm, ExpTransform()])
        self.thickness_flow_transforms = [self.thickness_flow_components, self.thickness_flow_constraint_transforms]
        # self.thickness_flow_preprocess = ComposeTransform([
        #     self.thickness_flow_lognorm,
        #     SigmoidTransform()
        # ])
        # self.thickness_flow_components = conditional_spline(1, 1, count_bins=8, bound=1)
        # self.thickness_flow_constraint_transforms = ComposeTransform([
        #     SigmoidTransform().inv,
        #     # self.thickness_flow_lognorm,
        #     # ExpTransform()
        # ])
        # self.thickness_flow_transforms = [
        #     self.thickness_flow_preprocess,
        #     self.thickness_flow_components,
        #     self.thickness_flow_constraint_transforms,
        # ]

    @pyro_method
    def pgm_model(self):
        label_dist = Categorical(self.label_probs)
        label = pyro.sample('label', label_dist)
        # print(f'label: {label.shape}')

        intensity_base_dist = Normal(self.intensity_base_loc, self.intensity_base_scale).to_event(1)
        intensity_dist = TransformedDistribution(intensity_base_dist, self.intensity_flow_transforms)

        intensity = pyro.sample('intensity', intensity_dist)
        intensity_ = self.intensity_flow_constraint_transforms.inv(intensity)
        # pseudo call to intensity_flow_transforms to register with pyro
        _ = self.intensity_flow_components

        thickness_base_dist = Normal(self.thickness_base_loc, self.thickness_base_scale).to_event(1)
        # print('thickness_dist params:', thickness_base_dist, self.thickness_flow_transforms)
        thickness_dist = ConditionalTransformedDistribution(
                thickness_base_dist, self.thickness_flow_transforms).condition(intensity_)

        # print('thickness dist:', thickness_dist)

        thickness = pyro.sample('thickness', thickness_dist)
        # print('thickness sample:', thickness.shape)
        # pseudo call to thickness_flow_transforms to register with pyro
        _ = self.thickness_flow_components

        return thickness, intensity, label

    @pyro_method
    def model(self):
        # print('MODEL')
        thickness, intensity, label = self.pgm_model()
        # print(f'label: {label.shape}')

        thickness_ = self.thickness_flow_constraint_transforms.inv(thickness)
        intensity_ = self.intensity_flow_norm.inv(intensity)

        z = pyro.sample('z', Normal(self.z_loc, self.z_scale).to_event(1))
        # print(f'z: {z.shape}, thickness: {thickness_.shape}, intensity: {intensity_.shape}, label: {label.shape}')

        latent = torch.cat([z, thickness_, intensity_, label.view(-1, 1)], 1)
        # print('LATENT SPACE:', latent.shape)

        x_dist = self._get_transformed_x_dist(latent)

        x = pyro.sample('x', x_dist)
        # print('Image shape:', x.shape)

        return x, z, thickness, intensity, label

    @pyro_method
    def guide(self, x, thickness, intensity, label):
        with pyro.plate('observations', x.shape[0]):
            # print('X GUIDE shape:', x.shape)
            hidden = self.encoder(x)

            # print('thickness data:', thickness.shape)

            intensity_ = self.intensity_flow_norm.inv(intensity)
            thickness_ = self.thickness_flow_constraint_transforms.inv(thickness)

            hidden = torch.cat([hidden, thickness_, intensity_, label], 1)
            latent_dist = self.latent_encoder.predict(hidden)

            z = pyro.sample('z', latent_dist)

        return z

    @pyro_method
    def infer_thickness_base(self, thickness):
        return self.thickness_flow_transforms.inv(thickness)

    @pyro_method
    def infer_intensity_base(self, thickness, intensity):
        intensity_base_dist = Normal(self.intensity_base_loc, self.intensity_base_scale)
        cond_intensity_transforms = ComposeTransform(
            ConditionalTransformedDistribution(intensity_base_dist, self.intensity_flow_transforms).condition(thickness).transforms)
        return cond_intensity_transforms.inv(intensity)


MODEL_REGISTRY[ConditionalClassReversedVISEM.__name__] = ConditionalClassReversedVISEM
