import torch
import pyro

from pyro.nn import pyro_method
from pyro.distributions import Normal, TransformedDistribution
from pyro.distributions.transforms import (
    ComposeTransform, ExpTransform, Spline, SigmoidTransform,
    conditional_spline
)
from pyro.distributions.torch_transform import ComposeTransformModule
from pyro.distributions.conditional import ConditionalTransformedDistribution
from deepscm.distributions.transforms.affine import LearnedAffineTransform

from .base_sem_experiment import BaseVISEM, MODEL_REGISTRY


class ConditionalReversedVISEM(BaseVISEM):
    context_dim = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Learned affine flow for intensity (Normal)
        # self.intensity_flow_components = LearnedAffineTransform()
        self.intensity_flow_components = ComposeTransformModule([LearnedAffineTransform(), Spline(1)])
        self.intensity_flow_constraint_transforms = ComposeTransform([SigmoidTransform(), self.intensity_flow_norm])
        self.intensity_flow_transforms = ComposeTransform([
            self.intensity_flow_components,
            self.intensity_flow_constraint_transforms
        ])

        # Conditional Spline flow for thickness (Gamma)
        # count_bins = 8 and bound = 3 are the default parameters
        self.thickness_flow_components = conditional_spline(1, 1, count_bins=8, bound=3., order='linear')
        self.thickness_flow_constraint_transforms = ComposeTransform([self.thickness_flow_lognorm, ExpTransform()])
        self.thickness_flow_transforms = [self.thickness_flow_components, self.thickness_flow_constraint_transforms]

    @pyro_method
    def pgm_model(self):
        intensity_base_dist = Normal(self.intensity_base_loc, self.intensity_base_scale).to_event(1)
        # pseudo call to intensity_flow_transforms to register with pyro
        _ = self.intensity_flow_components

        thickness_base_dist = Normal(self.thickness_base_loc, self.thickness_base_scale).to_event(1)
        thickness_dist = ConditionalTransformedDistribution(thickness_base_dist, self.thickness_flow_transforms).condition(intensity_)

        thickness = pyro.sample('thickness', thickness_dist)
        # pseudo call to thickness_flow_transforms to register with pyro
        _ = self.thickness_flow_components

        return thickness, intensity

    @pyro_method
    def model(self):
        thickness, intensity = self.pgm_model()

        thickness_ = self.thickness_flow_constraint_transforms.inv(thickness)
        intensity_ = self.intensity_flow_norm.inv(intensity)

        z = pyro.sample('z', Normal(self.z_loc, self.z_scale).to_event(1))

        latent = torch.cat([z, thickness_, intensity_], 1)

        x_dist = self._get_transformed_x_dist(latent)

        x = pyro.sample('x', x_dist)

        return x, z, thickness, intensity

    @pyro_method
    def guide(self, x, thickness, intensity):
        with pyro.plate('observations', x.shape[0]):
            hidden = self.encoder(x)

            intensity_ = self.intensity_flow_norm.inv(intensity)
            thickness_ = self.thickness_flow_constraint_transforms.inv(thickness)

            hidden = torch.cat([hidden, thickness_, intensity_], 1)
            latent_dist = self.latent_encoder.predict(hidden)

            z = pyro.sample('z', latent_dist)

        return z

    @pyro_method
    def infer_intensity_base(self, intensity):
        return self.intensity_flow_transforms.inv(intensity)

    @pyro_method
    def infer_thickness_base(self, thickness, intensity):
        thickness_base_dist = Normal(self.thickness_base_loc, self.thickness_base_scale)
        cond_thickness_transforms = ComposeTransform(
            ConditionalTransformedDistribution(thickness_base_dist, self.thickness_flow_transforms) \
                .condition(intensity) \
                .transforms
        )
        return cond_thickness_transforms.inv(thickness)


MODEL_REGISTRY[ConditionalReversedVISEM.__name__] = ConditionalReversedVISEM
