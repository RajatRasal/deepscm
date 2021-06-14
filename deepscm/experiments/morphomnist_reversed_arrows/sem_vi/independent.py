import torch
import pyro

from pyro.nn import pyro_method
from pyro.distributions.torch_transform import ComposeTransformModule
from pyro.distributions import Normal, TransformedDistribution
from pyro.distributions.transforms import ComposeTransform, ExpTransform, SigmoidTransform, Spline

from .base_sem_experiment import BaseVISEM, MODEL_REGISTRY
from deepscm.distributions.transforms.affine import LearnedAffineTransform


class IndependentReversedVISEM(BaseVISEM):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Learned affine flow for intensity (Normal)
        self.intensity_flow_components = ComposeTransformModule([LearnedAffineTransform(), Spline(1)])
        self.intensity_flow_constraint_transforms = ComposeTransform([SigmoidTransform(), self.intensity_flow_norm])
        self.intensity_flow_transforms = ComposeTransform([self.intensity_flow_components, self.intensity_flow_constraint_transforms])

        # Conditional Spline flow for thickness (Gamma)
        self.thickness_flow_components = ComposeTransformModule([Spline(1)])
        self.thickness_flow_constraint_transforms = ComposeTransform([self.thickness_flow_lognorm, ExpTransform()])
        self.thickness_flow_transforms = [self.thickness_flow_components, self.thickness_flow_constraint_transforms]

        # self.thickness_flow_preprocess = ComposeTransform([self.thickness_flow_lognorm, SigmoidTransform()])
        # self.thickness_flow_components = spline(input_dim=1, count_bins=8, bound=1)
        # self.thickness_flow_constraint_transforms = ComposeTransform([
        #     SigmoidTransform().inv,
        # ])
        # self.thickness_flow_transforms = [
        #     self.thickness_flow_preprocess,
        #     self.thickness_flow_components,
        #     self.thickness_flow_constraint_transforms,
        # ]

    @pyro_method
    def pgm_model(self):
        intensity_base_dist = Normal(self.intensity_base_loc, self.intensity_base_scale).to_event(1)
        intensity_dist = TransformedDistribution(intensity_base_dist, self.intensity_flow_transforms)

        intensity = pyro.sample('intensity', intensity_dist)
        intensity_ = self.intensity_flow_constraint_transforms.inv(intensity)
        # pseudo call to intensity_flow_transforms to register with pyro
        _ = self.intensity_flow_components

        thickness_base_dist = Normal(self.thickness_base_loc, self.thickness_base_scale).to_event(1)
        thickness_dist = TransformedDistribution(thickness_base_dist, self.thickness_flow_transforms)

        thickness = pyro.sample('thickness', thickness_dist)
        # pseudo call to thickness_flow_transforms to register with pyro
        _ = self.thickness_flow_components

        return thickness, intensity

    @pyro_method
    def model(self):
        thickness, intensity = self.pgm_model()

        z = pyro.sample('z', Normal(self.z_loc, self.z_scale).to_event(1))

        x_dist = self._get_transformed_x_dist(z)

        x = pyro.sample('x', x_dist)

        return x, z, thickness, intensity

    @pyro_method
    def guide(self, x, thickness, intensity):
        with pyro.plate('observations', x.shape[0]):
            hidden = self.encoder(x)

            latent_dist = self.latent_encoder.predict(hidden)

            z = pyro.sample('z', latent_dist)

        return z

    # @pyro_method
    # def infer_thickness_base(self, thickness):
    #     return self.thickness_flow_transforms.inv(thickness)

    # @pyro_method
    # def infer_intensity_base(self, thickness, intensity):
    #     intensity_base_dist = Normal(self.intensity_base_loc, self.intensity_base_scale)
    #     cond_intensity_transforms = ComposeTransform(
    #         ConditionalTransformedDistribution(intensity_base_dist, self.intensity_flow_transforms).condition(thickness).transforms)
    #     return cond_intensity_transforms.inv(intensity)


MODEL_REGISTRY[IndependentReversedVISEM.__name__] = IndependentReversedVISEM
