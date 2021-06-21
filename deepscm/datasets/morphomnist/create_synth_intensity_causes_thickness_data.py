import numpy as np
import os
import pandas as pd
import pyro
import torch

from pyro.distributions import Gamma, Normal, TransformedDistribution, Categorical
from pyro.distributions.transforms import SigmoidTransform, AffineTransform, ComposeTransform
from tqdm import tqdm

from deepscm.datasets.morphomnist import load_morphomnist_like, save_morphomnist_like
from deepscm.datasets.morphomnist.transforms import SetThickness, ImageMorphology


def get_intensity(img):
    threshold = 0.5

    img_min, img_max = img.min(), img.max()
    mask = (img >= img_min + (img_max - img_min) * threshold)
    avg_intensity = np.median(img[mask])

    return avg_intensity


def model(n_samples=None, scale=0.5, invert=False):
    with pyro.plate('observations', n_samples):
        intensity_transform = ComposeTransform([
            SigmoidTransform(),
            AffineTransform(66., 190.)]
        )
        
        normal_noise = pyro.sample('intensity_noise_normal', Normal(0, 1))
        gamma_noise = pyro.sample('intensity_noise_gamma', Gamma(10, 5))
        intensity_noise = scale * normal_noise + 2 * gamma_noise - 4
        intensity = intensity_transform(intensity_noise)
        
        noise = pyro.sample('thickness_noise', Normal(0, 1))
        _i = (intensity - 66) / 190
        thickness = (-torch.log((1 / _i) - 1) + 5 - scale * noise) / 2
        
        # image_class_probs = [
        #     0.0987, 0.1124, 0.0993, 0.1022, 0.0974,
        #     0.0904, 0.0986, 0.1044, 0.0975, 0.0992
        # ]
        # labels = pyro.sample('labels', Categorical(torch.Tensor(image_class_probs)))
    
    return intensity, thickness  # , labels


def gen_dataset(args, train=True):
    pyro.clear_param_store()
    images_, labels, _ = load_morphomnist_like(args.data_dir, train=train)

    if args.digit_class is not None:
        mask = (labels == args.digit_class)
        images_ = images_[mask]
        labels = labels[mask]

    images = np.zeros_like(images_)

    n_samples = len(images)
    with torch.no_grad():
        intensity, thickness = model(n_samples, scale=args.scale, invert=args.invert)

    metrics = pd.DataFrame(data={'thickness': thickness, 'intensity': intensity})

    for n, (thickness, intensity) in enumerate(tqdm(zip(thickness, intensity), total=n_samples)):
        morph = ImageMorphology(images_[n], scale=16)
        tmp_img = morph.downscale(np.float32(SetThickness(thickness)(morph)))

        avg_intensity = get_intensity(tmp_img)

        mult = intensity.numpy() / avg_intensity
        tmp_img = np.clip(tmp_img * mult, 0, 255)

        images[n] = tmp_img

    # TODO: do we want to save the sampled or the measured metrics?

    save_morphomnist_like(images, labels, metrics, args.out_dir, train=train)

def gen_dataset_thickness_intervention(args, train=False, thickness_sf=1, intensity_sf=1):
    images_, labels, metrics = load_morphomnist_like(args.data_dir, train=train)
    n_samples = 10  # len(_images)
    thickness_ = list(metrics.thickness.values * thickness_sf)
    intensity_ = list(metrics.intensity.values * intensity_sf)

    metrics = pd.DataFrame(data={'thickness': thickness, 'intensity': intensity})

    images = np.zeros_like(images_)

    for n, (thickness, intensity) in enumerate(tqdm(zip(thickness_, intensity_), total=n_samples)):
        morph = ImageMorphology(images_[n], scale=16)
        tmp_img = morph.downscale(np.float32(SetThickness(thickness)(morph)))

        avg_intensity = get_intensity(tmp_img)
        mult = intensity / avg_intensity
        tmp_img = np.clip(tmp_img * mult, 0, 255)
        images[n] = tmp_img
    
    save_morphomnist_like(images, labels, metrics, f'{args.out_dir}/t_{thickness_sf}/i_{intensity_sf}', train=train)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/vol/biomedic/users/dc315/mnist/original/', help="Path to MNIST (default: %(default)s)")
    parser.add_argument('-o', '--out-dir', type=str, help="Path to store new dataset")
    parser.add_argument('-d', '--digit-class', type=int, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], help="digit class to select")
    parser.add_argument('-s', '--scale', type=float, default=0.5, help="scale of logit normal")
    parser.add_argument('-i', '--invert', default=False, action='store_true', help="inverses correlation")

    args = parser.parse_args()

    print(f'Generating data for:\n {args.__dict__}')

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'args.txt'), 'w') as f:
        print(f'Generated data for:\n {args.__dict__}', file=f)

    print('Generating Training Set')
    print('#######################')
    gen_dataset(args, True)

    print('Generating Test Set')
    print('###################')
    gen_dataset(args, False)
