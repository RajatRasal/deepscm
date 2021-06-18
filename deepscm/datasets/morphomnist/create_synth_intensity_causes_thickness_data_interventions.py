import numpy as np
import os
import pandas as pd
import pyro
import torch

from pyro.distributions import Normal
from tqdm import tqdm

from deepscm.datasets.morphomnist import load_morphomnist_like, save_morphomnist_like
from deepscm.datasets.morphomnist.transforms import SetThickness, ImageMorphology
from deepscm.datasets.morphomnist.create_synth_intensity_causes_thickness_data import get_intensity 


def gen_dataset_thickness_intervention(args, scale=0.5):
    pyro.clear_param_store()

    images_, labels, metrics = load_morphomnist_like(args.data_dir, train=args.train)
    thickness_af = args.thickness_af
    intensity_af = args.intensity_af
    n_samples = len(images_)
    thickness_ = list(metrics.thickness.values + thickness_af)
    intensity_ = list(metrics.intensity.values + intensity_af)

    metrics = pd.DataFrame(data={'thickness': thickness_, 'intensity': intensity_})

    images = np.zeros_like(images_)

    for n, (thickness, intensity) in enumerate(tqdm(zip(thickness_, intensity_), total=n_samples)):
        morph = ImageMorphology(images_[n], scale=16)

        # do(i = x)
        if intensity_af != 0 and thickness_af == 0:
            # const to prevent errors or underflow
            _i = max((1 / ((intensity - 66) / 180)) - 1, 1e-2)
            noises = torch.stack([pyro.sample('thickness_noise', Normal(0, 1)) for i in range(args.particles)])
            thickness = ((-np.log(_i) + 5 - scale * noises.numpy()) / 2).mean()
            metrics.thickness[n] = thickness
            # print(thickness)

        tmp_img = morph.downscale(np.float32(SetThickness(thickness)(morph)))

        avg_intensity = get_intensity(tmp_img)
        mult = intensity / avg_intensity
        tmp_img = np.clip(tmp_img * mult, 0, 255)
        images[n] = tmp_img

    save_morphomnist_like(images, labels, metrics, args.out_dir, train=args.train)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='/vol/biomedic/users/dc315/mnist/original/', help="Path to MNIST (default: %(default)s)")
    parser.add_argument('-o', '--out-dir', type=str, help="Path to store new dataset")
    parser.add_argument('--train', type=bool, default=False, help="Train or test set")
    parser.add_argument('--thickness-af', type=float, default=0, help="Thickness scale factor")
    parser.add_argument('--intensity-af', type=float, default=0, help="Intensity additive factor")
    parser.add_argument('--particles', type=int, default=10, help="Particles")

    args = parser.parse_args()

    print(f'Generating data for:\n {args.__dict__}')

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'args.txt'), 'w') as f:
        print(f'Generated data for:\n {args.__dict__}', file=f)

    gen_dataset_thickness_intervention(args)

    # print('Generating Training Set')
    # print('#######################')
    # gen_dataset(args, True)

    # print('Generating Test Set')
    # print('###################')
    # gen_dataset(args, False)
