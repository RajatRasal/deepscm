from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='triangle_distance_cuda',
    ext_modules=[
        CUDAExtension('triangle_distance_cuda', [
            'triangle_distance_cuda.cpp',
            'triangle_distance_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })