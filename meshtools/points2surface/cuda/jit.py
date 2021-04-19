from torch.utils.cpp_extension import load
import os

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

dir_path = os.path.dirname(os.path.realpath(__file__))
with cd(dir_path):
    triangle_distance_cuda = load(
        'triangle_distance_cuda', ['triangle_distance_cuda.cpp', 'triangle_distance_cuda_kernel.cu'], verbose=True)
    #help(triangle_distance_cuda)