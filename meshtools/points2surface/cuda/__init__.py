import torch
if torch.cuda.is_available():
	from . import jit