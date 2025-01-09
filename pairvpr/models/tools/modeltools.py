import torch
import torch.nn as nn
from itertools import repeat
import collections.abc


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    # Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class RandomMask(nn.Module):
    def __init__(self, patch_size, mask_ratio):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

    def __call__(self, img):
        # using dynamic masking to handle any image size during training
        # image has B x 3 x img_size x img_size
        num_patches = (img.size(2) // self.patch_size) * (img.size(3) // self.patch_size)
        num_mask = int(self.mask_ratio * num_patches)
        noise = torch.rand(img.size(0), num_patches, device=img.device)
        argsort = torch.argsort(noise, dim=1)
        return argsort < num_mask
