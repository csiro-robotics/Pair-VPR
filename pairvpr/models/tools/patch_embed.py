import torch
import torch.nn as nn

from pairvpr.models.tools.modeltools import to_2tuple


# patch embedding
class PositionGetter(object):
    # a class to return the positions of patches, with a cache for speed

    def __init__(self):
        self.cache_positions = {}

    def __call__(self, b, h, w, device):
        if not (h, w) in self.cache_positions:
            x = torch.arange(w, device=device)
            y = torch.arange(h, device=device)
            self.cache_positions[h, w] = torch.cartesian_prod(y, x)  # (h, w, 2)
        pos = self.cache_positions[h, w].view(1, h * w, 2).expand(b, -1, 2).clone()
        return pos


class PatchEmbed(nn.Module):
    # based off timm.models.layers.patch_embed.PatchEmbed, licensed under the Apache License 2.o
    # link to Timm: https://github.com/huggingface/pytorch-image-models

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        self.position_getter = PositionGetter()

    def forward(self, x):
        B, C, H, W = x.shape
        torch._assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        torch._assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        pos = self.position_getter(B, x.size(2), x.size(3), x.device)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x, pos

    def _init_weights(self):
        w = self.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))