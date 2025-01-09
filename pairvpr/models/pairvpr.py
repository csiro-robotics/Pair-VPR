# Copyright 2024-present CSIRO.
# Licensed under CSIRO BSD-3-Clause-Clear (Non-Commercial)
#


import torch
from torch import nn
import torch.nn.functional as F
from functools import partial

from pairvpr.models.tools.pos_embed import get_2d_sincos_pos_embed
from pairvpr.models.dinov2 import DINOv2
from pairvpr.models.tools.blocks import DecoderBlock
from pairvpr.models.tools.modeltools import RandomMask


DINOV2_ARCHS = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
    'dinov2_vits14_reg': 384,
    'dinov2_vitb14_reg': 768,
    'dinov2_vitl14_reg': 1024,
    'dinov2_vitg14_reg': 1536,
}

class PairVPRNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.encoder = DINOv2(**cfg.encoder)

        self.enc_embed_dim = self.encoder.model.num_features
        self.enc_depth = self.encoder.model.n_blocks

        self._set_mask_generator(cfg.masking.patch_size, cfg.masking.mask_ratio)

        self.img_res = cfg.augmentation.img_res
        self.num_patches = (self.img_res // cfg.masking.patch_size) ** 2

        # we only add a class token to the decoder during stage two of training:
        if cfg.train.classtoken:
            self.decoder_clstoken = nn.Parameter(torch.zeros(1, 1, cfg.decoder.dec_embed_dim))
            # position embeddings change when we add a class token
            dec_pos_embed_cls = get_2d_sincos_pos_embed(cfg.decoder.dec_embed_dim, int(self.num_patches ** .5), n_cls_token=1)
            self.register_buffer('dec_pos_embed_cls', torch.from_numpy(dec_pos_embed_cls).float())
        else:
            self.decoder_clstoken = None

        dec_pos_embed = get_2d_sincos_pos_embed(cfg.decoder.dec_embed_dim, int(self.num_patches ** .5), n_cls_token=0)
        self.register_buffer('dec_pos_embed', torch.from_numpy(dec_pos_embed).float())

        self._set_mask_token(cfg.decoder.dec_embed_dim)
        embed_dim = DINOV2_ARCHS[cfg.encoder.model_name]

        # decoder
        self._set_decoder(embed_dim, **cfg.decoder)

        # prediction head
        self._set_prediction_head(cfg.decoder.dec_embed_dim, cfg.masking.patch_size)

        # setup new custom vpr modules
        self.globaldim = cfg.globaldesc.dim
        self._set_vpr_pair_predictors(cfg.decoder.dec_embed_dim)
        self._set_vpr_global_predictor()

        # WARNING THIS LINE OVERWRITES AND DELETES ALL DINO PRE-TRAINED WEIGHTS
        self._initialize_weights()

        # in random mode, we remove all dino weights in linear layers and layernorm
        # (but keep helpers like mask tokens, class tokens, etc)
        # this should effectively be the same as training from scratch
        if not cfg.train.random:
            self.encoder = DINOv2(**cfg.encoder)
            self.enc_embed_dim = self.encoder.model.num_features
            self.enc_depth = self.encoder.model.n_blocks

    def _encode_image(self, image, do_mask=True):
        # image has B x 3 x img_size x img_size
        if do_mask:
            masks = self.mask_generator(image)
        else:
            masks = None
        x = self.encoder(image, mask=masks)
        # x is both features and class token
        return x, masks
    
    def _decoder(self, feat1, feat2):
        # encoder to decoder layer
        f1 = self.decoder_embed(feat1)
        f2 = self.decoder_embed(feat2)

        if self.decoder_clstoken is not None:
            f1 = torch.cat((self.decoder_clstoken.expand(f1.shape[0], -1, -1), f1), dim=1)

            f1 = f1 + self.dec_pos_embed_cls
            f2 = f2 + self.dec_pos_embed
        else:
            f1 = f1 + self.dec_pos_embed
            f2 = f2 + self.dec_pos_embed

        out1 = f1
        out2 = f2
        for iter, blk in enumerate(self.dec_blocks):
            out1, out2 = blk(out1, out2)
        
        out1 = self.dec_norm(out1)
        return out1
    
    def forward(self, img1, img2, mode="default"):
        if mode == "pairvpr":
            # generate pair similarity score
            decfeat = self._decoder(img1, img2) # not actually images here, just features

            cls_token = decfeat[:, 0]
            diff = self.classvprmodule(cls_token)

            return diff  # where 0 is very similar, 1 is very different
        elif img2 == None and mode == "global":
            # generate global descriptor and dense features      
            feat, _ = self._encode_image(img1, do_mask=False)
            classtok = feat[1]
            map = feat[0]
            # process classtok through layers and norm to create a global descriptor (512 dim)
            classtok = self.globalizer(classtok)
            classtok = nn.functional.normalize(classtok, p=2, dim=-1)
            return map, classtok
        else:
            # stage one training
            # encoder of the masked first image
            feat1, mask1 = self._encode_image(img1, do_mask=True)
            # encoder of the second image
            feat2, _ = self._encode_image(img2, do_mask=False)

            # separate out class tokens - un-used in our work, but future work could use it
            classtok1 = feat1[1] # even though img1 is masked, the classtokens should still output similar embeddings
            map1 = feat1[0]
            classtok2 = feat2[1]
            map2 = feat2[0]

            # decoder
            decfeat = self._decoder(map1, map2)
            # prediction head
            out = self.prediction_head(decfeat)
            # get target
            target = self.patchify(img1)

            return out, mask1, target
        
    def _set_mask_generator(self, patch_size, mask_ratio):
        self.mask_generator = RandomMask(patch_size, mask_ratio)

    def _set_mask_token(self, dec_embed_dim):
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_embed_dim))

    def _set_prediction_head(self, dec_embed_dim, patch_size):
        self.prediction_head = nn.Linear(dec_embed_dim, patch_size**2 * 3, bias=True)

    def _set_vpr_pair_predictors(self, dec_embed_dim):
        self.classvprmodule = nn.Sequential(
            nn.Linear(dec_embed_dim, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
    
    def _set_vpr_global_predictor(self):
        self.globalizer = nn.Sequential(
            nn.Linear(self.enc_embed_dim, self.globaldim),
        )
    
    def _set_decoder(self, enc_embed_dim=512, dec_embed_dim=512, dec_num_heads=12, dec_depth=8, mlp_ratio=4,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_im2_in_dec=True):
        # Inspired by CrossBlock from Croco paper
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        # transfer from encoder to decoder
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        # transformer for the decoder
        self.dec_blocks = nn.ModuleList([
            DecoderBlock(dec_embed_dim, dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                        norm_mem=norm_im2_in_dec)
            for i in range(dec_depth)])
        # final norm layer
        self.dec_norm = norm_layer(dec_embed_dim)

    def _initialize_weights(self):
        # mask and class token init
        if self.mask_token is not None: torch.nn.init.normal_(self.mask_token, std=.02)
        if self.decoder_clstoken is not None: torch.nn.init.normal_(self.decoder_clstoken, std=1e-6)
        # linears and layer norms
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # using xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (B, 3, H, W)
        x: (B, L, patch_size**2 *3)
        """
        p = self.cfg.masking.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))

        return x

    def unpatchify(self, x, channels=3):
        """
        x: (N, L, patch_size**2 *channels)
        imgs: (N, 3, H, W)
        """
        patch_size = self.cfg.masking.patch_size
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, patch_size, patch_size, channels))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], channels, h * patch_size, h * patch_size))

        return imgs
    
