import numpy as np
import torch


def get_2d_sincos_pos_embed(embed_dim, grid_size, n_cls_token=0):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w is first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])

    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed(embed_dim // 2, grid[1])  # (H*W, D/2)

    pos_embed = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)

    if n_cls_token>0:
        pos_embed = np.concatenate([np.zeros([n_cls_token, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_1d_sincos_pos_embed(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega

    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)

    emb_sin = np.sin(out)
    emb_cos = np.cos(out)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # shape: [M,D]
    return emb

# function to interpolate position embeddings to handle different resolutions
def interpolate_pos_embed(cfg, model, checkpoint_model):
    if 'dec_pos_embed' in checkpoint_model:
        if cfg.train.classtoken:
            # requires very specific handcrafted setup to carefully transfer the stage one pre-trained MAE model to the cls_token version
            # first, transfer over the 2nd view positional embedding:
            pos_embed_checkpoint = checkpoint_model['dec_pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = (cfg.augmentation.img_res // cfg.masking.patch_size) ** 2
            orig_size = int(pos_embed_checkpoint.shape[-2] ** 0.5)
            new_size = int(num_patches ** 0.5)
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                pos_tokens = pos_embed_checkpoint
                pos_tokens = pos_tokens.reshape(orig_size, orig_size, embedding_size).permute(2, 0, 1).unsqueeze(0)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2).squeeze(0)
                checkpoint_model['dec_pos_embed'] = pos_tokens
                # add new dict item to checkpoint model
                checkpoint_model['dec_pos_embed_cls'] = torch.cat((model.dec_pos_embed_cls[:1,:], pos_tokens), dim=0)
            else:
                checkpoint_model['dec_pos_embed_cls'] = torch.cat((model.dec_pos_embed_cls[:1,:], checkpoint_model['dec_pos_embed']), dim=0)
        else:
            pos_embed_checkpoint = checkpoint_model['dec_pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = (cfg.augmentation.img_res // cfg.masking.patch_size) ** 2
            # num_extra_tokens_ckpt = checkpoint_model['dec_pos_embed'].shape[-2] - 256 # hacky: pre-training always 224x224
            num_extra_tokens = model.dec_pos_embed.shape[-2] - num_patches
            # height (== width) for the checkpoint position embedding
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int(num_patches ** 0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[num_extra_tokens:]
                pos_tokens = pos_tokens.reshape(orig_size, orig_size, embedding_size).permute(2, 0, 1).unsqueeze(0)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2).squeeze(0)
                # edge case: adding class token only during fine-tuning
                if num_extra_tokens == 0 and num_extra_tokens > 0:
                    new_pos_embed = torch.cat((model.dec_pos_embed[:1,:], pos_tokens), dim=0)
                else:
                    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=0)

                checkpoint_model['dec_pos_embed'] = new_pos_embed