# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
# ---------------
# This file has been modified from https://github.com/facebookresearch/dinov2
# You may obtain a copy of the License for this file at http://www.apache.org/licenses/LICENSE-2.0


from collections import defaultdict


def get_vit_lr_decay_rate(name, lr_decay_rate=1.0, enc_depth=12, dec_depth=12):
    """
    Calculate lr decay rate for different ViT blocks.
    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_layers (int): number of ViT blocks.
    Returns:
        lr decay rate for the given parameter.
    """
    if name in ("cls_token", "mask_token", "pos_embed", "global_tokens"):
        layer_id = 0
    elif name.startswith("patch_embed"):
        layer_id = 0
    elif name.startswith("encoder.model"):
        if (
            ".pos_embed" in name
            or ".patch_embed" in name
            or ".mask_token" in name
            or ".cls_token" in name
            or ".register_tokens" in name
        ):
            layer_id = 0
        elif (
            "pos_embed" in name
            or "patch_embed" in name
            or "mask_token" in name
            or "cls_token" in name
            or "register_tokens" in name
        ):
            layer_id = 0
        elif ".blocks." in name and ".residual." not in name:
            layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1
        elif "blocks." in name and "residual." not in name:
            layer_id = int(name[name.find("blocks.") :].split(".")[1]) + 1
        else:
            layer_id = enc_depth
    elif name.startswith('decoder_embed'):  # part of the last black
        layer_id = enc_depth
    elif name.startswith('dec_blocks'):
        layer_id = enc_depth + int(name.split('.')[1]) + 1
    elif name.startswith('dec_norm'):  # part of the last block
        layer_id = enc_depth + dec_depth
    elif any(name.startswith(k) for k in ['head', 'prediction_head']):
        layer_id = enc_depth + dec_depth + 1
    else:
        layer_id = 0

    num_layers = enc_depth + dec_depth
    return lr_decay_rate ** (num_layers + 1 - layer_id)


def get_params_groups_with_decay(model, lr_decay_rate=1.0, patch_embed_lr_mult=1.0):
    all_param_groups = []

    enc_depth = model.enc_depth
    dec_depth = model.dec_depth if hasattr(model, 'dec_blocks') else 0

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        decay_rate = get_vit_lr_decay_rate(
            name, lr_decay_rate, enc_depth=enc_depth, dec_depth=dec_depth
        )
        d = {"params": param, "lr_multiplier": decay_rate, "wd_multiplier": 1.0, "name": name}

        if name.endswith(".bias") or "norm" in name or "gamma" in name or len(param.shape) == 1:
            d.update({"wd_multiplier": 0.0})

        if "patch_embed" in name:
            d.update({"lr_multiplier": d["lr_multiplier"] * patch_embed_lr_mult})

        all_param_groups.append(d)

    return all_param_groups


def fuse_params_groups(all_params_groups, keys=("lr_multiplier", "wd_multiplier")):
    fused_params_groups = defaultdict(lambda: {"params": []})
    for d in all_params_groups:
        identifier = ""
        for k in keys:
            identifier += k + str(d[k]) + "_"

        for k in keys:
            fused_params_groups[identifier][k] = d[k]
        fused_params_groups[identifier]["params"].append(d["params"])

    return fused_params_groups.values()


def prepare_param_groups(all_params_groups):
    fused_params_groups = fuse_params_groups(all_params_groups)
    for g in fused_params_groups:
        g["foreach"] = True
    return fused_params_groups