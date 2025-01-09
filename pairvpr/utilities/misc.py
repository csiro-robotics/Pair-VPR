# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
# ---------------
# This file has been modified from https://github.com/facebookresearch/dinov2
# You may obtain a copy of the License for this file at http://www.apache.org/licenses/LICENSE-2.0


import os
import random
from pathlib import Path
import subprocess
import numpy as np
import torch
from torch import nn
import math
import pairvpr.utilities.distributed as distributed



def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode("ascii").strip()

    sha = "N/A"
    diff = "clean"
    branch = "N/A"
    try:
        sha = _run(["git", "rev-parse", "HEAD"])
        subprocess.check_output(["git", "diff"], cwd=cwd)
        diff = _run(["git", "diff-index", "HEAD"])
        diff = "has uncommitted changes" if diff else "clean"
        branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


class CosineScheduler(object):
    def __init__(self, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, freeze_iters=0):
        super().__init__()
        self.final_value = final_value
        self.total_iters = total_iters

        freeze_schedule = np.zeros((freeze_iters))

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters - freeze_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((freeze_schedule, warmup_schedule, schedule))

        assert len(self.schedule) == self.total_iters

    def __getitem__(self, it):
        if it >= self.total_iters:
            return self.final_value
        else:
            return self.schedule[it]


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False



def is_main_process():
    return distributed.get_global_rank() == 0


def load_model(args, model_without_ddp, optimizer, loss_scaler=None):
    args.start_epoch = 0
    args.start_iter = 1
    best_so_far = None
    if args.resume is not None:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        print("Resume checkpoint %s" % args.resume)

        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
        args.start_epoch = checkpoint['epoch'] + 1
        args.start_iter = checkpoint['global_iter'] + 1
        optimizer.load_state_dict(checkpoint['optimizer'])

        if loss_scaler is not None:
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
        if 'best_so_far' in checkpoint:
            best_so_far = checkpoint['best_so_far']
            print(" & best_so_far={:g}".format(best_so_far))
        else:
            print("")
        print("With optim & sched, start_epoch={:d}".format(args.start_epoch), end='')
    return best_so_far


def save_model(args, epoch, global_iter, model_without_ddp, optimizer, loss_scaler=None, fname=None, best_so_far=None):
    output_dir = Path(args.output_dir)
    if fname is None: fname = str(epoch)
    checkpoint_path = output_dir / ('checkpoint-%s.pth' % fname)
    if loss_scaler is None:
        to_save = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'args': args,
            'epoch': epoch,
            'global_iter': global_iter,
        }
    else:
        to_save = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': loss_scaler.state_dict(),
            'args': args,
            'epoch': epoch,
            'global_iter': global_iter,
        }
    if best_so_far is not None: to_save['best_so_far'] = best_so_far
    print(f'>> Saving model to {checkpoint_path} ...')
    if is_main_process():
        torch.save(to_save, checkpoint_path)



class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self, enabled=True):
        self._scaler = torch.amp.GradScaler("cuda", enabled=enabled)

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_gradnorm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_gradnorm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == math.inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                                norm_type)
    return total_norm


