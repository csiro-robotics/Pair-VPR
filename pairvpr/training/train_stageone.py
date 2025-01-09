# Copyright 2024-present CSIRO.
# Licensed under CSIRO BSD-3-Clause-Clear (Non-Commercial)
#


import os
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

# os.environ["CUDA_VISIBLE_DEVICES"]="1"

import argparse
import torch.distributed
import torch
import wandb
from tqdm import tqdm

from pairvpr.utilities.config import setup
from pairvpr.utilities import misc
import pairvpr.utilities.distributed as distributed
from pairvpr.utilities.misc import NativeScalerWithGradNormCount as NativeScaler
from pairvpr.utilities.param_groups import get_params_groups_with_decay, prepare_param_groups
from pairvpr.training.losses import MaskedMSELoss
from pairvpr.datasets.pairs_dataset import PairsDataset

from pairvpr.models.pairvpr import PairVPRNet

torch.backends.cuda.matmul.allow_tf32 = True  # PyTorch 1.12 sets this to False by default


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("Pair-VPR training stage one", add_help=add_help)
    parser.add_argument("--config-file", "--config_file", default=os.path.join(root_dir, 'pairvpr', 'configs', 'stageone_default_config.yaml'),
                        metavar="FILE", help="path to config file")
    parser.add_argument("--resume_train", type=str, default=None,
                        help="path to checkpoint to resume, e.g. runs/.../last_checkpoint.pth")
    parser.add_argument("--output-dir", "--output_dir", default="", type=str, required=True, 
                        help="Output directory to save logs and checkpoints, e.g. runs/...")
    parser.add_argument("-d", "--datasets_used", nargs='+', required=True, choices=["sf","gsv","gldv2"],
                        help="list of datasets used during training. Choices: sf, gsv, gldv2")
    parser.add_argument("--dsetroot", default="", type=str, required=True,
                        help="Root dir where all datasets are saved to (both training and inference)")
    parser.add_argument("--usewandb", action='store_true', 
                        help='Use wandb logging')
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help='Can use to modify a config parameter via an argument parse')

    return parser


def build_schedulers(cfg, epoch_length: int):
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * epoch_length,
        warmup_iters=cfg.optim["warmup_epochs"] * epoch_length,
        start_warmup_value=0,
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * epoch_length,
    )
    lr_schedule = misc.CosineScheduler(**lr)
    wd_schedule = misc.CosineScheduler(**wd)

    return (lr_schedule, wd_schedule)


def apply_optim_scheduler(optimizer, lr, wd):
    for param_group in optimizer.param_groups:
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = lr * lr_multiplier


def train_one_epoch(cfg, args, model, optimizer, lr_schedule, wd_schedule, loss_scaler,
                    criterion, pairsdataloader, global_iter: int, global_rank: int):

    epoch_loss = torch.tensor(0.0)

    for iteration, (images1, images2, dsetsource) in tqdm(enumerate(pairsdataloader)):
        images1, images2 = images1.to('cuda', non_blocking=True), images2.to('cuda', non_blocking=True)

        lr = lr_schedule[global_iter]
        wd = wd_schedule[global_iter]
        apply_optim_scheduler(optimizer, lr, wd)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=bool(cfg.optim.amp)):
            out, mask, target = model(images1, images2)

            if len(args.datasets_used) == 3:
                loss, lossgld, lossgsv, losssf = criterion(out, mask, target, dsetsource)
                # note: dset specific losses are detached (no grads), as they are currently only used for logging
            else:
                loss = criterion(out, mask, target)

            if global_rank == 0:
                loss_value = loss.item()
                if iteration % 20 == 0:
                    print('Loss: ', str(loss_value))
                if args.usewandb:
                    if len(args.datasets_used) == 3:
                        lv_gld = lossgld.item()
                        lv_gsv = lossgsv.item()
                        lv_sfxl = losssf.item()
                        wandb.log({"loss": loss_value, "loss_gld": lv_gld, "loss_gsv": lv_gsv, "loss_sfxl": lv_sfxl})
                    else:
                        wandb.log({"loss": loss_value})
                        # note we haven't tried adjusting mask ratios per dataset, this would be a good future work idea
                epoch_loss += loss_value

            loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=True)
            torch.cuda.synchronize()
            global_iter += 1
            del loss

    return epoch_loss, global_iter


def main(args):
    assert torch.cuda.is_available(), "Pair-VPR requires a GPU to be available during training."

    cfg = setup(args)

    model = PairVPRNet(cfg).to(torch.device("cuda"))

    global_rank = distributed.get_global_rank()
    world_size = distributed.get_global_size()
    global_batchsize = cfg.train.batch_size_per_gpu * world_size

    torch.backends.cudnn.benchmark = True

    model_without_ddp = model
    if distributed.is_enabled():
        # device_id = rank % torch.cuda.device_count() # alternate method
        deviceid = distributed.get_local_rank()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[deviceid], find_unused_parameters=True, static_graph=True)
        model_without_ddp = model.module

    param_groups = get_params_groups_with_decay(model_without_ddp,
                                                lr_decay_rate=cfg.optim.layerwise_decay,
                                                patch_embed_lr_mult=cfg.optim.patch_embed_lr_mult)

    optimizer = torch.optim.AdamW(prepare_param_groups(param_groups),
                                  betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))

    loss_scaler = NativeScaler()

    criterion = MaskedMSELoss(norm_pix_loss=bool(cfg.train.norm_pix_loss)) # mask reconstruction task

    if args.resume_train is not None:  # manual resume
        last_ckpt_fname = args.resume_train # resume train requires full path or relative path to checkpoint .pth file
    else: # auto resume
        last_ckpt_fname = os.path.join(args.output_dir, f'checkpoint-last.pth')
    args.resume = last_ckpt_fname if os.path.isfile(last_ckpt_fname) else None

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if global_rank == 0:
        if args.usewandb:
            myrun = wandb.init(
                project="pairvpr_stageone",
                config={
                    "learning_rate": cfg.optim.base_lr,
                    "epochs": cfg.optim.epochs,
                    "batch_size": cfg.train.batch_size_per_gpu,
                    "mask_ratio": cfg.masking.mask_ratio,
                    "img_res": cfg.augmentation.img_res,
                    "num_trainable_blocks": cfg.encoder.num_trainable_blocks
                },
            )

    dataset_location_gsv = os.path.join(args.dsetroot, cfg.dataset_locations.gsv)
    dataset_location_gldv2 = os.path.join(args.dsetroot, cfg.dataset_locations.gldv2)
    dataset_location_sfxl = os.path.join(args.dsetroot, cfg.dataset_locations.sfxl)

    pairsdataset = PairsDataset(cfg, root_dir, args.datasets_used,
        dataset_location_sfxl, dataset_location_gsv, dataset_location_gldv2,
        M=cfg.sfxl.M, N=1, focal_dist=cfg.sfxl.focal_dist, focal_dist_max=cfg.sfxl.focal_dist_max,
        current_group=0, min_images_per_class=cfg.sfxl.min_images_per_class,
        dynamicmode=cfg.sfxl.dynamic_mode, angcheck=cfg.sfxl.angle_check, rank=global_rank)

    if world_size > 1:
        torch.distributed.barrier() # wait until all threads have loaded the dataset

    print(f"There are {len(pairsdataset)} classes for the first group, " +
                 f"each epoch has {len(pairsdataset) / global_batchsize} iterations " +
                 f"with batch_size {global_batchsize}.")

    # epoch_length is number of samples in dataset divided by total batch size across all devices
    epoch_length = int(len(pairsdataset) // global_batchsize)
    (lr_schedule, wd_schedule) = build_schedulers(cfg, epoch_length)

    if world_size > 1:
        sampler_train = torch.utils.data.DistributedSampler(
            pairsdataset, num_replicas=world_size, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(pairsdataset)

    pairsdataloader = torch.utils.data.DataLoader(
        pairsdataset, sampler=sampler_train,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    if global_rank == 0:
        print("Start training ...")

    number_of_iterations = len(pairsdataloader)
    print('number of iterations per epoch: ', str(number_of_iterations))

    model.train(True)

    global_iter = args.start_iter

    for epoch_num in range(args.start_epoch, cfg.optim.epochs):
        if world_size > 1:
            pairsdataloader.sampler.set_epoch(epoch_num)

        epoch_loss, global_iter = train_one_epoch(cfg, args, model, optimizer, lr_schedule, wd_schedule, loss_scaler,
                        criterion, pairsdataloader, global_iter, global_rank)

        if global_rank == 0:
            epoch_loss /= number_of_iterations
            if args.usewandb:
                wandb.log({"epoch_loss": epoch_loss})

        if args.output_dir and epoch_num % cfg.train.save_freq == 0:
            misc.save_model(
                args=args, global_iter=global_iter, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch_num, fname='last')

        if args.output_dir and (epoch_num % cfg.train.keep_freq == 0 or epoch_num + 1 == cfg.optim.epochs) and (
                epoch_num > 0 or cfg.optim.epochs == 1):
            misc.save_model(
                args=args, global_iter=global_iter, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch_num)

        print('epoch complete')

    print('training complete')



if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)