# Copyright 2024-present CSIRO.
# Licensed under CSIRO BSD-3-Clause-Clear (Non-Commercial)
#


import os
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

# os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4"

import copy
import argparse
import torch
from torch.optim import lr_scheduler
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm

from pairvpr.configs import stagetwo_default_config
from pairvpr.models.pairvpr import PairVPRNet
from pairvpr.models.tools.pos_embed import interpolate_pos_embed
from pairvpr.training.losses import VPRLoss
from pairvpr.utilities.misc import fix_random_seeds
from pairvpr.training.validation import validation

from pairvpr.datasets.gsvcities_dataset import load_train_dataset as gsv_loaddataset
from pairvpr.datasets.gsvcities_dataset import load_val_dataset


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("Pair-VPR training stage two", add_help=add_help)
    parser.add_argument("--config-file-finetuned", "--config_file_finetuned", 
                        default=os.path.join(root_dir, 'pairvpr', 'configs', 'stagetwo_default_config.yaml'), type=str, metavar="FILE",
                        help="path to config file")
    parser.add_argument("--dsetroot", default="", type=str, required=True,
                        help="Root dir where all datasets are saved to (both training and inference)")
    parser.add_argument("--pretrained_ckpt", type=str, default=None,
                        help="path to pretrained network to init with")
    parser.add_argument("--usewandb", action='store_true', 
                        help='Use wandb logging')
    parser.add_argument("--output-dir", "--output_dir", default="", type=str, required=True,
                        help="Output directory to save logs and checkpoints")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help='Can use to modify a config parameter via an argument parse')

    return parser


def get_cfg_from_args_stagetwo(args):
    args.output_dir = os.path.abspath(args.output_dir)
    args.opts += [f"train.output_dir={args.output_dir}"]
    default_cfg = OmegaConf.create(stagetwo_default_config)
    cfg = OmegaConf.load(args.config_file_finetuned)
    cfg = OmegaConf.merge(default_cfg, cfg, OmegaConf.from_cli(args.opts))
    return cfg


def main(args):
    assert torch.cuda.is_available(), "Pair-VPR requires at least two GPUs to be available during training."
    assert (torch.cuda.device_count() > 1), "Pair-VPR second stage training requires at least two GPUs."

    cfg = get_cfg_from_args_stagetwo(args)
    os.makedirs(args.output_dir, exist_ok=True)

    fix_random_seeds(int(0))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load pretrained_ckpt from stage one training
    if args.pretrained_ckpt is not None:
        pretrained_ckpt = torch.load(args.pretrained_ckpt)

    model = PairVPRNet(cfg).to(torch.device("cuda"))
    
    if args.pretrained_ckpt is not None:
        interpolate_pos_embed(cfg, model, pretrained_ckpt['model']) # interpolate the decoder pos embed for different res
        model.load_state_dict(pretrained_ckpt['model'], strict=False)

    model.to(torch.device("cuda"))

    # dataparallel is important to use, so that the contrastive loss with online mining can be calculated across all devices
    multinet = torch.nn.DataParallel(model, device_ids=list(range(cfg.train.num_gpus)))

    optimizer = torch.optim.AdamW(
        multinet.parameters(),
        lr=cfg.optim.base_lr,
        weight_decay=cfg.optim.weight_decay
    )

    scheduler = lr_scheduler.LinearLR(
        optimizer,
        start_factor=1,
        end_factor=0.2,
        total_iters=cfg.optim.schedulemaxiters
    )

    scaler = torch.amp.GradScaler("cuda")

    lossobj = VPRLoss() # miners and global descriptor criteria

    paircriterion = torch.nn.BCEWithLogitsLoss()

    if args.usewandb:
        myrun = wandb.init(
            project="pairvpr_stagetwo",
            config={
                "learning_rate": cfg.optim.base_lr,
                "epochs": cfg.optim.epochs,
                "image_size": cfg.augmentation.img_res,
                "batch_size": cfg.train.batch_size_per_gpu,
                "numlayerstrain": cfg.encoder.num_trainable_blocks
            },
        )

    GSVdataset = gsv_loaddataset(cfg, args.dsetroot)

    batch_size = cfg.train.batch_size_per_gpu * cfg.train.num_gpus

    GSVdataloader = torch.utils.data.DataLoader(
        GSVdataset,
        batch_size=batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=False, # SALAD and GSV-Cities has this set to false. False means a batch always contains images from the same city.
        pin_memory=True,
        drop_last=False,
    )

    val_set_names = ['msls_val']
    valdatasets = load_val_dataset(cfg, dsetroot=args.dsetroot, src_root_dir=root_dir, val_set_names=val_set_names)

    valid_loader_config = {
        'batch_size': 40,
        'num_workers': 5,
        'drop_last': False,
        'pin_memory': True,
        'shuffle': False}
    val_dataloaders = []
    for val_dataset in valdatasets:
        val_dataloaders.append(torch.utils.data.DataLoader(
            dataset=val_dataset, **valid_loader_config))

    device = torch.device("cuda")

    for epoch in range(cfg.optim.epochs):
        multinet.train(True)
        batch_acc_all = []
        for iteration, batch in tqdm(enumerate(GSVdataloader)):
            places, labels = batch

            # Note that GSVCities yields places (each containing N images)
            # which means the dataloader will return a batch containing BS places
            BS, N, ch, h, w = places.shape

            # reshape places and labels
            images = places.to(device, non_blocking=True).view(BS * N, ch, h, w)
            labels = labels.view(-1)

            # Feed forward the batch to the model
            with torch.amp.autocast("cuda"):
                feats, descriptors = multinet(images, None, "global")

                if torch.isnan(descriptors).any():
                    raise ValueError('NaNs in descriptors')

                # run descriptors through miners
                mined_multisim, mined_batchhard = lossobj.mining(descriptors, labels)

                loss1, batch_acc = lossobj.loss_function_global(descriptors, labels, mined_multisim)

                # can process batched for efficiency
                dp = multinet(feats[mined_batchhard[0]], feats[mined_batchhard[1]], "pairvpr") # positives
                dn = multinet(feats[mined_batchhard[0]], feats[mined_batchhard[2]], "pairvpr") # strong negatives

                dp = dp.flatten()
                dn = dn.flatten()

                targets = torch.cat((torch.ones(dp.shape[0]), torch.zeros(dn.shape[0]))).to(device)

                loss2 = paircriterion(torch.cat((dp, dn)), targets)

                loss = loss1 + (cfg.optim.locallossweight * loss2) # simultaneously optimize both global and two-stage losses

            batch_acc_all.append(batch_acc)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            torch.cuda.synchronize()

            print(loss.item())

            if args.usewandb and (iteration % 20 == 0) and (iteration > 0):
                wandb.log({"loss": loss.item(), "b_acc": sum(batch_acc_all) / len(batch_acc_all),
                           "loss_global": loss1.item(), "loss_fine": loss2.item()})

        net = copy.deepcopy(multinet.module)
        net = net.eval()
        net = net.to(torch.device("cuda:1"))  # cuda:1 is important, because in DP training, GPU 0 has a slightly higher GPU memory usage.
        val_results = validation(args, val_dataloaders, net, val_set_names, valdatasets, device=torch.device("cuda:1"))

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        checkpoint_filename = os.path.join(args.output_dir,
                                           'locas' + '_' + str(epoch) + '.pth')

        # as second stage training is fast, we currently don't support resuming from a checkpoint.
        # here we only save the state_dict, for downstream VPR tasks
        torch.save(multinet.module.state_dict(), checkpoint_filename)
        print('epoch done')
    print('done')


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)