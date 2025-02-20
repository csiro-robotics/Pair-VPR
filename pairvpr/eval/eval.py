# Copyright 2024-present CSIRO.
# Licensed under CSIRO BSD-3-Clause-Clear (Non-Commercial)
#


import os
import sys
from pathlib import Path
from typing import List

root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

import argparse
import faiss
import faiss.contrib.torch_utils
from prettytable import PrettyTable
from tqdm import tqdm

import numpy as np
import torch
from omegaconf import OmegaConf
import time

from pairvpr.models.pairvpr import PairVPRNet
from pairvpr.models.tools.pos_embed import interpolate_pos_embed
from pairvpr.configs import pairvpr_speed
import pairvpr.eval.get_datasets as dataset_getter


def get_cfg_from_args_eval(args):
    default_cfg = OmegaConf.create(pairvpr_speed)
    cfg = OmegaConf.load(args.config_file_eval)
    cfg = OmegaConf.merge(default_cfg, cfg)
    return cfg


VAL_DATASETS = ['tokyo247', 'MSLS_val', 'MSLS_test',
                'pitts30k_test', 'pitts250k_test', 'Nordland']


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("Pair-VPR inference", add_help=add_help)
    parser.add_argument("--config-file-eval", "--config_file_eval", type=str, default=None, metavar="FILE",
                        help="path to config file")
    parser.add_argument("--trained_ckpt", "--trained-ckpt", type=str, default=None,
                        help="path to finetuned network to eval")
    parser.add_argument("--dsetroot", default="", type=str, required=True,
                        help="Root dir where all datasets are saved to (both training and inference)")
    parser.add_argument(
        '--val_datasets',
        nargs='+',
        default=VAL_DATASETS,
        help='Validation datasets to use',
        choices=VAL_DATASETS,
    )

    return parser


def get_test_recalls(dataset_name: str, k_values: List,
                     predictions: np.array, reranked_predictions: np.array,
                     gt: np.array, val_dataset, refinetopcands: int = 50):
    if dataset_name == 'MSLS_test':
        savepath = 'results_' + dataset_name # edit to suit where you want to save MSLS challenge submission files
        os.makedirs(savepath, exist_ok=True)
        val_dataset.save_predictions(predictions[:, :refinetopcands], os.path.join(savepath, 'PairVPR-global' + '.csv'))
        val_dataset.save_predictions(reranked_predictions[:, :refinetopcands], os.path.join(savepath, 'PairVPR-refined' + '.csv'))
        return {}, {}
    
    # calculate recall_at_k
    correct_at_k = np.zeros(len(k_values))
    for q_idx, pred in enumerate(predictions):
        for i, n in enumerate(k_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[q_idx])):
                correct_at_k[i:] += 1
                break

    # calculating recall_at_k reranked
    correct_at_k_refined = np.zeros(len(k_values))
    for q_idx, pred in enumerate(reranked_predictions):
        for i, n in enumerate(k_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[q_idx])):
                correct_at_k_refined[i:] += 1
                break

    correct_at_k = correct_at_k / len(predictions)
    results_dict = {k: v for (k, v) in zip(k_values, correct_at_k)}

    correct_at_k_refined = correct_at_k_refined / len(predictions)
    results_dict_refined = {k: v for (k, v) in zip(k_values, correct_at_k_refined)}

    print()  # print a new line
    table = PrettyTable()
    table.field_names = ['K'] + [str(k) for k in k_values]
    table.add_row(['Recall@K'] + [f'{100 * v:.2f}' for v in correct_at_k])
    print(table.get_string(title=f"Performances on {dataset_name}"))

    print()  # print a new line
    table = PrettyTable()
    table.field_names = ['K'] + [str(k) for k in k_values]
    table.add_row(['Recall@K'] + [f'{100 * v:.2f}' for v in correct_at_k_refined])
    print(table.get_string(title=f"Reranked performances on {dataset_name}"))

    return results_dict, results_dict_refined


def test(cfg, val_dataloaders, model, val_set_names, val_datasets, 
         all_num_references, all_num_queries, all_ground_truth, device):
    recalls_global = []
    recalls_refined = []

    for dataloader_idx, (valdataloader, val_dataset, val_set_name,
                         num_references, num_queries, ground_truth) in \
        enumerate(zip(val_dataloaders, val_datasets, val_set_names,
                      all_num_references, all_num_queries, all_ground_truth)):

        # extract features
        feats = []
        densefeats = []
        extract_time_start = time.time()
        with torch.no_grad():
            for iteration, batch in enumerate(tqdm(valdataloader)):
                images, labels = batch
                map, descriptors = model(images.to(device), None, mode='global')
                if not cfg.eval.memoryeffmode:
                    densefeats.append(map.detach().cpu()) # 1.55MB/image for 322x322 images
                feats.append(descriptors.detach().cpu())

        extract_time_end = time.time()
        timeperimage = (extract_time_end - extract_time_start) / len(val_dataset)
        print('---PairVPR took this many seconds per query for extract: ', str(timeperimage))

        feats = torch.concat(feats, dim=0)
        if not cfg.eval.memoryeffmode:
            densefeats = torch.concat(densefeats, dim=0)
            total_memory = ((densefeats.numpy().size * densefeats.numpy().itemsize)/1024) # kB
            print('---PairVPR requires this many kB per image: ', str((total_memory/len(val_dataset))+2))

        r_list = feats[: num_references]
        q_list = feats[num_references:]

        # match features (global descriptor)
        embed_size = r_list.shape[1]
        faiss_index = faiss.IndexFlatL2(embed_size)
        match_time_start = time.time()

        # add references
        faiss_index.add(r_list)

        # search for queries in the index
        k_values=[1, 5, 10, 15, 20, 50, 100, 250, 500, 1000]
        _, predictions = faiss_index.search(q_list, max(k_values))

        match_time_end = time.time()
        timetomatch = match_time_end - match_time_start
        print('---PairVPR took this many sec/query to match: ', str(timetomatch/val_dataset.num_queries))

        if not cfg.eval.memoryeffmode:
            db_df = densefeats[: num_references]
            q_df = densefeats[num_references:]
    
        reranked_predictions = torch.zeros(predictions.shape[0], cfg.eval.refinetopcands, dtype=int)
        reranked_scores = torch.zeros(predictions.shape[0])

        # start re-ranking
        rerank_time_start = time.time()
        print('---Starting re-ranking')
        with torch.no_grad():
            for q_idx, pred in enumerate(tqdm(predictions)):
                pred = pred[:cfg.eval.refinetopcands]

                if not cfg.eval.memoryeffmode:
                    qfeats = q_df[q_idx].repeat(len(pred), 1, 1).to(device)
                    dbfeats = db_df[pred].to(device)
                else:
                    qimg, _ = val_dataset[len(r_list) + q_idx]
                    dbimgs = []
                    for p in pred: # [val_dataset[p][0] for p in pred]
                        dbimg, _ = val_dataset[p]
                        dbimgs.append(dbimg)
                    dbimgs_tensor = torch.stack(dbimgs, dim=0)
                    qfeat, _ = model(qimg.to(device).unsqueeze(0), None, mode='global')
                    dbfeats, _ = model(dbimgs_tensor.to(device), None, mode='global')
                    qfeats = qfeat.repeat(len(pred), 1, 1)

                # now assemble the pairs
                scoresa = model(qfeats, dbfeats, "pairvpr")
                scoresb = model(dbfeats, qfeats, "pairvpr")
                scores = scoresa + scoresb # inspired by multi-process fusion

                # highest scores are best matches. Rerank preds based on scores
                sortedscores = scores.cpu().squeeze(1).sort(descending=True)
                reranked_scores[q_idx] = sortedscores[0][0]
                reranked_preds = pred[sortedscores[1]]
                reranked_predictions[q_idx, :] = reranked_preds
                del scores, qfeats, dbfeats

        rerank_time_end = time.time()
        timeperimage = (rerank_time_end - rerank_time_start) / val_dataset.num_queries
        print('---PairVPR took this many seconds per query for refine: ', str(timeperimage))

        results_dict, results_dict_refined = get_test_recalls(val_set_name,
                                                        k_values,
                                                        predictions.numpy(),
                                                        reranked_predictions.numpy(),
                                                        ground_truth,
                                                        val_dataset,
                                                        cfg.eval.refinetopcands)
        
        del r_list, q_list, feats, num_references, ground_truth, densefeats
        recalls_global.append(results_dict)
        recalls_refined.append(results_dict_refined)

    return recalls_global, recalls_refined # dict type



def main(args):
    cfg = get_cfg_from_args_eval(args)

    pairvpr_ckpt = torch.load(args.trained_ckpt)

    if torch.cuda.is_available():
        cudaflag = True
    else:
        cudaflag = False
        print('---Warning without a GPU Pair-VPR will be slow')
    
    if cudaflag:
        device = torch.device("cuda")
        model = PairVPRNet(cfg).to(device)
    else:
        model = PairVPRNet(cfg)
    interpolate_pos_embed(cfg, model, pairvpr_ckpt)
    model.load_state_dict(pairvpr_ckpt, strict=False)
    model = model.eval()

    val_dataloaders = []
    val_set_names = []
    val_datasets = []
    all_num_queries = []
    all_num_references = []
    all_ground_truth = []
    for val_name in args.val_datasets:
        val_dataset, num_references, num_queries, ground_truth = dataset_getter.get_test_dataset(
            cfg, args.dsetroot, root_dir, val_name, cfg.augmentation.img_res)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 num_workers=0, # needs to be zero for two-stage VPR
                                                 batch_size=50,
                                                 shuffle=False,
                                                 pin_memory=True)
        val_dataloaders.append(val_loader)
        val_set_names.append(val_name)
        val_datasets.append(val_dataset)
        all_num_queries.append(num_queries)
        all_num_references.append(num_references)
        all_ground_truth.append(ground_truth)

    recalls_global, recalls_refined = test(cfg, val_dataloaders, model, val_set_names, val_datasets, 
         all_num_references, all_num_queries, all_ground_truth, device)
    
    if cudaflag:
        torch.cuda.empty_cache()

    return recalls_global, recalls_refined



if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    recalls_global, recalls_refined = main(args)
    print('done')

