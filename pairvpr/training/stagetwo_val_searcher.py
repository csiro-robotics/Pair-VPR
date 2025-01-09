'''
Train stagetwo uses the global descriptor as the validation metric. 
However to guarantee the best checkpoint, it is recommended to run the two stage pipeline on the val set
to find the optimal checkpoint. This is slow, therefore it is provided in a separate script.
'''

import os
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

import glob
import argparse
import pandas as pd
from pairvpr.eval.eval import main


VAL_DATASETS = ['tokyo247', 'MSLS_val', 'MSLS_test',
                'pitts30k_test', 'pitts250k_test', 'Nordland']

def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("Locas fine tune training", add_help=add_help)
    parser.add_argument("--ckpt_folder", "--ckpt-folder", type=str, required=True)
    parser.add_argument("--config_filename", "--config-filename", type=str, required=True, 
                        default="pairvpr/configs/stagetwo_default_config.yaml")
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


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    ckpt_list = [
        args.ckpt_folder
    ]
    runsfolder = "runs"
    folders = glob.glob(runsfolder + '/*')
    allresults = []
    for folder in folders:
        if folder not in ckpt_list:
            continue
        checkpoint_filenames = glob.glob(folder + '/*')
        p = Path(folder)
        fname = p.stem
        print(fname)
        if 'stagetwo' in fname:
            configfilename = args.config_filename
            for ckpt_filename in checkpoint_filenames:
                evalargs = {
                    'config_file_eval': configfilename,
                    'trained_ckpt': ckpt_filename, # trained_models/pairvpr-pretrained-500epochs-vitB.pth
                    'val_datasets': args.val_datasets,
                    'dsetroot': args.dsetroot
                }
                mynamespace = argparse.Namespace(**evalargs)
                try:
                    recalls_global, recalls_refined = main(mynamespace)
                except:
                    print('error in this folder, skipping')
                    continue
                resultsdict = {
                    'name': ckpt_filename,
                    'recall@1': recalls_global[0][1],
                    'recall@5': recalls_global[0][5],
                    'recall@10': recalls_global[0][10],
                    'recall_refined@1': recalls_refined[0][1],
                    'recall_refined@5': recalls_refined[0][5],
                    'recall_refined@10': recalls_refined[0][10]
                }
                allresults.append(resultsdict)
        else:
            continue

    resultdf = pd.DataFrame(allresults)
    resultdf.to_csv(args.ckpt_folder + '/allvalresults.csv', index=False)
    print('finished')
