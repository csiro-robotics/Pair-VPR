export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:/YOURPATH/Pair-VPR"
python pairvpr/training/stagetwo_val_searcher.py --dsetroot /YOURDATASETPATH --ckpt_folder runs/pairvpr_stagetwo_vitg --config_filename pairvpr/configs/stagetwo_vitg_config.yaml --val_datasets MSLS_val
