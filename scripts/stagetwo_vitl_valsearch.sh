export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:/YOURPATH/Pair-VPR"
python pairvpr/training/stagetwo_val_searcher.py --dsetroot /YOURDATASETPATH --ckpt_folder runs/pairvpr_stagetwo_vitl --config_filename pairvpr/configs/stagetwo_vitl_config.yaml --val_datasets MSLS_val
