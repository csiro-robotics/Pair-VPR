export PYTHONPATH="${PYTHONPATH}:/YOURPATH/Pair-VPR"
python pairvpr/training/train_stagetwo.py --dsetroot /YOURDATASETPATH --config-file-finetuned pairvpr/configs/stagetwo_vitg_config.yaml --pretrained_ckpt trained_models/pairvpr_stageone_vitG/pairvpr-pretrained-1000epochs-vitG.pth --output_dir runs/pairvpr_stagetwo_vitg --usewandb
