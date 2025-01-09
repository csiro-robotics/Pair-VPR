export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export PYTHONPATH="${PYTHONPATH}:/YOURPATH/Pair-VPR"
python pairvpr/training/train_stagetwo.py --dsetroot /YOURDATASETPATH --config-file-finetuned pairvpr/configs/stagetwo_default_config.yaml --pretrained_ckpt trained_models/pairvpr_stageone_vitB/pairvpr-pretrained-500epochs-vitB.pth --output_dir runs/pairvpr_stagetwo_vitb --usewandb
