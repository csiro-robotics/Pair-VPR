export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export PYTHONPATH="${PYTHONPATH}:/YOURPATH/Pair-VPR"
python pairvpr/training/train_stagetwo.py --dsetroot /YOURDATASETPATH --config-file-finetuned pairvpr/configs/stagetwo_vitl_config.yaml --pretrained_ckpt trained_models/pairvpr_stageone_vitL/pairvpr-pretrained-1000epochs-vitL.pth --output_dir runs/pairvpr_stagetwo_vitl --usewandb
