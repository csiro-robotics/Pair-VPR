export CUDA_VISIBLE_DEVICES=0,1
export PYTHONPATH="${PYTHONPATH}:/YOURPATH/Pair-VPR"
torchrun --rdzv-endpoint=localhost:29501 --nproc_per_node=2 pairvpr/training/train_stageone.py -d sf gsv gldv2 --dsetroot /YOURDATASETPATH --config-file pairvpr/configs/stageone_default_config.yaml --output_dir runs/pairvpr_stageone_vitb --usewandb
