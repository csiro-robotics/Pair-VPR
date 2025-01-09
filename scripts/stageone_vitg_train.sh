export PYTHONPATH="${PYTHONPATH}:/YOURPATH/Pair-VPR"
torchrun --rdzv-endpoint=localhost:29503 --nproc_per_node=8 pairvpr/training/train_stageone.py -d sf gsv gldv2 --dsetroot /YOURDATASETPATH --config-file pairvpr/configs/stageone_vitg_config.yaml --output_dir runs/pairvpr_stageone_vitg --usewandb
