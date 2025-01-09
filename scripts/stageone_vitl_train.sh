export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH="${PYTHONPATH}:/YOURPATH/Pair-VPR"
torchrun --rdzv-endpoint=localhost:29502 --nproc_per_node=4 pairvpr/training/train_stageone.py -d sf gsv gldv2 --dsetroot /YOURDATASETPATH --config-file pairvpr/configs/stageone_vitl_config.yaml --output_dir runs/pairvpr_stageone_vitl --usewandb
