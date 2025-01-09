export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:/YOURPATH/Pair-VPR"
python pairvpr/eval/eval.py --dsetroot /YOURDATASETPATH --config-file-eval pairvpr/configs/pairvpr_speed.yaml --trained_ckpt trained_models/pairvpr-vitB.pth --val_datasets MSLS_val # MSLS_val tokyo247 Nordland MSLS_test pitts30k_test
