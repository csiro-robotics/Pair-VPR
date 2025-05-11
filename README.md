# [IEEE RA-L 2025] Pair-VPR: Place-Aware Pre-training and Contrastive Pair Classification for Visual Place Recognition with Vision Transformers 

<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<p align="center">
  <a href="https://github.com/csiro-robotics/Pair-VPR/commits/main">
    <img src="https://img.shields.io/github/last-commit/csiro-robotics/Pair-VPR" alt="Last Commit" />
  </a>
  <a href="https://github.com/csiro-robotics/Pair-VPR/issues">
    <img src="https://img.shields.io/github/issues/csiro-robotics/Pair-VPR" alt="Issues" />
  </a>
  <a href="https://github.com/csiro-robotics/Pair-VPR/pulls">
    <img src="https://img.shields.io/github/issues-pr/csiro-robotics/Pair-VPR" alt="Pull Requests" />
  </a>
  <a href="https://github.com/csiro-robotics/Pair-VPR/stargazers">
    <img src="https://img.shields.io/github/stars/csiro-robotics/Pair-VPR?style=social" alt="Stars" />
  </a>
</p>


<div align="center" style="line-height: 1;">
  <a href="https://huggingface.co/CSIRORobotics/Pair-VPR" target="_blank" style="margin: 2px;">
    <img alt="Hugging Face"
         src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Pair--VPR-ffc107?color=ffc107&logoColor=white"
         style="display: inline-block; vertical-align: middle;"/>
  </a>

  <!-- RA-L badge -->
  <a href="https://ieeexplore.ieee.org/document/10906598/" target="_blank" style="margin: 2px;">
    <img alt="IEEE RA-L"
         src="https://img.shields.io/badge/%F0%9F%93%84%20RA--L-Published-blue?logo=ieee&logoColor=white"
         style="display: inline-block; vertical-align: middle;" />
  </a>

  <!-- Project Website badge -->
  <a href="https://csiro-robotics.github.io/Pair-VPR/" target="_blank" style="margin: 2px;">
    <img alt="Project Website"
         src="https://img.shields.io/badge/%F0%9F%8C%90%20Project-Website-2ea44f?logo=githubpages&logoColor=white"
         style="display: inline-block; vertical-align: middle;" />
  </a>

</div>

</div>


The official repository of "[Pair-VPR: Place-Aware Pre-training and Contrastive Pair Classification for Visual Place Recognition with Vision Transformers](https://ieeexplore.ieee.org/document/10906598)" published in the IEEE Robotics and Automation Letters, 2025. 

Pre-print paper link: [Arxiv](https://arxiv.org/abs/2410.06614)



### Overview

Pair-VPR is a two-stage VPR method that uses a global descriptor to calculate a list of top database candidates for the current query image, then uses a second stage to re-rank the top candidates for improved place recognition accuracy. 

### Package Requirements

Please refer to the environment.yml file for the list of dependencies. Key requirements include PyTorch and xformers. The environment can be created using conda as follows: conda env create -f environment.yml.

### Setup

First, download the desired datasets. Store all datasets in a common directory. Support is available in this repository for the MSLS dataset, Nordland, Pittsburgh, Tokyo, GSV-Cities, Google Landmarks and SFXL.

Modify the scripts to suit where you clone this repository, and where you saved the datasets to (replace YOURPATH and YOURDATASETPATH).

Finally, to save time it is recommended to download our pre-computed sfxl dataset file sfxl_M15_N1_mipc5.torch, containing the EigenPlaces style groupings of images for the SFXL dataset, which is used in stage one training. This file is located in the Dropbox folder containing our pre-trained models as detailed below. Place the sfxl_M15_N1_mipc5.torch file inside the folder: pairvpr/datasets/datasetfiles/sfxl.

### Pre-trained Model Download

Pre-trained Pair-VPR models can be downloaded from: https://tinyurl.com/Pair-VPR-models or from our HuggingFace page: 

<div align="center">

|      **Model**      | **Download**                                                                      |
| :-----------------: | ------------------------------------------------------------------------------    |
| pairvpr-vitB        | [🤗 HuggingFace](https://huggingface.co/CSIRORobotics/Pair-VPR/pairvpr-vitB.pth)  |
| pairvpr-vitL        | [🤗 HuggingFace](https://huggingface.co/CSIRORobotics/Pair-VPR/pairvpr-vitL.pth)  |
| pairvpr-vitG        | [🤗 HuggingFace](https://huggingface.co/CSIRORobotics/Pair-VPR/pairvpr-vitG.pth)  |


</div>


### Inference

Please refer to scripts/eval_performance.sh and scripts/eval_speed.sh. Performance mode uses the ViT-G checkpoint and runs slower as it uses the top 500 candidates by default. Speed mode uses 100 top candidates and the ViT-B model.

The argument --val_datasets sets which datasets you want to evaluate on. The evaluation script will run on each dataset sequentially.

For customisation, the configs directory contain the default settings such as the number of re-ranked candidates.

Inference requires a GPU and consumes between 5 and 10GB GPU memory depending if using the speed or performance mode. By default, the configs set "memoryeffmode" to False. When memoryeffmode is off, all dense features are saved to CPU RAM for improved compute speed. However if you are on a device with limited RAM, please set memoryeffmode to True. This will avoid saving dense features, and re-compute features during the second stage candidate re-ranking.

### Stage One Training

The scripts starting with "stageone" are the scripts for pre-training Pair-VPR using Mask Image Modelling. Stage One training requires the Google Landmarks v2 dataset, the SFXL dataset and the GSV-Cities dataset. 

ViT-B training requires 2 GPUs, 4 for ViT-L and 8 for ViT-G (ideally with 80GB GPU memory per GPU). Training is expected to take a few days to complete. Training can also be performed with less GPUs, however training will take longer to complete.

### Stage Two Training

The scripts starting with "stagetwo" are the scripts for second stage Pair-VPR training, on GSV-Cities.

Vit-B training requires 5 GPUs, 7 for ViT-L and 8 for ViT-G and each GPU must have at least 80GB GPU memory. Training only takes a few hours to complete. Training can be performed with less GPUs (hard minimum of two GPUs), however the VPR results are likely to reduce.

For speed during training, validation is only performed using the global descriptor. Therefore, a second script called "stagetwo_val_searcher.py" can be used to loop over all trained checkpoints (per epoch) and runs the full two-stage Pair-VPR pipeline during validation. This second script can be used to find the best checkpoint for the best final recall performance of Pair-VPR. By default, MSLS-val is used as the validation set. Please note that the final recall numbers will not exactly match our provided checkpoints, due to slight unavoidable non-deterministic differences in training. The deviations are between zero and one percent in recall on MSLS val.

### Citation

```
@article{hausler2025pairvpr,
  author={Hausler, Stephen and Moghadam, Peyman},
  journal={IEEE Robotics and Automation Letters}, 
  title={Pair-VPR: Place-Aware Pre-Training and Contrastive Pair Classification for Visual Place Recognition With Vision Transformers}, 
  year={2025},
  volume={10},
  number={4},
  pages={4013-4020},
  doi={10.1109/LRA.2025.3546512}
}
```

### Acknowledgements

We acknowledge and thank the contributors to the open source codes and datasets including Dinov2 (https://github.com/facebookresearch/dinov2), EigenPlaces (https://github.com/gmberton/EigenPlaces), GSV-Cities (https://github.com/amaralibey/gsv-cities), SFXL (https://github.com/gmberton/CosPlace) and Google Landmarks (https://arxiv.org/abs/2004.01804).
