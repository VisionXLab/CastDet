# [Oriented GroundingDINO] Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection

> - [An Open and Comprehensive Pipeline for Unified Object Grounding and Detection](https://arxiv.org/abs/2401.02361)
> - [Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://arxiv.org/abs/2303.05499)


## Dataset Preparation

- Step1: download NWPU dataset, format as:

```text
├── NWPU-RESISC45
    └── NWPU-RESISC45
        ├── CLASS 1
        ├── CLASS 2
        └── ...
```

- Step2: prepare OVD dataset.

```
python projects/GroundingDINO/tools/prepare_ovdg_dataset.py \
    --data_dir data/NWPU-RESISC45/NWPU-RESISC45 \
    --save_path data/NWPU-RESISC45/annotations/nwpu45_unlabeled_2.json
```

## Quick Start:

```shell
bash projects/GroundingDINO/run.sh
```


## Training

> **Note**: we follow the similar training pipeline as CastDet.

-  Step1: train base-detector

```shell
exp1="grounding_dino_swin-t_visdrone_base-set_adamw"
python tools/train.py \
    projects/GroundingDINO/configs/$exp1.py
```

- **[Optional]** Step2: pseudo-labeling

```shell
# 2.1. pseudo-labeling
exp2="grounding_dino_swin-t_visdrone_base-set_adamw_nwpu45_pseudo_labeling"
python tools/test.py \
    projects/GroundingDINO/configs/$exp2.py \
    work_dirs/$exp1/iter_20000.pth

# 2.2. merge predictions
python projects/GroundingDINO/tools/merge_ovdg_preds.py \
    --ann_path data/NWPU-RESISC45/annotations/nwpu45_unlabeled_2.json \
    --pred_path work_dirs/$exp2/nwpu45_pseudo_labeling_2.bbox.json \
    --save_path work_dirs/$exp2/nwpu45_unlabeled_with_gdino_pseudos_swin-t_adamw_top1.json \
    --topk 1

# move to data folder
cp work_dirs/$exp2/snwpu45_unlabeled_with_gdino_pseudos_swin-t_adamw_top1.json data/NWPU-RESISC45/annotations/nwpu45_unlabeled_with_gdino_pseudos_swin-t_adamw_top1.json
```

- **[Optional]** Step3: post-training

```shell
exp3="grounding_dino_swin-t_visdrone_base-set_adamw_nwpu45"
python tools/train.py \
    projects/GroundingDINO/configs/$exp3.py \
    --work-dir work_dirs/$exp3
```

## Evaluation

```shell
python tools/test.py \
    projects/GroundingDINO/configs/$exp3.py \
    work_dirs/$exp3/iter_10000.pth \
    --work-dir work_dirs/$exp3/dior_test
```

## Acknowledgement

Thanks the wonderful open source projects [MMDetection](https://github.com/open-mmlab/mmdetection), [MMRotate](https://github.com/open-mmlab/mmrotate), [RHINO](https://github.com/SIAnalytics/RHINO), and [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)!


## Citation

```
// Oriented GroundingDINO (this repo)
@misc{li2024exploitingunlabeleddatamultiple,
      title={Exploiting Unlabeled Data with Multiple Expert Teachers for Open Vocabulary Aerial Object Detection and Its Orientation Adaptation}, 
      author={Yan Li and Weiwei Guo and Xue Yang and Ning Liao and Shaofeng Zhang and Yi Yu and Wenxian Yu and Junchi Yan},
      year={2024},
      eprint={2411.02057},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.02057}, 
}

// GroundingDINO (Horizontal detection)
@article{liu2023grounding,
  title={Grounding dino: Marrying dino with grounded pre-training for open-set object detection},
  author={Liu, Shilong and Zeng, Zhaoyang and Ren, Tianhe and Li, Feng and Zhang, Hao and Yang, Jie and Li, Chunyuan and Yang, Jianwei and Su, Hang and Zhu, Jun and others},
  journal={arXiv preprint arXiv:2303.05499},
  year={2023}
}
```
