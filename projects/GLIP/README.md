# [Oriented GLIP] GLIP: Grounded Language-Image Pre-training

> [GLIP: Grounded Language-Image Pre-training](https://arxiv.org/abs/2112.03857)

## Abstract

This paper presents a grounded language-image pre-training (GLIP) model for learning object-level, language-aware, and semantic-rich visual representations. GLIP unifies object detection and phrase grounding for pre-training. The unification brings two benefits: 1) it allows GLIP to learn from both detection and grounding data to improve both tasks and bootstrap a good grounding model; 2) GLIP can leverage massive image-text pairs by generating grounding boxes in a self-training fashion, making the learned representation semantic-rich. In our experiments, we pre-train GLIP on 27M grounding data, including 3M human-annotated and 24M web-crawled image-text pairs. The learned representations demonstrate strong zero-shot and few-shot transferability to various object-level recognition tasks. 1) When directly evaluated on COCO and LVIS (without seeing any images in COCO during pre-training), GLIP achieves 49.8 AP and 26.9 AP, respectively, surpassing many supervised baselines. 2) After fine-tuned on COCO, GLIP achieves 60.8 AP on val and 61.5 AP on test-dev, surpassing prior SoTA. 3) When transferred to 13 downstream object detection tasks, a 1-shot GLIP rivals with a fully-supervised Dynamic Head.

<div align=center>
<img src="https://github.com/open-mmlab/mmyolo/assets/17425982/b87228d7-f000-4a5d-b103-fe535984417a"/>
</div>

## Installation

```shell
cd $MMDETROOT

# source installation
pip install -r requirements/multimodal.txt

# or mim installation
mim install mmdet[multimodal]
```

- NOTE

GLIP utilizes BERT as the language model, which requires access to https://huggingface.co/. If you encounter connection errors due to network access, you can download the required files on a computer with internet access and save them locally. Finally, modify the `lang_model_name` field in the config to the local path. Please refer to the following code:

```python
from transformers import BertConfig, BertModel
from transformers import AutoTokenizer

config = BertConfig.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased", add_pooling_layer=False, config=config)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

config.save_pretrained("your path/bert-base-uncased")
model.save_pretrained("your path/bert-base-uncased")
tokenizer.save_pretrained("your path/bert-base-uncased")
```


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
bash projects/GLIP/run.sh
```


## Training

> **Note**: we follow the similar training pipeline as CastDet.

-  Step1: train base-detector

```shell
exp1="glip_atss_r50_a_fpn_dyhead_visdronezsd_base"
python tools/train.py projects/GLIP/configs/$exp1.py
```


- **[Optional]** Step2: pseudo-labeling

```shell
# 2.1 pseudo-labeling
exp2="glip_atss_r50_a_fpn_dyhead_visdronezsd_base_nwpu45_pseudo_labeling"
python tools/test.py \
    projects/GLIP/configs/$exp2.py \
    work_dirs/$exp1/iter_20000.pth

# 2.2 merge predictions
python projects/GroundingDINO/tools/merge_ovdg_preds.py \
    --ann_path data/NWPU-RESISC45/annotations/nwpu45_unlabeled_2.json \
    --pred_path work_dirs/$exp2/nwpu45_pseudo_labeling_2.bbox.json \
    --save_path work_dirs/$exp2/nwpu45_unlabeled_with_glip_pseudos_2.json

# move to data folder
cp work_dirs/$exp2/nwpu45_unlabeled_with_glip_pseudos_2.json data/NWPU-RESISC45/annotations/nwpu45_unlabeled_with_glip_pseudos_2.json
```

- **[Optional]** Step3: post-training

```shell
exp3="glip_atss_r50_a_fpn_dyhead_visdronezsd_base_nwpu"
python tools/train.py \
    projects/GLIP/configs/$exp3.py
```

## Evaluation

```shell
python tools/test.py \
    projects/GLIP/configs/$exp3.py \
    work_dirs/$exp3/iter_10000.pth \
    --work-dir work_dirs/$exp3/dior_test
```

## Acknowledgement

Thanks the wonderful open source projects [MMDetection](https://github.com/open-mmlab/mmdetection), [MMRotate](https://github.com/open-mmlab/mmrotate), and [GLIP](https://github.com/microsoft/GLIP)!


## Citation

```
// Oriented GLIP (this repo)
@misc{li2024exploitingunlabeleddatamultiple,
      title={Exploiting Unlabeled Data with Multiple Expert Teachers for Open Vocabulary Aerial Object Detection and Its Orientation Adaptation}, 
      author={Yan Li and Weiwei Guo and Xue Yang and Ning Liao and Shaofeng Zhang and Yi Yu and Wenxian Yu and Junchi Yan},
      year={2024},
      eprint={2411.02057},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.02057}, 
}

// GLIP (Horizontal detection)
@inproceedings{li2021grounded,
      title={Grounded Language-Image Pre-training},
      author={Liunian Harold Li* and Pengchuan Zhang* and Haotian Zhang* and Jianwei Yang and Chunyuan Li and Yiwu Zhong and Lijuan Wang and Lu Yuan and Lei Zhang and Jenq-Neng Hwang and Kai-Wei Chang and Jianfeng Gao},
      year={2022},
      booktitle={CVPR},
}
```