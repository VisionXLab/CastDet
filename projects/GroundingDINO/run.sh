DEVICES_ID=1

# Step1: train base-detector
exp1="grounding_dino_swin-t_visdrone_base-set_adamw"
CUDA_VISIBLE_DEVICES=$DEVICES_ID python tools/train.py \
    projects/GroundingDINO/configs/$exp1.py

# Step2.1: pseudo-labeling
exp2="grounding_dino_swin-t_visdrone_base-set_adamw_nwpu45_pseudo_labeling"
CUDA_VISIBLE_DEVICES=$DEVICES_ID python tools/test.py \
    projects/GroundingDINO/configs/$exp2.py \
    work_dirs/$exp1/iter_20000.pth

# Step2.2: merge predictions
python projects/GroundingDINO/tools/merge_ovdg_preds.py \
    --ann_path data/NWPU-RESISC45/annotations/nwpu45_unlabeled_2.json \
    --pred_path work_dirs/$exp2/nwpu45_pseudo_labeling_2.bbox.json \
    --save_path work_dirs/$exp2/nwpu45_unlabeled_with_gdino_pseudos_swin-t_adamw_top1.json \
    --topk 1

cp work_dirs/$exp2/nwpu45_unlabeled_with_gdino_pseudos_swin-t_adamw_top1.json data/NWPU-RESISC45/annotations/nwpu45_unlabeled_with_gdino_pseudos_swin-t_adamw_top1.json

# Step3: self-training
exp3="grounding_dino_swin-t_visdrone_base-set_adamw_nwpu45"
exp3_="grounding_dino_swin-t_visdrone_base-set_adamw_nwpu45_"
CUDA_VISIBLE_DEVICES=$DEVICES_ID python tools/train.py \
    projects/GroundingDINO/configs/$exp3.py \
    --work-dir work_dirs/$exp3_

# Step4: test
CUDA_VISIBLE_DEVICES=$DEVICES_ID python tools/test.py \
    projects/GroundingDINO/configs/$exp3.py \
    work_dirs/$exp3_/iter_10000.pth \
    --work-dir work_dirs/$exp3_/dior_test
