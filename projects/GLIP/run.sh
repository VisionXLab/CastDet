DEVICES_ID=3

exp1="glip_atss_r50_a_fpn_dyhead_visdronezsd_base"

# Step1: train base-detector
CUDA_VISIBLE_DEVICES=$DEVICES_ID python tools/train.py \
    projects/GLIP/configs/$exp1.py

# Step2.1: pseudo-labeling
exp2="glip_atss_r50_a_fpn_dyhead_visdronezsd_base_nwpu45_pseudo_labeling"
CUDA_VISIBLE_DEVICES=$DEVICES_ID python tools/test.py \
    projects/GLIP/configs/$exp2.py \
    work_dirs/$exp1/iter_20000.pth

# Step2.2: merge predictions
python projects/GroundingDINO/tools/merge_ovdg_preds.py \
    --ann_path data/NWPU-RESISC45/annotations/nwpu45_unlabeled_2.json \
    --pred_path work_dirs/$exp2/nwpu45_pseudo_labeling_2.bbox.json \
    --save_path work_dirs/$exp2/nwpu45_unlabeled_with_glip_pseudos_2.json

cp work_dirs/$exp2/nwpu45_unlabeled_with_glip_pseudos_2.json data/NWPU-RESISC45/annotations/nwpu45_unlabeled_with_glip_pseudos_2.json

# Step3: self-training
exp3="glip_atss_r50_a_fpn_dyhead_visdronezsd_base_nwpu"
CUDA_VISIBLE_DEVICES=$DEVICES_ID python tools/train.py \
    projects/GLIP/configs/$exp3.py

# Step4: test
CUDA_VISIBLE_DEVICES=$DEVICES_ID python tools/test.py \
    projects/GLIP/configs/$exp3.py \
    work_dirs/$exp3/iter_10000.pth \
    --work-dir work_dirs/$exp3/dior_test
