DEVICES_ID=0

# Step1: train the base model
exp1="oriented-rcnn_r50-fpn_20k_visdronezsd_base-set"
CUDA_VISIBLE_DEVICES=$DEVICES_ID python tools/train.py \
    projects/ViLD/configs/$exp1.py

## Step2: merge weights
python projects/CastDetv2/tools/merge_weights.py \
    --clip_path checkpoints/RemoteCLIP-RN50.pt \
    --base_path work_dirs/$exp1/iter_20000.pth \
    --save_path work_dirs/$exp1/merged_vild_init_iter20k.pth
    --target_model vild


# Step3: prepare pseudo labels
exp2="vild_oriented-rcnn_r50_fpn_visdronezsd_step1_prepare"
CUDA_VISIBLE_DEVICES=$DEVICES_ID python tools/test.py \
    projects/ViLD/configs/$exp2.py \
    work_dirs/$exp1/merged_vild_init_iter20k.pth

# Step4: self-training
exp3="vild_oriented-rcnn_r50_fpn_visdronezsd_step2_finetune"
CUDA_VISIBLE_DEVICES=$DEVICES_ID python tools/train.py \
    projects/ViLD/configs/$exp3.py

# Step5: test
CUDA_VISIBLE_DEVICES=$DEVICES_ID python tools/test.py \
    projects/ViLD/configs/$exp3.py \
    work_dirs/$exp3/iter_10000.pth \
    --work-dir work_dirs/$exp3/dior_test

