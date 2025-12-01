#!/usr/bin/env bash
GPUS=$1
# export CUDA_VISIBLE_DEVICES=0,1,5,6
python -m torch.distributed.launch \
    --nproc_per_node $GPUS \
    --node_rank 0 \
    --master_port 29580 \
    --use_env \
    ./cal_frequency_dist.py \



# --split 'val_scene-1060' \