#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
# PORT=${PORT:-29505}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.2"}

# PORT=${PORT:-28501}
# 自动寻找一个可用的端口
get_free_port(){
    while :
    do
        PORT=$(shuf -i 20000-65000 -n 1)
        (echo >/dev/tcp/127.0.0.1/$PORT) &>/dev/null || break
    done
    echo $PORT
}
PORT=${PORT:-$(get_free_port)}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    --use_env \
    $(dirname "$0")/test_rayiou.py \
    $CONFIG \
    $CHECKPOINT \
    --launcher pytorch \
    ${@:4}
