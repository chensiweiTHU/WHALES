#!/usr/bin/env bash
CONFIG=$1
GPUS=$2
PORT=${PORT:-11353}
GPU_IDS=${GPU_IDS:-$(seq -s, 0 $((GPUS - 1)))}

CUDA_VISIBLE_DEVICES=$GPU_IDS \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
