#!/usr/bin/env bash
# bash ./tools
CONFIG=$1
GPUS=$2
PORT=${PORT:-29600}

CUDA_VISIBLE_DEVICES=1,2,3,5 \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
