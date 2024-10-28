#!/usr/bin/env bash
# bash ./toolsCONFIG=$1
CONFIG=$1
GPUS=$2
CHECKPOINT=$3
PORT=${4:-11350} # Default port if not specified

# Set the CUDA_VISIBLE_DEVICES environment variable
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.run --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch --resume-from $CHECKPOINT "${@:5}"