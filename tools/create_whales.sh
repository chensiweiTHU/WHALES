#!/usr/bin/env bash

export PYTHONPATH=`pwd`:$PYTHONPATH

python tools/create_data.py whales \
    --root-path ./data/whales \
    --out-dir ./data/whales \
    --extra-tag whales \