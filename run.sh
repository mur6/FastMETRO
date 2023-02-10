#!/bin/bash
PYTHONPATH=. python scripts/custom/train.py \
    --train_yaml "../orig-MeshGraphormer/freihand/train.yaml" \
    --val_yaml "../orig-MeshGraphormer/freihand/test.yaml" \
    --ring_info_pkl_rootdir data/ring_info/ \
    --batch_size 64
