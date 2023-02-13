#!/bin/bash
PYTHONPATH=. python scripts/custom/train.py \
    --train_yaml "../Datasets/freihand/train.yaml" \
    --val_yaml "../Datasets/freihand/test.yaml" \
    --fastmetro_resume_checkpoint "models/fastmetro_checkpoint/FastMETRO-L-H64_freihand_state_dict.bin" \
    --ring_info_pkl_rootdir "../ring_info/" \
    --mymodel_resume_dir "models/checkpoint-40/" \
    --lr 0.000095 \
    --batch_size 32
