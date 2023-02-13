#!/bin/bash
PYTHONPATH=. python scripts/tests/test_load_mymodel_and_infer.py \
    --train_yaml "../orig-MeshGraphormer/freihand/train.yaml" \
    --val_yaml "../orig-MeshGraphormer/freihand/test.yaml" \
    --ring_info_pkl_rootdir data/ring_info/ \
    --fastmetro_resume_checkpoint "ckpt/FastMETRO-L-H64_freihand_state_dict.bin" \
    --mymodel_resume_dir "/Users/taichi.muraki/out/2023/010/checkpoint-40/" \
    --device cpu

