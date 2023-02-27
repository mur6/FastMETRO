#!/bin/bash

if [ "$1" = "logic" ]; then
    echo "Start infer with logic:"
    PYTHONPATH=. python scripts/tests/test_infer_with_logic.py
else
    PYTHONPATH=. python scripts/tests/test_load_mymodel_and_infer.py \
        --train_yaml "../orig-MeshGraphormer/freihand/train.yaml" \
        --val_yaml "../orig-MeshGraphormer/freihand/test.yaml" \
        --ring_info_pkl_rootdir data/ring_info/ \
        --fastmetro_resume_checkpoint "ckpt/FastMETRO-L-H64_freihand_state_dict.bin" \
        --mymodel_resume_dir "/Users/taichi.muraki/out/2023/012/checkpoint-84" \
        --device cpu
    #     --mymodel_resume_dir "/Users/taichi.muraki/out/2023/011/checkpoint-15/" \
fi

# PYTHONPATH=. python scripts/dataset/test_loader.py \
#     --train_yaml "../orig-MeshGraphormer/freihand/train.yaml" \
#     --val_yaml "../orig-MeshGraphormer/freihand/test.yaml" \
#     --ring_info_pkl_rootdir data/ring_info/ \
