python -m torch.distributed.launch --nproc_per_node=1 \
 src/tools/run_fastmetro_handmesh.py \
 --train_yaml temp/freihand/train.yaml \
 --val_yaml temp/freihand/test.yaml \
--arch resnet50 \
--model_name FastMETRO-L \
--num_workers 4 \
--per_gpu_train_batch_size 16 \
--per_gpu_eval_batch_size 16 \
--lr 1e-4 \
--num_train_epochs 200 \
--output_dir FastMETRO_outputs \
--saving_epochs 1 \
--resume_epoch 2 \
--resume_checkpoint FastMETRO-small/checkpoint-3-24420/state_dict.bin
