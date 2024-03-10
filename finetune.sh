export CUDA_VISIBLE_DEVICES=1
seed=46



model_name_or_path='bert-base-cased'

nohup python -u finetune.py \
--model_name_or_path ${model_name_or_path} \
--num_train_epochs 10 \
--learning_rate 2e-5 \
--seed ${seed} \
> ./log/finetune_${seed}.log &