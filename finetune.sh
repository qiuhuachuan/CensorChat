seed=42
cuda=1


model_name_or_path='bert-base-cased'

nohup python -u finetune.py \
--model_name_or_path ${model_name_or_path} \
--num_train_epochs 5 \
--learning_rate 2e-5 \
--seed ${seed} \
--cuda ${cuda} \
> ./log/finetune_${seed}.log &