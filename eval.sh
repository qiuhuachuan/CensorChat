export CUDA_VISIBLE_DEVICES=2
seed=46

nohup python -u eval.py --seed ${seed}  > ./log/eval-${seed}.log &
