cuda_no=1,2
target_file=ctx_valid_test_set
# target_file=ctx_train_set
# nohup python -u gemma.py --target_file ${target_file} --cuda_no ${cuda_no} > ./log/gemma-7b-${target_file}.log &
# nohup python -u chatglm2-6b.py --target_file ${target_file} --cuda_no ${cuda_no} > ./log/chatglm2-6b-${target_file}.log &
# nohup python -u chatglm2-6b-ctx.py --target_file ${target_file} --cuda_no ${cuda_no} > ./log/chatglm2-6b-${target_file}-ctx.log &
# nohup python -u gemma-ctx.py --target_file ${target_file} --cuda_no ${cuda_no} > ./log/gemma-2b-${target_file}-ctx.log &
# nohup python -u gemma-ctx.py --target_file ${target_file} --cuda_no ${cuda_no} > ./log/gemma-7b-${target_file}-ctx.log &

# export CUDA_VISIBLE_DEVICES=0,1
# nohup python -u mixtral-8x7b-it.py --target_file ${target_file} --cuda_no ${cuda_no} > ./log/mixtral-8x7b-it-${target_file}.log &

# export CUDA_VISIBLE_DEVICES=1
# nohup python -u llama2-chat.py --target_file ${target_file} --cuda_no ${cuda_no} > ./log/llama2-7b-chat-both.log &

# export CUDA_VISIBLE_DEVICES=1
# nohup python -u qwen1-5-7b-chat.py --target_file ${target_file} --cuda_no ${cuda_no} > ./log/qwen1-5-7b-chat-both.log &

export CUDA_VISIBLE_DEVICES=2
nohup python -u qwen1-5-7b-chat-ctx.py --target_file ${target_file} --cuda_no ${cuda_no} > ./log/qwen1-5-7b-chat-both-ctx.log &