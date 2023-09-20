import os
import time
import ujson
import argparse

import openai


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', required=True, type=str)
    parser.add_argument('--model',
                        required=True,
                        type=str,
                        choices=['gpt-3.5-turbo-0613', 'gpt-4'])
    parser.add_argument('--data_type',
                        required=True,
                        type=str,
                        choices=['utterance', 'ctx'])
    parser.add_argument('--dataset_type',
                        required=True,
                        type=str,
                        choices=['train', 'valid_test'])
    args = parser.parse_args()

    return args


def GPT_generation(user_msg: str, max_tokens: int, target_dir: str, idx: int,
                   model: str) -> None:
    messages = []
    user_msg = {'role': 'user', 'content': f'''{user_msg}'''}
    messages.append(user_msg)
    completion = openai.ChatCompletion.create(model=model,
                                              temperature=1.0,
                                              top_p=1.0,
                                              messages=messages,
                                              max_tokens=max_tokens)
    completion['query'] = messages
    data_store(completion=completion, target_dir=target_dir, idx=idx)
    time.sleep(12)


def data_store(completion: dict, target_dir: str, idx: int) -> None:
    with open(f'{target_dir}/{idx}.json', 'w', encoding='utf-8') as f:
        ujson.dump(completion, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    # python GPT_annotation.py --model gpt-3.5-turbo-0613 --data_type utterance --dataset_type valid_test --api_key ${place your key here}
    args = parse_args()
    openai.api_key = args.api_key

    utterance_prompt = 'Please determine whether the following text is NSFW or not. Your answer should be either NSFW or SFW.\nText: {}\n\nLabel:'
    ctx_prompt = 'Please determine whether the candidate response is NSFW or not by conditioning on the dialogue context. Your answer should be either NSFW or SFW.\n{}\n\nLabel:'

    with open(
            f'./unlabeled_data/{args.data_type}_{args.dataset_type}_set.json',
            'r',
            encoding='utf-8') as f:
        data = ujson.load(f)
    target_dir = f'./pseudo_label_data/{args.model}/{args.data_type}/{args.dataset_type}'
    os.makedirs(target_dir, exist_ok=True)
    existing_files = os.listdir(target_dir)

    for idx, item in enumerate(data):
        if f'{idx}.json' in existing_files:
            print(f'DONE: {target_dir}/{idx}.json')
        else:
            if args.data_type == 'utterance':
                prompt = utterance_prompt.format(item['text'])
            else:
                dialogue = item['dialogue']
                ctx = dialogue['user']
                response = dialogue['bot']
                ctx_wrapper = f'Dialogue context: <user>: {ctx}\nCandidate response: {response}'
                prompt = ctx_prompt.format(ctx_wrapper)
            try:
                user_msg = prompt
                GPT_generation(user_msg=user_msg,
                               max_tokens=2000,
                               target_dir=target_dir,
                               idx=idx,
                               model=args.model)
                print(f'SUCCESS: {target_dir}/{idx}.json')
            except Exception as e:
                print(f'ERROR-INFO: {e}')
                print(f'ERROR: {target_dir}/{idx}.json')
                time.sleep(60)

    print('All Done!')
