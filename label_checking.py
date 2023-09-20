import os
import ujson
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
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


if __name__ == '__main__':
    # python label_checking.py --model gpt-3.5-turbo-0613 --data_type utterance --dataset_type valid_test
    args = parse_args()

    target_dir = f'./pseudo_label_data/{args.model}/{args.data_type}/{args.dataset_type}'
    existing_files = os.listdir(target_dir)

    for each_file in existing_files:
        with open(f'{target_dir}/{each_file}', 'r', encoding='utf-8') as f:
            data = ujson.load(f)

        label = data['choices'][0]['message']['content']

        if label not in ['NSFW', 'SFW']:
            print(f'{target_dir}/{each_file}')
