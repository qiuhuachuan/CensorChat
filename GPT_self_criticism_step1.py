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
                        choices=['valid_test'])
    args = parser.parse_args()

    return args


def GPT_generation(messages: list, max_tokens: int, model: str) -> None:

    completion = openai.ChatCompletion.create(model=model,
                                              temperature=1.0,
                                              top_p=1.0,
                                              messages=messages,
                                              max_tokens=max_tokens)
    response = completion['choices'][0]['message']['content']
    messages.append({"role": "assistant", "content": response})

    return messages


def data_store(messages: list, target_dir: str, idx: int) -> None:
    with open(f'{target_dir}/{idx}.json', 'w', encoding='utf-8') as f:
        ujson.dump(messages, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    args = parse_args()
    openai.api_key = args.api_key

    source_dir = './pseudo_label_data_without_self_criticism'

    for data_type in ['utterance', 'ctx']:
        target_dir = f'./self_criticism/{args.model}/{args.data_type}/step1'
        os.makedirs(target_dir, exist_ok=True)
        existing_files = os.listdir(target_dir)

        prompt2 = 'Please re-read your above response. Do you see any issues or mistakes with your response? If so, please identify these issues or mistakes and make the necessary edits.'

        with open(f'{source_dir}/{data_type}_valid_test_set.json',
                  'r',
                  encoding='utf-8') as f:
            data = ujson.load(f)

        for idx, item in enumerate(data):
            prompt1 = item['sended_msg']
            gpt4_label = item['gpt4_label']
            chatgpt_label = item['chatgpt_label']

            messages = []
            if gpt4_label != chatgpt_label:
                messages.append({"role": "user", "content": prompt1})
                messages.append({
                    "role":
                    "assistant",
                    "content":
                    gpt4_label if args.model == 'gpt-4' else chatgpt_label
                })
                messages.append({"role": "user", "content": prompt2})

                if f'{idx}.json' in existing_files:
                    print(f'DONE: {target_dir}/{idx}.json')
                else:
                    try:
                        print(messages)
                        new_messages = GPT_generation(messages=messages,
                                                      max_tokens=2000)
                        data_store(messages=new_messages,
                                   target_dir=target_dir,
                                   idx=idx)
                        print(f'SUCCESS: {target_dir}/{idx}.json')
                        time.sleep(12)
                    except Exception as e:
                        print(f'ERROR-INFO: {e}')
                        print(f'ERROR: {target_dir}/{idx}.json')
                        time.sleep(60)

    print('DONE')
