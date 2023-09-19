import os
import argparse
import logging
import ujson

import torch
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from transformers import BertConfig, BertTokenizer
from sklearn.metrics import classification_report
from torch.nn.functional import softmax

from utils.models import BertForSequenceClassification

MODEL_CLASS_MAPPING = {'bert-base-cased': BertForSequenceClassification}

logger = get_logger(__name__)

label_mapping = {0: 'NSFW', 1: 'SFW'}


def parse_args():
    parser = argparse.ArgumentParser(
        'Finetune a transformers model on a text classification task.')
    parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help=
        ('The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,'
         ' sequences shorter will be padded if `--pad_to_max_length` is passed.'
         ))
    parser.add_argument(
        '--pad_to_max_length',
        action='store_true',
        help=
        'If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.'
    )
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        help=
        'Path to pretrained model or model identifier from huggingface.co/models.',
        default='bert-base-cased')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='A seed for reproducible training.')

    parser.add_argument('--cuda', type=str, default='0', help='cuda device')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    config = BertConfig.from_pretrained(args.model_name_or_path,
                                        num_labels=2,
                                        finetuning_task='text classification')
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path,
                                              use_fast=False,
                                              never_split=['[user]', '[bot]'])
    tokenizer.vocab['[user]'] = tokenizer.vocab.pop('[unused1]')
    tokenizer.vocab['[bot]'] = tokenizer.vocab.pop('[unused2]')

    MODEL_CLASS = MODEL_CLASS_MAPPING[args.model_name_or_path]

    model = MODEL_CLASS(config, args.model_name_or_path)
    PATH = f'out/pytorch_model.bin'
    model.load_state_dict(torch.load(PATH))
    model.cuda()

    padding = 'max_length' if args.pad_to_max_length else False
    model.eval()

    with open('./data/test.json', 'r', encoding='utf-8') as f:
        test_data = ujson.load(f)

    equal = 0
    total = 0
    pred_label_list = []
    true_label_list = []
    for idx, item in enumerate(test_data):
        text = item['text']
        label = item['label']

        result = tokenizer.encode_plus(text=text,
                                       padding=padding,
                                       max_length=args.max_length,
                                       truncation=True,
                                       add_special_tokens=True,
                                       return_token_type_ids=True,
                                       return_tensors='pt')
        result = result.to('cuda')

        with torch.no_grad():
            outputs = model(**result)
            prob = softmax(outputs.logits, dim=-1)[0]
            # print(prob)
            predictions = outputs.logits.argmax(dim=-1)
            pred_label = predictions.item()
            pred_label_list.append(pred_label)
            true_label_list.append(label)

            if pred_label == label:
                equal += 1
            total += 1

    print('equal:', equal)
    print('total:', total)
    print(classification_report(y_true=true_label_list,
                                y_pred=pred_label_list))


if __name__ == '__main__':
    main()
    print('done')