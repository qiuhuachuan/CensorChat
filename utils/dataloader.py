import ujson

from datasets import Dataset, DatasetDict


def load_data(data_dir: str, file_names: list):
    data_wrapper = {}
    for file_name in file_names:
        instances = {'text': [], 'label': []}
        with open(f'{data_dir}/{file_name}.json', 'r', encoding='utf-8') as f:
            data = ujson.load(f)

        for item in data:
            text = item['text']
            label = item['label']

            instances['text'].append(text)
            instances['label'].append(label)
        data_wrapper[file_name] = Dataset.from_dict(instances)

    dataset = DatasetDict(data_wrapper)
    return dataset


if __name__ == '__main__':
    '''
    To test, run this file in the root folder.
    `python ./utils/dataloader.py`
    '''
    dataset = load_data(data_dir='./data',
                        file_names=['train', 'valid', 'test'])
    print(dataset)
