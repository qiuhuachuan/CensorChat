import ujson

total = 0
for filename in ['train', 'valid', 'test']:
    obj = {'is_ctx': {}, 'not_ctx': {}}
    with open(f'./{filename}.json', 'r', encoding='utf-8') as f:
        data = ujson.load(f)
        total += len(data)
    
    for item in data:
        is_ctx = item['is_ctx']
        label = item['label']

        if is_ctx:
            if label in obj['is_ctx']:
                obj['is_ctx'][label] += 1
            else:
                obj['is_ctx'][label] = 1
        else:
            if label in obj['not_ctx']:
                obj['not_ctx'][label] += 1
            else:
                obj['not_ctx'][label] = 1

    print(filename)
    print(obj)
print(total)