import ujson

def get_three_labels(data: list, idx: int, model_name: str):
    user_label = data[idx][f'{model_name}-user_label']
    chatbot_label = data[idx][f'{model_name}-chatbot_label']
    ctx_label = data[idx][f'{model_name}-ctx_label']

    return user_label, chatbot_label, ctx_label

def get_final_label(labels: list):
    obj = {}
    for label in labels:
        if label in obj:
            obj[label] += 1
        else:
            obj[label] = 1
    result = sorted(obj.items(), key=lambda x: x[1], reverse=True)
    if result[0][1] == 3 or result[0][1] == 4:
        return result[0][0]
    else:
        return ''


all_data = []
for target_file in ['ctx_train_set', 'ctx_valid_test_set']:
    with open(f'./llms/chatglm2-6b/{target_file}-final.json', 'r', encoding='utf-8') as f:
        data_1 = ujson.load(f)
    
    with open(f'./llms/gemma-2b-it/{target_file}-final.json', 'r', encoding='utf-8') as f:
        data_2 = ujson.load(f)

    with open(f'./llms/gemma-7b-it/{target_file}-final.json', 'r', encoding='utf-8') as f:
        data_3 = ujson.load(f)
    
    with open(f'./llms/qwen1-5-7b-chat/{target_file}-final.json', 'r', encoding='utf-8') as f:
        data_4 = ujson.load(f)

    for idx in range(len(data_1)):
        dialogue = data_1[idx]['dialogue']
        user_flag = data_1[idx]['user_flag']
        chatbot_flag = data_1[idx]['chatbot_flag']
        ctx_flag = data_1[idx]['ctx_flag']

        user_labels = []
        chatbot_labels = []
        ctx_labels = []
        for data, model_name in [
            (data_1, 'chatglm2-6b'),
            (data_2, 'gemma-2b-it'),
            (data_3, 'gemma-7b-it'),
            (data_4, 'qwen1-5-7b-chat')
            ]:
            user_label, chatbot_label, ctx_label = get_three_labels(data=data, idx=idx, model_name=model_name)
            user_labels.append(user_label)
            chatbot_labels.append(chatbot_label)
            ctx_labels.append(ctx_label)
        item = {
            'dialogue': dialogue,
            'user_flag': user_flag,
            'chatbot_flag': chatbot_flag,
            'ctx_flag': ctx_flag,
            'user_labels': user_labels,
            'user_label': get_final_label(user_labels),
            'chatbot_labels': chatbot_labels,
            'chatbot_label': get_final_label(chatbot_labels),
            'ctx_labels': ctx_labels,
            'ctx_label': get_final_label(ctx_labels)
        }
        all_data.append(item)

with open(f'./all_data.json', 'w', encoding='utf-8') as f:
    ujson.dump(all_data, f, ensure_ascii=False, indent=2)
print('done')