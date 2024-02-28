import ujson

obj = {}
obj_ctx = {}
for target_file in ['ctx_train_set', 'ctx_valid_test_set']:
    with open(f'./unlabeled_data/{target_file}.json', 'r', encoding='utf-8') as f:
        data = ujson.load(f)

    for idx, item in enumerate(data):
        print('idx:', idx)
        dialogue = item['dialogue']
        user_text = dialogue['user']
        if user_text in obj:
            obj[user_text] += 1
            dialogue['user_flag'] = False
        else:
            obj[user_text] = 1
            dialogue['user_flag'] = True

        chatbot_response = dialogue['bot']
        if chatbot_response in obj:
            obj[chatbot_response] += 1
            dialogue['chatbot_flag'] = False
        else:
            obj[chatbot_response] = 1
            dialogue['chatbot_flag'] = True
        ctx = f'[user] {user_text} [chatbot] {chatbot_response}'
        if ctx in obj_ctx:
            obj_ctx[ctx] += 1
            dialogue['ctx_flag'] = False
        else:
            obj_ctx[ctx] = 1
            dialogue['ctx_flag'] = True


    with open(f'./llms/deepl/{target_file}.json', 'w', encoding='utf-8') as f:
        ujson.dump(data, f, ensure_ascii=False, indent=2)
# for key in obj.keys():
#     if obj[key] > 1:
#         print(obj)

print(len(obj.keys()))
print(len(obj_ctx.keys()))

# train
# 68679 utterance
# 40371 ctx

# valid_test
# 10890 utterance
# 7256 ctx
print('done')