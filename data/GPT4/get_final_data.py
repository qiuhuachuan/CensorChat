import ujson

target_file = 'valid.json'
with open(f'./self_criticism_output_2/{target_file}', 'r', encoding='utf-8') as f:
    data = ujson.load(f)
not_equal = 0
gpt4_not_equal = 0
total = 0
for item in data:
    label = item['label']
    is_ctx = item['is_ctx']
    gpt4_prediction = item['gpt4_prediction'].lower()
    if gpt4_prediction == 'yes':
        gpt4_label = 'porn'
    elif gpt4_prediction == 'no':
        gpt4_label = 'normal'
    else:
        print('error')
    
    if label != gpt4_label:
        not_equal += 1
        self_criticism = item['self_criticism'].lower()
        gpt4_final_answer = item['gpt4_final_answer'].lower()
        if gpt4_prediction == gpt4_final_answer:
            pass
        else:
            gpt4_not_equal += 1
            print(gpt4_final_answer)

        if gpt4_final_answer == 'yes':
            item['label'] = 'porn'
        elif gpt4_final_answer == 'no':
            item['label'] = 'normal'
        else:
            print('error') 
            # print(item)  
    total += 1
with open(f'./final_data/{target_file}', 'w', encoding='utf-8') as f:
    ujson.dump(data, f, ensure_ascii=False, indent=2)
    

print(not_equal)
print(gpt4_not_equal)
print(total)