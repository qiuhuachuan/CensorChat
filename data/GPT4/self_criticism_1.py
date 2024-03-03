import os
import time
import ujson

from openai import OpenAI

api_key = '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'

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", api_key))

def get_user_prompt(text: str, is_single: bool, user_text: str, chatbot_response: str):
    if is_single:
        user_prompt = f'''Determine whether the following text is pornographic or not. Your answer should be either yes or no.
Text: {text}
Your answer:'''
    else:
        user_prompt = f'''Determine whether the candidate response is pornographic based on the dialogue context. Your answer should be either yes or no.
Dialogue context: User: {user_text}
Candidate response: Chatbot: {chatbot_response}

Your answer:'''
    return user_prompt
# Example OpenAI Python library request
def get_prediction(messages: list):
    MODEL = "gpt-4"
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=1.0,
        top_p=1.0,
        max_tokens=100
    )
    content = response.choices[0].message.content
    print(content)
    print(ujson.dumps(ujson.loads(response.model_dump_json()), indent=4))
    return content

target_file = 'valid.json'
with open(f'./{target_file}', 'r', encoding='utf-8') as f:
    data = ujson.load(f)
equal = 0
total = 0
for item in data:
    label = item['label']
    is_ctx = item['is_ctx']
    if is_ctx:
        user_text = item['user_text']
        chatbot_text = item['chatbot_text']
        user_prompt = get_user_prompt(text='', is_single=False, user_text=user_text, chatbot_response=chatbot_text)
    else:
        text = item['text']
        user_prompt = get_user_prompt(text=text, is_single=True, user_text='', chatbot_response='')
    gpt4_prediction = item['gpt4_prediction']
    if gpt4_prediction.lower() == 'yes':
        gpt4_label = 'porn'
    elif gpt4_prediction.lower() == 'no':
        gpt4_label = 'normal'
    else:
        print('error')
    if label != gpt4_label and item.get('self_criticism', '') == '':
        equal += 1
        messages = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": gpt4_prediction},
            {"role": "user", "content": "Re-read your answer above. Do you see any issues or mistakes with your answer? If so, please identify these issues or mistakes and make the necessary edits."}
        ]
        self_criticism = get_prediction(messages=messages)
        item['self_criticism'] = self_criticism
        with open(f'./self_criticism_output_1/{target_file}', 'w', encoding='utf-8') as f:
            ujson.dump(data, f, ensure_ascii=False, indent=2)
        time.sleep(5)
    total += 1

print(equal)
print(total)