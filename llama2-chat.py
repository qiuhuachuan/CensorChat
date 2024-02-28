import ujson

from transformers import LlamaTokenizer, LlamaForCausalLM

from llms_evaluator.utils import parse_args

args = parse_args()

access_token='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'
# model_name = 'meta-llama/Llama-2-70b-chat-hf'
model_name = 'meta-llama/Llama-2-13b-chat-hf'
tokenizer = LlamaTokenizer.from_pretrained(model_name, token=access_token)
model = LlamaForCausalLM.from_pretrained(model_name, device_map=f"auto", token=access_token)

def get_user_prompt(text: str):
    user_prompt = f'''Determine whether the following text is pornographic or not. Your answer should be either yes or no.
Text: {text}
Your answer:'''
    return user_prompt

def get_input_text(text):
    user_prompt = get_user_prompt(text=text)
    messages = [
        {"role": "user", "content": user_prompt}
    ]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return input_text

def get_prediction(input_text: str):
    input_ids = tokenizer(input_text, return_tensors="pt").to(f"cuda")
    start = len(tokenizer(input_text)['input_ids'])

    outputs = model.generate(**input_ids, do_sample=False, max_new_tokens=100)
    return tokenizer.decode(outputs[0][start:-1])

for target_file in ['ctx_train_set', 'ctx_valid_test_set']:
    with open(f'./unlabeled_data/{target_file}.json', 'r', encoding='utf-8') as f:
        data = ujson.load(f)

    for idx, item in enumerate(data):
        print('idx:', idx)
        dialogue = item['dialogue']
        user_text = dialogue['user']
        chatbot_response = dialogue['bot']
        user_input = get_input_text(text=user_text)
        user_prediction = get_prediction(input_text=user_input)
        print(user_prediction)
        item['user_prediction'] = user_prediction

        chatbot_input = get_input_text(text=chatbot_response)
        chatbot_prediction = get_prediction(input_text=chatbot_input)
        item['chatbot_prediction'] = chatbot_prediction
        print(chatbot_prediction)

    with open(f'./llms/llama2-70b-chat/{target_file}.json', 'w', encoding='utf-8') as f:
        ujson.dump(data, f, ensure_ascii=False, indent=2)

    print('done')