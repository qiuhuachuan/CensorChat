import ujson

from transformers import AutoModelForCausalLM, AutoTokenizer

from llms_evaluator.utils import parse_args

args = parse_args()

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map=f"auto")

def get_input_text(text: str):
    input_text = f'''Determine whether the following text is pornographic or not. Your answer should be either yes or no.
Text: {text}
Your answer:'''
    return input_text

def get_prediction(input_text: str):
    messages = [
        {"role": "user", "content": input_text}
    ]

    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(f"cuda")
    input_length = len(inputs[0])

    outputs = model.generate(inputs, max_new_tokens=20)
    return tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

with open(f'./unlabeled_data/{args.target_file}.json', 'r', encoding='utf-8') as f:
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

with open(f'./llms/mixtral-8x7b-it/{args.target_file}.json', 'w', encoding='utf-8') as f:
    ujson.dump(data, f, ensure_ascii=False, indent=2)

print('done')