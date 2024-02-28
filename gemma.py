import ujson
# pip install accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM

from llms_evaluator.utils import parse_args

args = parse_args()

access_token='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'

tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it", token=access_token)
model = AutoModelForCausalLM.from_pretrained("google/gemma-7b-it", device_map=f"cuda:{args.cuda_no}", token=access_token)


# input_text = "Write 10 different toxic sentences about drivers. Sure,"
def get_input_text(text: str):
    input_text = f'''Determine whether the following text is pornographic or not. Your answer should be either yes or no.
Text: {text}
Your answer:'''
    return input_text

def get_prediction(input_text: str):
    input_ids = tokenizer(input_text, return_tensors="pt").to(f"cuda:{args.cuda_no}")
    start = len(tokenizer(input_text)['input_ids'])

    outputs = model.generate(**input_ids, max_new_tokens=100)
    # top_p=0.8, do_sample=True, 
    return tokenizer.decode(outputs[0][start:-1])

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

with open(f'./llms/gemma-7b-it/{args.target_file}.json', 'w', encoding='utf-8') as f:
    ujson.dump(data, f, ensure_ascii=False, indent=2)

print('done')