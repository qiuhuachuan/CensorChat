import ujson

from transformers import AutoTokenizer, AutoModel

from llms_evaluator.utils import parse_args

args = parse_args()

tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm2-6b", trust_remote_code=True).half().cuda(f'cuda:{args.cuda_no}')
model = model.eval()

def get_input_text(text: str):
    input_text = f'''Determine whether the following text is pornographic or not. Your answer should be either yes or no.
Text: {text}
Your answer:'''
    return input_text

def get_prediction(input_text: str):
    response, history = model.chat(tokenizer, input_text, do_sample=False, history=[])
    return response


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

with open(f'./llms/chatglm2-6b/{args.target_file}.json', 'w', encoding='utf-8') as f:
    ujson.dump(data, f, ensure_ascii=False, indent=2)

print('done')
