from transformers import AutoTokenizer
from tqdm import tqdm

from src.model import clm_load_tweeteval, load_clm_model
from src.prompt import generate_gen_turns, load_phrases
from src.utils import load_config, write_jsonl

def generate(model, inputs):
    return model.generate(
        inputs,
        temperature=0.6,
        top_p=0.95,
        repetition_penalty=1.2,
        top_k=50,
        max_new_tokens=1024,
    )


config = load_config()
config = config | {
    'RESULT_PATH':"results/tweeteval_base_open_single_zs.jsonl",
    'CLM_PHRASES_PATH':"src/prompts/gen_single_phrases.json"
}

tokenizer = AutoTokenizer.from_pretrained(config['CLM_MODEL_NAME'], token=config['HF_TOKEN'])
tokenizer.add_special_tokens({'sep_token':'<SEP>', 'pad_token':'<PAD>', 'cls_token':'<CLS>', 'mask_token':'<MASK>'})
tokenizer.use_default_system_prompt = False

model = load_clm_model(config['CLM_MODEL_NAME'], method=config['LOAD_MODEL_METHOD'], token=config['HF_TOKEN'])
model.eval()

data = clm_load_tweeteval(config)
phrases, _ = load_phrases(config['CLM_PHRASES_PATH'])

# item = data[0]
results = []

for item in tqdm(data, "Generation loop:"):
    turns, seed_phs, subs = generate_gen_turns(item, phrases)

    inputs = tokenizer.apply_chat_template(turns, return_tensors='pt').to(model.device)
    outputs = generate(model, inputs)

    turns.extend([
        {"role": "assistant", "content":tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)},
        {"role": "user", "content":"List the keywords that led you to your conclusion."},
    ])

    inputs = tokenizer.apply_chat_template(turns, return_tensors='pt').to(model.device)
    outputs = generate(model, inputs)
    

    results.append({
        'id_original': item['id_original'],
        'gold':item['label'],
        'turns':turns, 
        'seed_phs':seed_phs,
        'subs':subs,
        'outputs':tokenizer.decode(outputs[0])
    })

    write_jsonl(config['RESULT_PATH'], results)

