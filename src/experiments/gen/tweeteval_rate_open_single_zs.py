from transformers import AutoTokenizer
from pathlib import Path
from tqdm import tqdm
import time

from src.model import load_clm_model
from src.prompt import generate_gen_turns, load_phrases
from src.utils import load_config, read_jsonl, write_jsonl
from src.preprocessing import load_tweeteval

N = 10

def generate(model, inputs):
    return model.generate(
        inputs,
        max_new_tokens=512,
        do_sample=True,
        top_p=0.9,
        top_k=50,
        temperature=0.6,
        num_beams=1,
        repetition_penalty=1.2,
    )

config = load_config()
config = config | {
    'RESULT_PATH':"results/tweeteval_rate_open_test_zs.jsonl",
}

tokenizer = AutoTokenizer.from_pretrained(config['CLM_MODEL_NAME'], token=config['HF_TOKEN'])
tokenizer.add_special_tokens({'sep_token':'<SEP>', 'pad_token':'<PAD>', 'cls_token':'<CLS>', 'mask_token':'<MASK>'})
tokenizer.use_default_system_prompt = False

model = load_clm_model(config['CLM_MODEL_NAME'], method=config['LOAD_MODEL_METHOD'], token=config['HF_TOKEN'])
model.eval()

_, _, data = load_tweeteval()

# item = data[0]
results = []

if Path(config['RESULT_PATH']).is_file():
    results = read_jsonl(config['RESULT_PATH'])

for item in tqdm(data, "Generation loop:"):

    if any(r['id_original'] == item['id_original'] for r in results):
        continue

    start_time = time.time()

    turns = [
        {"role":"user", "content":"Do you think that the following tweet: \"{text}\" is ironic?".format(**item)}
    ]

    inputs = tokenizer.apply_chat_template(turns, return_tensors='pt').to(model.device)
    outputs = generate(model, inputs)

    turns.extend([
        {"role": "assistant", "content":tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)},
        {"role": "user", "content":"From 1 to {n}, how ironic do you think the previous tweet is ? Format your answer like x/{n}.".format(n=N)},
    ])

    inputs = tokenizer.apply_chat_template(turns, return_tensors='pt').to(model.device)
    outputs = generate(model, inputs)
    

    results.append({
        'id_original': item['id_original'],
        'gold':item['label'],
        'outputs':tokenizer.decode(outputs[0]),
        'duration': time.time() - start_time
    })

    write_jsonl(config['RESULT_PATH'], results)

