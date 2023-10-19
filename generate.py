import torch
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.preprocessing import make_dataset
from tqdm import tqdm
import json
from pathlib import Path

TOKEN = "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"
DATASET_NAME = "CreativeLang/EPIC_Irony"
DEVICE = "cuda"

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", token=TOKEN, device_map="auto")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", token=TOKEN, device_map="auto")

data = Dataset.from_pandas(make_dataset(pd.DataFrame(load_dataset(DATASET_NAME)['train']), equality='not'))

prompt = "[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\nQUESTION:\n{instruct_prompt}\nINPUT:\n - {parent_text} \n - {text} [/INST]\nANSWER: "
system_prompt = 'Answer to the following question with the given input by "Yes" or "No"'
instruct_prompt = "Is the following exchange ironic?"

def tokenize(tokenizer, prompt, system_prompt, instruct_prompt, item):
    tokens = prompt.format(
        system_prompt=system_prompt, 
        instruct_prompt=instruct_prompt, 
        parent_text=item['parent_text'], 
        text=item['text']
    )
    tokens = tokenizer.encode(tokens)
    tokens = torch.tensor(tokens).long().unsqueeze(0)
    return tokens.to(DEVICE)

results = {}

for item in tqdm(data):
    input_ids = tokenize(tokenizer, prompt, system_prompt, instruct_prompt, item)
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=16,
        do_sample=True,
    )
    results[item['id_original']] = tokenizer.decode(outputs[0], skip_special_tokens=True)
    Path('llama_with_sys.json').write_text(json.dumps(results))