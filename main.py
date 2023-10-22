from transformers import AutoTokenizer
from datasets import load_dataset
from dotenv import load_dotenv
import pandas as pd
import torch
import sys
import os

from src.model import clm_inference, cls_train, load_clm_model, load_cls_model, cls_inference
from src.preprocessing import make_dataset, iter_splits
from src.utils import write_jsonl

load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')

DATASET_NAME = "CreativeLang/EPIC_Irony"
SPLITS_PATH = "results/splits.jsonl"

# CLS_MODEL_NAME = "cardiffnlp/twitter-roberta-base-irony"
# CLS_MODEL_PATH = "results/roberta-irony-ft"
# CLS_RESULT_PATH = "results/cls_roberta-irony_ft.jsonl"

CLS_MODEL_NAME = "roberta-base"
CLS_MODEL_PATH = "results/roberta-base-ft"
CLS_RESULT_PATH = "results/cls_roberta-base_ft.jsonl"

CLM_MODEL_NAME = 'meta-llama/Llama-2-7b-hf'
CLM_RESULT_PATH = "results/clm_llama_0.jsonl"

# PROMPT = "[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\nQUESTION:\n{instruct_prompt}\nINPUT:\n - {parent_text} \n - {text} [/INST]\nANSWER: "
# SYSTEM_PROMPT = 'Answer to the following question with only one word'
# INSTRUCT_PROMPT = "Is the following input exchange ironic?"

PROMPT = "[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\nQUESTION:\n{instruct_prompt}\nDIALOG:\n\tB: {parent_text}\n\tA: {text}\nANSWER: [/INST]\n\n"
SYSTEM_PROMPT = 'Answer only with the word "Yes" or "No"'
INSTRUCT_PROMPT = "Classify this dialog as ironic or not"

mode = sys.argv[1]

if mode == 'cls_inf':
    
    tokenizer = AutoTokenizer.from_pretrained(CLS_MODEL_NAME)
    model = load_cls_model(CLS_MODEL_NAME, method="cuda")

    df = make_dataset(pd.DataFrame(load_dataset(DATASET_NAME)['train']))

    results = []
    current_split = 1

    for _, _, test in iter_splits(SPLITS_PATH, df):
        torch.cuda.empty_cache()

        print(f'##### Starting split: {current_split} #####')
        
        results.append(cls_inference(tokenizer, model, test))
        current_split+=1
    
    write_jsonl(CLS_RESULT_PATH, results)

elif mode == 'cls_train':

    tokenizer = AutoTokenizer.from_pretrained(CLS_MODEL_NAME)
    model = load_cls_model(CLS_MODEL_NAME, method="cuda")
    
    df = make_dataset(pd.DataFrame(load_dataset(DATASET_NAME)['train']))

    results = []
    current_split = 1

    for train, val, test in iter_splits(SPLITS_PATH, df):
        torch.cuda.empty_cache()

        current_path = f'{CLS_MODEL_PATH}-{current_split}'
        
        model = cls_train(tokenizer, model, train, val, current_path)
        
        results.append(cls_inference(tokenizer, model, test))
        
        write_jsonl(CLS_RESULT_PATH, results)
        
        current_split+=1

elif mode == 'clm_inf':

    tokenizer = AutoTokenizer.from_pretrained(CLM_MODEL_NAME, token=HF_TOKEN)
    model = load_clm_model(CLM_MODEL_NAME, method="cuda", token=HF_TOKEN)
    
    df = make_dataset(pd.DataFrame(load_dataset(DATASET_NAME)['train']))
    
    results = []
    current_split = 1
    
    for _, _, test in iter_splits(SPLITS_PATH, df):
        torch.cuda.empty_cache()
        
        print(f'##### Starting split: {current_split} #####')
        
        results.append(clm_inference(tokenizer, model, test, PROMPT, SYSTEM_PROMPT, INSTRUCT_PROMPT))
        current_split+=1

        write_jsonl(CLM_RESULT_PATH, results)
    

else:
    raise ValueError(f"mode possible values are: 'cls_inf, cls_train, clm_inf'. {mode} is not recognized")