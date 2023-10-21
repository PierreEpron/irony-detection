from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
import pandas as pd
import torch
import sys

from src.utils import write_jsonl

from src.preprocessing import make_dataset, iter_splits
from src.model import cls_train, load_cls_model, cls_inference

mode = sys.argv[1]

DATASET_NAME = "CreativeLang/EPIC_Irony"
CLS_MODEL_NAME = "cardiffnlp/twitter-roberta-base-irony"
CLS_MODEL_PATH = "results/roberta-irony-ft"
CLS_RESULT_PATH = "results/cls_roberta-irony_ft.jsonl"
SPLITS_PATH = "results/splits.jsonl"

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

else:
    raise ValueError(f"mode possible values are: 'cls_inf, cls_train'. {mode} is not recognized")