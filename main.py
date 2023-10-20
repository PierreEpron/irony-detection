from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
import pandas as pd
import sys

from src.utils import write_jsonl

from src.preprocessing import make_dataset, iter_splits
from src.model import load_cls_model, cls_inference

mode = sys.argv[1]

DATASET_NAME = "CreativeLang/EPIC_Irony"
CLS_MODEL_NAME = "cardiffnlp/twitter-roberta-base-irony"
CLS_RESULT_PATH = "cls_roberta-irony_zs.jsonl"
SPLITS_PATH = "./results/splits.jsonl"

if mode == 'cls_inf':
    
    tokenizer = AutoTokenizer.from_pretrained(CLS_MODEL_NAME)
    model = load_cls_model(CLS_MODEL_NAME, method="cuda")

    # Change when cv usable
    df = make_dataset(pd.DataFrame(load_dataset(DATASET_NAME)['train']))

    results = []
    current_split = 1

    for _, _, test in iter_splits(SPLITS_PATH, df):
        print(f'##### Starting split: {current_split} #####')
        results.append(cls_inference(tokenizer, model, test))
        current_split+=1
    
    write_jsonl(CLS_RESULT_PATH, results)

else:
    raise ValueError(f"mode possible values are: 'cls_inf'. {mode} is not recognized")