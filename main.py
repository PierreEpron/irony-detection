from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
import pandas as pd
import sys

from preprocessing import make_dataset
from src.model import load_cls_model, cls_inference

mode = sys.argv[1]

DATASET_NAME = "CreativeLang/EPIC_Irony"
CLS_MODEL_NAME = "cardiffnlp/twitter-roberta-base-irony"
CLS_RESULT_PATH = "cls_roberta-irony_zs.json"

if mode == 'cls_inf':
    
    tokenizer = AutoTokenizer.from_pretrained(CLS_MODEL_NAME)
    model = load_cls_model(CLS_MODEL_NAME)

    # Change when cv usable
    data = Dataset.from_pandas(make_dataset(pd.DataFrame(load_dataset(DATASET_NAME)['train'])))

    cls_inference(tokenizer, model, CLS_RESULT_PATH)

else:
    raise ValueError(f"mode possible values are: 'cls_inf'. {mode} is not recognized")