from tqdm import tqdm
from transformers import TrainingArguments, AutoTokenizer, DataCollatorWithPadding
from datasets import Dataset, load_dataset
import pandas as pd
import torch

from src.preprocessing import iter_splits, make_dataset
from src.utils import load_config, write_jsonl
from src.tokenizer import cls_tokenize
from src.training import IronyTrainer, MCC_Loss
from src.model import load_cls_model
 

OUTPUT_DIR = "results/roberta_irony_wce"
RESULT_PATH = "results/roberta_irony_wce.jsonl"

config = load_config()

def train_loop(tokenizer, model, train, val, current_path):
    training_args = TrainingArguments(
        output_dir=current_path,
        do_train =True,
        do_eval=True,
        evaluation_strategy='epoch',
        prediction_loss_only=False,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=6e-5,
        num_train_epochs=10,
        save_strategy='epoch',
        save_total_limit=5,
        optim='adamw_torch',
        load_best_model_at_end=True,
        logging_strategy="epoch",
        fp16=True 
    )

    train_set = Dataset.from_list(train).map(lambda x: cls_tokenize(tokenizer, x['parent_text'], x['text']))
    val_set = Dataset.from_list(val).map(lambda x: cls_tokenize(tokenizer, x['parent_text'], x['text']))

    label_weights = torch.Tensor([.314, .686]).to(model.device)

    trainer = IronyTrainer(
        loss_funcs = [
            (torch.nn.CrossEntropyLoss(label_weights), 1),
            # (MCC_Loss(), 1), 
        ],
        model=model,
        args=training_args,
        train_dataset=train_set if training_args.do_train else None,
        eval_dataset=val_set if training_args.do_eval else None,
        data_collator=DataCollatorWithPadding(tokenizer)
    )

    trainer.train()
    trainer.save_state()

    return trainer.model

def inference_loop(tokenizer, model, data):
    results = []

    softmax = torch.nn.Softmax(dim=-1)
    model.eval() 

    with torch.no_grad():
        for item in tqdm(data, 'CLS inference loop'):
            tokens = cls_tokenize(tokenizer, item['parent_text'], item['text'], return_tensors='pt').to(model.device)
            outputs = model(**tokens)
            scores = softmax(outputs.logits).detach().cpu().numpy()
            results.append({
                'id_original': item['id_original'],
                'scores': scores[0].tolist(),
                'gold':item['label'],
                'pred': int(scores.argmax()),
            })
            
    return results

def run():

    df = make_dataset(pd.DataFrame(load_dataset(config['DATASET_NAME'])['train']))

    results = []
    current_split = 1

    for train, val, test in iter_splits(config['SPLITS_PATH'], df):
        torch.cuda.empty_cache()
        tokenizer = AutoTokenizer.from_pretrained(config['CLS_MODEL_NAME'], token=config['HF_TOKEN'])
        model = load_cls_model(config['CLS_MODEL_NAME'], method=config['LOAD_MODEL_METHOD'], token=config['HF_TOKEN'])

        current_path = f'{OUTPUT_DIR}_{current_split}'
        
        # train, val, test = train[:2], val[:2], test[:2]
        model = train_loop(tokenizer, model, train, val, current_path)
        
        results.append(inference_loop(tokenizer, model, test))
        
        write_jsonl(RESULT_PATH, results)
        
        current_split+=1

if __name__ == "__main__":
    run()