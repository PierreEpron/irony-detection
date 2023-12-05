from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset, load_dataset
from tqdm import tqdm
import pandas as pd
import torch

from src.preprocessing import iter_splits, make_dataset, load_tweeteval
from src.tokenizer import cls_double_tokenize
from src.utils import write_jsonl
from src.training import IronyTrainer

def load_cls_model(model_name, method='acc', **kwargs):
    if method == 'acc':
        return load_cls_acc_model(model_name, **kwargs)
    elif method == 'cuda':
        return load_cls_cuda_model(model_name, **kwargs)
    else:
        raise ValueError(f"method should be equal to 'acc' or ''cuda' not '{method}'")
    
def load_clm_model(model_name, method='acc', **kwargs):
    if method == 'acc':
        return load_clm_acc_model(model_name, **kwargs)
    elif method == 'cuda':
        return load_clm_cuda_model(model_name, **kwargs)
    else:
        raise ValueError(f"method should be equal to 'acc' or ''cuda' not '{method}'")

def load_cls_acc_model(model_name, **kwargs):
    return AutoModelForSequenceClassification.from_pretrained(model_name, device_map="auto", **kwargs)

def load_cls_cuda_model(model_name, **kwargs):
    return AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs).to('cuda')

def load_clm_acc_model(model_name, **kwargs):
    return AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", max_memory={0:'5GiB', 'cpu':'10GiB'}, **kwargs)

def load_clm_cuda_model(model_name, **kwargs):
    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs).to('cuda')

def cls_load_epic(config):
    df = make_dataset(pd.DataFrame(load_dataset(config['DATASET_NAME'])['train']))
    return iter_splits(config['SPLITS_PATH'], df)

def cls_load_tweeteval(config):
    return [load_tweeteval()]

def cls_train(tokenizer, model, train, val, current_path, loss_funcs):
    
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

    trainer = IronyTrainer(
        loss_funcs = loss_funcs,
        model=model,
        args=training_args,
        train_dataset=train if training_args.do_train else None,
        eval_dataset=val if training_args.do_eval else None,
        data_collator=DataCollatorWithPadding(tokenizer)
    )

    trainer.train()
    trainer.save_state()

    return trainer.model

def cls_inference(tokenizer, model, data):
    results = []

    softmax = torch.nn.Softmax(dim=-1)
    model.eval() 

    with torch.no_grad():
        for item in tqdm(data, 'CLS inference loop'):
            outputs = model(**item)
            scores = softmax(outputs.logits).detach().cpu().numpy()
            results.append({
                'id_original': item['id_original'],
                'scores': scores[0].tolist(),
                'gold':item['label'],
                'pred': int(scores.argmax()),
            })
            
    return results

def run(
        config,
        load_data_func=cls_load_epic, 
        tokenize_func=cls_double_tokenize, 
        train_func=cls_train, 
        inference_func=cls_inference):


    results = []
    current_split = 1

    for train, val, test in load_data_func(config):

        torch.cuda.empty_cache()

        tokenizer = AutoTokenizer.from_pretrained(config['CLS_MODEL_NAME'], token=config['HF_TOKEN'])
        model = load_cls_model(config['CLS_MODEL_NAME'], method=config['LOAD_MODEL_METHOD'], token=config['HF_TOKEN'])

        # train, val, test = train[:2], val[:2], test[:2]
        train_set = Dataset.from_list(train).map(lambda x: tokenize_func(tokenizer, **x))
        val_set = Dataset.from_list(val).map(lambda x: tokenize_func(tokenizer, **x))
        test_set = Dataset.from_list(test).map(lambda x: tokenize_func(tokenizer, **x))

        current_path = f"{config['OUTPUT_DIR']}_{current_split}"
        
        model = train_func(tokenizer, model, train_set, val_set, current_path, config['LOSS_FUNCS'])
        
        results.append(inference_func(tokenizer, model, test_set))
        
        write_jsonl(config['RESULT_PATH'], results)
        
        current_split+=1