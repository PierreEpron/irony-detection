import torch
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset
from tqdm import tqdm
import evaluate
import copy

from src.tokenizer import clm_template_tokenize, clm_tokenize, cls_tokenize, label_tokenize

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

def cls_inference(tokenizer, model, data):

    results = []

    softmax = torch.nn.Softmax(dim=1)
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

def compute_acc_metrics(p):
    metric = evaluate.load("accuracy")
    preds = torch.nn.functional.softmax(torch.from_numpy(p.predictions), dim=1).argmax(dim=1)
    return metric.compute(predictions=preds, references=p.label_ids)

def cls_train(tokenizer, model, train, val, output_dir):
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        do_train =True,
        do_eval=True,
        evaluation_strategy='epoch',
        prediction_loss_only=False,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=6e-5,
        num_train_epochs=10,
        save_strategy='epoch',
        save_total_limit=1,
        metric_for_best_model="accuracy",
        optim='adamw_torch',
        load_best_model_at_end=True,
        logging_strategy="epoch"
    )

    train_set = Dataset.from_list(train).map(lambda x: cls_tokenize(tokenizer, x['parent_text'], x['text']))
    val_set = Dataset.from_list(val).map(lambda x: cls_tokenize(tokenizer, x['parent_text'], x['text']))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set if training_args.do_train else None,
        eval_dataset=val_set if training_args.do_eval else None,
        compute_metrics=compute_acc_metrics,
        data_collator=DataCollatorWithPadding(tokenizer)
    )

    trainer.train()
    trainer.save_state()

    return trainer.model

def clm_inference(tokenizer, model, data, prompt, system_prompt, instruct_prompt):
    
    results = []

    model.eval()

    with torch.no_grad():
        for item in tqdm(data, 'CLM inference loop'):
            input_ids = clm_tokenize(tokenizer, prompt, system_prompt, instruct_prompt, item, return_tensors='pt').to(model.device)
            
            outputs = model.generate(
                input_ids=input_ids,
                max_new_tokens=8,
                do_sample=True,
            )

            results.append({
                'id_original': item['id_original'],
                'gold':item['label'],
                'output': tokenizer.decode(outputs[0], skip_special_tokens=True)
            })
            
    return results

def clm_next_token(tokenizer, model, data, turns, labels):
    
    model.eval()

    results = []
    softmax = torch.nn.Softmax(dim=1)
    label_ids = label_tokenize(tokenizer, labels, return_tensors='pt').to(model.device)

    with torch.no_grad():
        for item in tqdm(data, 'CLM next token loop'):
            input_ids = clm_template_tokenize(tokenizer, copy.deepcopy(turns), item, return_tensors='pt')[..., :-1].to(model.device)
            logits = model(input_ids).logits
            scores = softmax(logits[..., -1, label_ids]).detach().cpu().numpy()
            results.append({
                'id_original': item['id_original'],
                'scores': scores[0].tolist(),
                'gold':item['label'],
                'pred': int(scores.argmax()),
            })

    return results