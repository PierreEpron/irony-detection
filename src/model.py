import torch
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset
from tqdm import tqdm
import evaluate

from src.tokenizer import cls_tokenize

def load_cls_model(model_name, method='acc'):
    if method == 'acc':
        return load_cls_acc_model(model_name)
    elif method == 'cuda':
        return load_cls_cuda_model(model_name)
    else:
        raise ValueError(f"method should be equal to 'acc' or ''cuda' not '{method}'")
    
def load_clm_model(model_name, method='acc'):
    if method == 'acc':
        return load_clm_acc_model(model_name)
    elif method == 'cuda':
        return load_clm_cuda_model(model_name)
    else:
        raise ValueError(f"method should be equal to 'acc' or ''cuda' not '{method}'")

def load_cls_acc_model(model_name):
    return AutoModelForSequenceClassification.from_pretrained(model_name, device_map="auto")

def load_cls_cuda_model(model_name):
    return AutoModelForSequenceClassification.from_pretrained(model_name).to('cuda')

def load_clm_acc_model(model_name):
    return AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

def load_clm_cuda_model(model_name):
    return AutoModelForCausalLM.from_pretrained(model_name).to('cuda')

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
        load_best_model_at_end=True
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