import torch
from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM
from tqdm import tqdm

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
                'pred': int(scores.argmax()),
            })
            
    return results