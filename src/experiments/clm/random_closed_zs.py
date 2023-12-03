from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import torch

from src.prompt import generate_turns, load_phrases
from src.utils import load_config, write_jsonl
from src.preprocessing import make_dataset
from src.tokenizer import label_tokenize
from src.model import load_clm_model

RESULT_PATH = "results/clm_random_closed_zs.jsonl"

def epoch(tokenizer, model, data, phrases, output_func, label_ids):
    results = []

    for item in tqdm(data, 'CLM next token loop'):
        turns, seed_phs, subs  = generate_turns(item, phrases)
        input_ids = tokenizer.apply_chat_template(turns, return_tensors='pt')[..., :-1].to(model.device)

        logits = model(input_ids).logits
        scores = output_func(logits[..., -1, label_ids]).detach().cpu().numpy()

        results.append({
            'id_original': item['id_original'],
            'scores': scores[0].tolist(),
            'gold':item['label'],
            'pred': int(scores.argmax()),
            'turns':turns, 
            'seed_phs':seed_phs,
            'subs': subs
        })

    return results

def run():
    config = load_config()
    phrases, labels = load_phrases(config['CLM_PHRASES_PATH'])
    tokenizer = AutoTokenizer.from_pretrained(config['CLM_MODEL_NAME'], token=config['HF_TOKEN'])
    model = load_clm_model(config['CLM_MODEL_NAME'], method=config['LOAD_MODEL_METHOD'], token=config['HF_TOKEN'])

    # Silence tokenization prints
    tokenizer.add_special_tokens({'sep_token':'<SEP>', 'pad_token':'<PAD>', 'cls_token':'<CLS>', 'mask_token':'<MASK>'})
    # embbeding = model.resize_token_embeddings(len(tokenizer))
    # print(len(tokenizer))
    # print(len(embbeding.embedding_dim))

    tokenizer.use_default_system_prompt = False

    data = make_dataset(pd.DataFrame(load_dataset(config['DATASET_NAME'])['train'])).to_dict(orient='records')
    # data = data[:2]

    results = []

    model.eval()

    softmax = torch.nn.Softmax(dim=1)
    label_ids = label_tokenize(tokenizer, labels, return_tensors='pt').to(model.device)

    with torch.no_grad():
        for i in range(int(config['CLM_N_STEPS'])):
            torch.cuda.empty_cache()
            print(f'##### Starting epoch: {i+1} #####')
            results.append(epoch(tokenizer, model, data, phrases, softmax, label_ids))
            write_jsonl(RESULT_PATH, results)

if __name__ == "__main__":
    run()