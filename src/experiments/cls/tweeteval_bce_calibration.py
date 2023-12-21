import torch
from src.utils import load_config
from src.model import cls_run
from src.model import cls_load_tweeteval
from src.tokenizer import cls_single_tokenize


if __name__ == "__main__":
    config = load_config()
    config = config | {
        'CLS_MODEL_NAME':'cardiffnlp/twitter-roberta-large-2022-154m',
        'LOSS_FUNCS': [
            (torch.nn.BCELoss(), 1),
        ],
    }

    for bs in [16, 32, 48, 64]:
        for lr in [5e-5, 1e-5, 5e-6]:    
            lr_n = f'{lr:e}'
            path = f"results/tweeteval_{bs}_{lr_n}_bce"
            config = config | {
                'OUTPUT_DIR':path, 
                'RESULT_PATH':f"{path}.jsonl",
                'CLS_BATCH_SIZE':bs,
                'CLS_LR':lr,
                'CLS_EPOCHS':30
            }
            cls_run(config, load_data_func=cls_load_tweeteval, tokenize_func=cls_single_tokenize)

