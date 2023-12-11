from src.utils import load_config
from src.model import cls_run
from src.model import cls_load_tweeteval
from src.tokenizer import cls_single_tokenize
from src.training import MCC_Loss


if __name__ == "__main__":
    config = load_config()
    config = config | {
        'CLS_MODEL_NAME':'cardiffnlp/twitter-roberta-large-2022-154m',
        'LOSS_FUNCS': [
            (MCC_Loss(), 1), 
        ],
    }

    for bs in [16, 32, 48]:
        for lr in [1e-3, 1e-4, 1e-5]:    
            lr_n = f'{lr:e}'[-1]
            path = f"results/tweeteval_{bs}_{lr_n}_mcc"
            config = config | {
                'OUTPUT_DIR':path, 
                'RESULT_PATH':f"{path}.jsonl",
            }
            cls_run(config, load_data_func=cls_load_tweeteval, tokenize_func=cls_single_tokenize)

