from src.utils import load_config
from src.model import run
from src.model import cls_load_tweeteval, cls_single_tokenize
from src.training import MCC_Loss


if __name__ == "__main__":
    config = load_config()
    config = config | {
        'OUTPUT_DIR':"results/tweeteval_mcc", 
        'RESULT_PATH':"results/tweeteval_mcc.jsonl",
        'LOSS_FUNCS': [
            (MCC_Loss(), 1), 
        ],
    }
    run(config, load_data_func=cls_load_tweeteval, tokenize_func=cls_single_tokenize)