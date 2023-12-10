from src.utils import load_config
from src.model import cls_run
from src.model import cls_load_tweeteval
from src.tokenizer import cls_single_tokenize

import torch

if __name__ == "__main__":
    config = load_config()

    # Can't use the device from the model so going back to old school way
    # device = torch.device('cuda' if config['LOAD_MODEL_METHOD'] == 'cuda' else 'cpu')
    # label_weights = torch.Tensor([[.314, .686]] * 32).to(device)

    config = config | {
        'OUTPUT_DIR':"results/tweeteval_wce", 
        'RESULT_PATH':"results/tweeteval_wce.jsonl",
        'LOSS_FUNCS': [
            (torch.nn.BCELoss(), 1),
        ],
    }
    
    cls_run(config, load_data_func=cls_load_tweeteval, tokenize_func=cls_single_tokenize)