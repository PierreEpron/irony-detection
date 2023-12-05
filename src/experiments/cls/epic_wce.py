from src.utils import load_config
from src.model import run

import torch 

if __name__ == "__main__":

    config = load_config()

    # Can't use the device from the model so going back to old school way
    device = torch.device('cuda' if config['LOAD_MODEL_METHOD'] == 'cuda' else 'cpu')
    label_weights = torch.Tensor([.314, .686]).to(device)

    config = config | {
        'OUTPUT_DIR':"results/epic_wce", 
        'RESULT_PATH':"results/epic_wce.jsonl",
        'LOSS_FUNCS': [
            (torch.nn.CrossEntropyLoss(label_weights), 1),
        ],
    }
    
    run(config)

   