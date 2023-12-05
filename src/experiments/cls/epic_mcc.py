from src.utils import load_config
from src.model import run
from src.training import MCC_Loss
 

OUTPUT_DIR = "results/roberta_irony_mcc"
RESULT_PATH = "results/roberta_irony_mcc.jsonl"

if __name__ == "__main__":
    config = load_config()
    config = config | {
        'OUTPUT_DIR':"results/roberta_irony_mcc", 
        'RESULT_PATH':"results/roberta_irony_mcc.jsonl",
        'LOSS_FUNCS': [
            (MCC_Loss(), 1), 
        ],
    }
    run(config)