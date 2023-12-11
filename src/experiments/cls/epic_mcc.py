from src.utils import load_config
from src.model import cls_run
from src.training import MCC_Loss

if __name__ == "__main__":
    config = load_config()
    config = config | {
        'OUTPUT_DIR':"results/epic_mcc", 
        'RESULT_PATH':"results/epic_mcc.jsonl",
        'LOSS_FUNCS': [
            (MCC_Loss(), 1), 
        ],
    }
    cls_run(config)