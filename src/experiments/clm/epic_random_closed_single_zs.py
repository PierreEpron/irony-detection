from src.utils import load_config
from src.model import clm_run

if __name__ == "__main__":
    config = load_config()
    config = config | {
        'RESULT_PATH':"results/epic_random_closed_single_zs.jsonl",
        'CLM_PHRASES_PATH':"src/prompts/single_phrases.json"
    }
    clm_run(config)