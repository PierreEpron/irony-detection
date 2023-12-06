from src.utils import load_config
from src.model import clm_run

if __name__ == "__main__":
    config = load_config()
    config = config | {
        'RESULT_PATH':"results/epic_random_closed_double_zs.jsonl",
        'CLM_PHRASES_PATH':"src/prompts/double_phrases.json"
    }
    clm_run(config)