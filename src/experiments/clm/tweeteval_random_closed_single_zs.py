from src.utils import load_config
from src.model import clm_load_tweeteval, clm_run

if __name__ == "__main__":
    config = load_config()
    config = config | {
        'RESULT_PATH':"results/tweeteval_random_closed_zs.jsonl",
        'CLM_PHRASES_PATH':"src/prompts/single_phrases.json"
    }
    clm_run(config, load_data_func=clm_load_tweeteval)