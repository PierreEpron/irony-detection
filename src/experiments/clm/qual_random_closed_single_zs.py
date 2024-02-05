from pathlib import Path
from src.utils import load_config
from src.model import clm_load_tweeteval, clm_run

def load_qual(config):
    return [{'id_original':i, 'text':l, 'label':0} for i, l in enumerate(Path('data/qualitative.txt').read_text().splitlines())]

if __name__ == "__main__":
    config = load_config()

    config = config | {
        'RESULT_PATH':"results/qual.jsonl",
        'CLM_PHRASES_PATH':"src/prompts/single_phrases.json"
    }
    clm_run(config, load_data_func=load_qual)