from pathlib import Path
import shutil
import json
import sys

prefix = sys.argv[1]

home_path = Path.home()
repo_path = home_path / 'irony-detection'
results_path = repo_path / 'results'

def get_new_path(path):
    return Path('/'.join(path.parts[:2] + path.parts[:3]))

for path in results_path.glob(f'{prefix}*'):
    if path.is_dir():
        ts_path = path / 'trainer_state.json'
        cp_path = repo_path / json.loads(ts_path.read_text())['best_model_checkpoint']
        jsonl_path = path.with_suffix('jsonl')
        shutil.copyfile(ts_path, get_new_path(ts_path))
        shutil.copytree(cp_path, get_new_path(cp_path))
        shutil.copyfile(jsonl_path, get_new_path(jsonl_path))
        shutil.rmtree(path)