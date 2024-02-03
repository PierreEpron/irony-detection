from pathlib import Path
import torch
import shutil
import json
import sys

from src.utils import write_jsonl

prefix = sys.argv[1]

home_path = Path.home()
repo_path = home_path / 'irony-detection'
results_path = repo_path / 'results' / prefix

for k in range(5):
    k_path = results_path / str(k)
    log_path = list(k_path.glob('cv_logs/*/version_*/metrics.csv'))
    assert len(log_path) == 1, f"Multiple or no version found for log_path: {log_path}"
    log_path = log_path[0]

    predictions = []
    for predictions_path in k_path.glob('predictions_*.pt'):
        predictions += torch.load(predictions_path)
    
    write_jsonl(home_path / f'{prefix}_predictions_{k}.jsonl', predictions)
    shutil.copyfile(k_path / 'monitoring.json', home_path / f'{prefix}_monitoring_{k}.json')
    shutil.copyfile(log_path, home_path / f'{prefix}_logs_{k}.json')

# for path in results_path.glob(f'{prefix}*'):
#     if path.is_dir():
#         ts_path = path / 'trainer_state.json'
#         cp_path = repo_path / json.loads(ts_path.read_text())['best_model_checkpoint']

#         if not cp_path.is_dir():
#             continue

#         jsonl_path = Path(str(path)[:-2] + '.jsonl')