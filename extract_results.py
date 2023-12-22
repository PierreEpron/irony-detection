from pathlib import Path
import shutil
import json
import sys

prefix = sys.argv[1]

home_path = Path.home()
repo_path = home_path / 'irony-detection'
results_path = repo_path / 'results'

for path in results_path.glob(f'{prefix}*'):
    if path.is_dir():
        ts_path = path / 'trainer_state.json'
        if not ts_path.is_file():
            continue
        cp_path = repo_path / json.loads(ts_path.read_text())['best_model_checkpoint']
        jsonl_path = Path(str(path)[:-2] + '.jsonl')
        shutil.copyfile(ts_path, home_path / f'{ts_path.parts[-2]}_state.json')
        shutil.copyfile(jsonl_path, home_path / jsonl_path.parts[-1])

        shutil.copytree(cp_path, home_path / cp_path.parts[-2])
        shutil.rmtree(path)
        shutil.copytree(home_path / cp_path.parts[-2], results_path / cp_path.parts[-2])
        shutil.rmtree(home_path / cp_path.parts[-2])