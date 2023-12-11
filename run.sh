export HF_HOME="/srv/storage/idmctal@storage1.nancy.grid5000.fr/2023/m2/pepron/.cache/huggingface"
export HF_DATASETS_CACHE="/srv/storage/idmctal@storage1.nancy.grid5000.fr/2023/m2/pepron/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/srv/storage/idmctal@storage1.nancy.grid5000.fr/2023/m2/pepron/.cache/huggingface/models"

cd ${HOME}/irony-detection/
nvcc --version
source venv/bin/activate
python3 -m src.experiments.$1