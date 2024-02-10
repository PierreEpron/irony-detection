# irony-detection

## Prompt ccl

- yes/no work better than true/false
- "you are a helpful assistant" is the best system prompt atm
- Putting input sentence after newlines seems to work better
- No clear difference on question
-

## G5K procedure

lille: oarsub -p chifflot -l "gpu=1, walltime=4:00:00" "source run.sh"
lyon: oarsub -t exotic -p gemini -l "gpu=1, walltime=4:00:00" "source run.sh" 

oarsub -t exotic -p grouille-1 -l "gpu=1, walltime=10:00:00" "source nrun.sh cls.tweeteval_bce_calibration"


```
# clone repo
git clone https://github.com/PierreEpron/irony-detection.git

# create python venv
cd irony-detection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install correct version of torch (depending on cuda https://pytorch.org/get-started/previous-versions/)
pip uninstall torch
pip install ...

# add .env file with HF_TOKEN

# check diskspace usage
cd .. # cd to your root folder
du -s -h


# nancy gpu
oarsub -t exotic -p grouille -l "gpu=1, walltime=4:00:00" "source run.sh {mode}"
# genoble gpu 
oarsub -t exotic -p drac -l "gpu=1, walltime=4:00:00" "source run.sh {mode}"

oarstat -fj {job_id}
oardel {job_id}


# If u need space delete pip cache
cd .. # cd to your root folder
rm -r .cache/pip

```