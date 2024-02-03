from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks import BasePredictionWriter
from dotenv import dotenv_values
from pathlib import Path
import torch
import time
import json
import os


class MonitoringMetrics:
    '''
        Utility class to record and save Monitoring metrics.
    '''
    def __init__(self) -> None:       
        self.t = time.time()
        self.records = {
            'times': {},
            'sizes': {}
        }

    def set_time(self, key):
        ''' Use to record time elapsed since last set_time. Record it the given key'''
        t = time.time()
        self.records['times'][key] = t - self.t
        self.t = t 

    def set_size(self, key, value):
        ''' Use to record size. Record it the given key'''
        self.records['sizes'][key] = value
        
    def save(self, result_path):
        ''' use to save recors in json file'''
        result_path = Path(result_path) if isinstance(result_path, str) else result_path
        result_path.write_text(json.dumps(self.records))

def get_plt_loggers(result_path, name):
    '''
        Shortcut to get plt loggers
    '''

    if not result_path.is_dir():
        result_path.mkdir()

    # tb_logger = TensorBoardLogger(result_path / "tb_logs", name=name)
    csv_logger = CSVLogger(result_path / "cv_logs", name=name)
    return [csv_logger]


class CustomWriter(BasePredictionWriter):
    '''
        Used to save predictions with a ddp strategy
        From : https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.BasePredictionWriter.html#lightning.pytorch.callbacks.BasePredictionWriter
    '''
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        torch.save(predictions, os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt"))


def read_jsonl(path, encoding='utf-8'):
    """
        Shortcut for read jsonl file

        Parameters
        ----------
        path : str or Path, path of file to read.
        encoding : str, default='utf-8', encoding format to use.
    """
    path = Path(path) if isinstance(path, str) else path
    return [json.loads(line) for line in path.read_text(encoding=encoding).strip().split('\n')]


def write_jsonl(path, data, encoding='utf-8'):
    """
        Shortcut for write jsonl file

        Parameters
        ----------
        path : str or Path, path of file to write.
        data : List, list of json data to write.
        encoding : str, default='utf-8', encoding format to use.
    """
    path = Path(path) if isinstance(path, str) else path
    path.write_text('\n'.join([json.dumps(item) for item in data]), encoding=encoding)


def load_config(path='.env'):
    return {
        **dotenv_values(path),  # load sensitive variables
        **os.environ,  # override loaded values with environment variables
    }


def find_closest(num, collection):
    ''' 
        Find closest integer from a list of integer
        from https://stackoverflow.com/a/12141215
    '''
    return min(collection, key=lambda x:abs(x-num))